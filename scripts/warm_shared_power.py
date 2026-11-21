#!/usr/bin/env python3
"""Fit one shared MLP for power curves and integrated energy on the warm split."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date
import hashlib
from inspect import getsource
import json
import multiprocessing
from pathlib import Path
import pickle
from time import monotonic

import numpy as np
import pandas as pd

from independent_t2 import POWER_HOURS, SOURCES, bracket, interpolate
from score_forecasts import MAX_PREDICTION, score_track


FROZEN_SPLIT = 'ac5708425e3ed19d9975732a9d3aac965cc95d6a047d0c49862422b45fbbb837'
COUNTS = {'train': 586, 'development': 42, 'sealed': 92}
POWER_CAP = MAX_PREDICTION['T1']
SEED = 20261121


def manifest_days(path):
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != FROZEN_SPLIT:
        raise ValueError(f'Unexpected warm split SHA256 {actual_hash}.')
    frame = pd.read_csv(path, dtype=str)
    if frame['role'].value_counts().to_dict() != COUNTS or frame['date'].duplicated().any():
        raise ValueError('Invalid warm split roles or date keys.')
    return {
        role: [date.fromisoformat(day) for day in frame.loc[frame['role'].eq(role), 'date']]
        for role in ('train', 'development')
    }


def sample_rectangle(variable, row_indices, col_indices, weights, time_indices):
    if tuple(variable.dimensions) != ('time', 'lat', 'lon'):
        raise ValueError(f'Unexpected dimensions for {variable.name}: {variable.dimensions}.')
    south, north = row_indices
    west, east = col_indices
    lat_weight, lon_weight = weights
    row_start = int(min(south.min(), north.min()))
    row_end = int(max(south.max(), north.max())) + 1
    col_start = int(min(west.min(), east.min()))
    col_end = int(max(west.max(), east.max())) + 1
    # One contiguous netCDF read, then gather station cells from the in-memory rectangle.
    block = np.ma.asarray(variable[:, row_start:row_end, col_start:col_end], dtype=np.float32).filled(np.nan)
    south, north = south - row_start, north - row_start
    west, east = west - col_start, east - col_start
    lower = interpolate(block[:, south, west], block[:, south, east], lon_weight)
    upper = interpolate(block[:, north, west], block[:, north, east], lon_weight)
    spatial = interpolate(lower, upper, lat_weight)
    first, second, time_weight = time_indices
    return interpolate(spatial[first], spatial[second], time_weight[:, None]).T


def daily_summary(values):
    mean = np.full((values.shape[0], 1), np.nan, dtype=np.float32)
    upper = mean.copy()
    for station, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if finite.size:
            mean[station, 0] = finite.mean()
            upper[station, 0] = np.quantile(finite, 0.9)
    return [np.broadcast_to(stat, values.shape) for stat in (mean, upper)]


def weather_features(folder, coordinates, dataset_factory):
    fields, names, speeds = [], [], []
    for prefix, (filename, expected, _) in SOURCES.items():
        path = folder / 'nwp' / filename
        with dataset_factory(path, mode='r') as ds:
            times, lat, lon = (np.ma.asarray(ds.variables[key][:], dtype=float).filled(np.nan)
                               for key in ('time', 'lat', 'lon'))
            if not np.array_equal(times, expected):
                raise ValueError(f'{path} forecast times differ from the specification.')
            south, north, lat_weight = bracket(lat[::-1], coordinates[:, 0])
            rows = len(lat) - 1 - south, len(lat) - 1 - north
            west, east, lon_weight = bracket(lon, coordinates[:, 1])
            time_indices = bracket(times, POWER_HOURS)
            keys = ['u100', 'v100', 't2m', 'sp']
            if prefix == 'tianji':
                keys += ['ws100']
            sampled = {key: sample_rectangle(ds.variables[key], rows, (west, east), (lat_weight, lon_weight), time_indices)
                       for key in keys}
        speed = np.hypot(sampled['u100'], sampled['v100'])
        power_density = 0.5 * sampled['sp'] * speed**3 / (287.05 * np.where(sampled['t2m'] > 0, sampled['t2m'], np.nan))
        mean, upper = daily_summary(speed)
        sampled.update({'speed100': speed, 'density_x_v_cubed_half': power_density,
                        'daily_speed_mean': mean, 'daily_speed_p90': upper})
        fields.extend(sampled.values())
        names.extend(f'{prefix}_{name}' for name in sampled)
        speeds.append(speed)
    fields.extend([np.abs(speeds[0] - speeds[1]), np.abs(speeds[0] - speeds[2])])
    names.extend(['tianji_ec_speed_difference', 'tianji_aifs_speed_difference'])
    angle = 2 * np.pi * np.arange(len(POWER_HOURS)) / len(POWER_HOURS)
    fields.extend([np.broadcast_to(np.sin(angle), speeds[0].shape), np.broadcast_to(np.cos(angle), speeds[0].shape)])
    names.extend(['time_of_day_sin', 'time_of_day_cos'])
    for index, name in enumerate(('latitude', 'longitude')):
        fields.append(np.broadcast_to(coordinates[:, index, None], speeds[0].shape))
        names.append(name)
    return np.stack(fields, axis=-1).astype(np.float32), names


def extract_or_read(task, dataset_factory=None):
    folder, coordinates, cache = task
    fingerprint = []
    for filename, _, _ in SOURCES.values():
        stat = (folder / 'nwp' / filename).stat()
        fingerprint.extend((stat.st_size, stat.st_mtime_ns))
    tag = hashlib.sha256(repr(fingerprint).encode()).hexdigest()[:16]
    path = cache / f'{folder.name}-{tag}.npz'
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            return saved['features'], saved['columns'].tolist(), True
    if dataset_factory is None:
        from netCDF4 import Dataset

        dataset_factory = Dataset
    features, columns = weather_features(folder, coordinates, dataset_factory)
    pending = path.with_suffix('.tmp.npz')
    np.savez(pending, features=features, columns=columns)
    pending.replace(path)
    return features, columns, False


def fetch_features(root, train_days, dev_days, roster, coordinates, cache_root, workers, progress, dataset_factory=None):
    extraction_code = ''.join(getsource(func) for func in
                              (bracket, interpolate, sample_rectangle, daily_summary, weather_features, extract_or_read))
    source_contract = {name: (filename, times.tolist()) for name, (filename, times, _) in SOURCES.items()}
    cache_fingerprint = hashlib.sha256(extraction_code.encode() + coordinates.tobytes() + POWER_HOURS.tobytes() +
                                       json.dumps([roster, source_contract, str(root.resolve())]).encode()).hexdigest()[:20]
    cache = cache_root / cache_fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    tasks = [(root / 'train' / f'{day:%Y%m%d}_0000', coordinates, cache) for day in [*train_days, *dev_days]]

    def read_results(results):
        pieces = []
        cache_hits = 0
        columns = None
        for index, (features, feature_columns, hit) in enumerate(results, 1):
            if columns is None:
                columns = feature_columns
            if columns != feature_columns or features.shape != (len(roster), 96, len(columns)):
                raise ValueError('Unexpected feature cache dimensions or column order.')
            pieces.append(features)
            cache_hits += hit
            if index == 1 or index % 25 == 0 or index == len(tasks):
                progress(f'Weather {index}/{len(tasks)} days, cache hits {cache_hits}')
        return np.concatenate(pieces), columns, cache_hits

    if workers == 1 or dataset_factory is not None:
        result = read_results(extract_or_read(task, dataset_factory) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            result = read_results(pool.map(extract_or_read, tasks, chunksize=4))
    return (*result, cache_fingerprint)


def labels(root, days, roster):
    indexed = []
    expected_keys = pd.MultiIndex.from_product([roster, range(1, 97)], names=['station_id', 'point_index'])
    for day in days:
        frame = pd.read_csv(root / 'train' / f'{day:%Y%m%d}_0000' / 'power_target.csv', dtype={'station_id': str, 'date': str},
                            usecols=['station_id', 'date', 'point_index', 'power', 'power_limit'])
        frame = frame.loc[frame['station_id'].isin(roster)].set_index(['station_id', 'point_index'])
        if (not frame.index.is_unique or len(frame) != len(expected_keys) or
                not expected_keys.isin(frame.index).all() or not frame['date'].eq(day.isoformat()).all()):
            raise ValueError(f'Missing, duplicate or wrong-date labels on {day}.')
        indexed.append(frame.reindex(expected_keys).reset_index())
    return pd.concat(indexed, ignore_index=True)


def normalize(train, development):
    finite = np.isfinite(train)
    count = finite.sum(axis=(0, 1))
    safe_values = np.where(finite, train, 0)
    means = np.divide(safe_values.sum(axis=(0, 1), dtype=np.float64), count,
                      out=np.zeros(train.shape[-1]), where=count > 0)
    variance = np.divide(np.where(finite, (train - means)**2, 0).sum(axis=(0, 1)), count,
                         out=np.ones(train.shape[-1]), where=count > 0)
    scales = np.sqrt(np.maximum(variance, 1e-8))
    def transform(array):
        available = np.isfinite(array)
        values = np.where(available, (array - means) / scales, 0)
        return np.concatenate([values, ~available], axis=-1).astype(np.float32)

    return transform(train), transform(development), means, scales


def fit_power(train, target, valid, devel, epochs, batch, device_name, progress, *, seed=SEED):
    import torch
    from torch import nn

    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('PyTorch CUDA is unavailable. Use --device cpu only for a deliberate CPU run.')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    network = nn.Sequential(nn.Linear(train.shape[-1], 96), nn.LayerNorm(96), nn.SiLU(),
                            nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 1)).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0015, weight_decay=0.001)
    keep = valid.any(axis=1)
    if not np.any(keep):
        raise ValueError('No training days with valid points.')
    features = torch.as_tensor(train[keep], device=device)
    truth = torch.as_tensor(np.where(valid, target / POWER_CAP, 0)[keep], dtype=torch.float32, device=device)
    mask = torch.as_tensor(valid[keep], device=device)
    progress(f'Model and training arrays on {features.device}, stations-days per batch {batch}, dtype {features.dtype}')
    trajectory = []
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(features), device=device)
        loss_sum = torch.zeros((), device=device)
        for ids in permutation.split(batch):
            prediction = torch.sigmoid(network(features[ids]).squeeze(-1))
            diff_sq = ((prediction - truth[ids])**2 * mask[ids]).sum(dim=1)
            loss = torch.sqrt(diff_sq / mask[ids].sum(dim=1) + 1e-8).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(ids)
        rmse_mw = float(loss_sum.item() / len(features) * POWER_CAP)
        trajectory.append(rmse_mw)
        progress(f'Epoch {epoch}/{epochs}, train daily RMSE {rmse_mw:.4f} MW')
    train_memory = torch.cuda.max_memory_allocated() if device.type == 'cuda' else None

    def predict():
        network.eval()
        with torch.inference_mode():
            output = [torch.sigmoid(network(torch.as_tensor(part, device=device)).squeeze(-1)).cpu().numpy()
                      for part in np.array_split(devel, max(1, (len(devel) + batch - 1) // batch))]
        return np.clip(np.concatenate(output).astype(float) * POWER_CAP, 0, POWER_CAP)

    device_label = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    return network, predict, trajectory, device_label, train_memory


def load_roster(root):
    roster = sorted(pd.read_csv(root / 'submission/T1_dataset/submission_template.csv',
                                usecols=['station_id'], dtype={'station_id': str})['station_id'].unique().tolist())
    second_roster = set(pd.read_csv(root / 'submission/T2_dataset/submission_template.csv',
                                   usecols=['station_id'], dtype={'station_id': str})['station_id'])
    if len(roster) != 12 or set(roster) != second_roster:
        raise ValueError('The T1 and T2 templates must have the same 12-station roster.')
    return roster


def score_saved(root, dev_days, out, report):
    start = monotonic()
    if not (out / 'power_model.pt').is_file():
        raise ValueError('Expected the saved power_model.pt and both development prediction CSVs.')
    roster = load_roster(root)
    dtype = {'station_id': str, 'date': str}
    predictions = pd.read_csv(out / 'T1_development.csv', dtype=dtype)
    energy = pd.read_csv(out / 'T2_development.csv', dtype=dtype)
    keys = ['station_id', 'date']
    integrated = predictions.groupby(keys)['prediction'].sum() * 0.25
    energy_indexed = energy.set_index(keys)['energy']
    if not energy_indexed.index.is_unique:
        raise ValueError('Saved T2 forecasts contain duplicate station-days.')
    energy_aligned = energy_indexed.reindex(integrated.index)
    if len(energy) != len(integrated) or not np.allclose(integrated, energy_aligned, rtol=0, atol=1e-9):
        raise ValueError('Saved T2 forecasts do not equal the integral of the saved T1 curves.')
    # Read development labels only after both forecasts and the shared model exist.
    truth = labels(root, dev_days, roster)
    report['scores'] = {
        'T1': score_track('T1', truth, predictions, roster),
        'T2': score_track('T2', truth, energy, roster, allow_unscored_stations=True),
    }
    report.setdefault('phase_seconds', {})['scoring'] = monotonic() - start
    report['t2_local_policy'] = 'Average stations with eligible days; undefined scores are omitted and listed.'
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    staging = out / 'results.json.tmp'
    staging.write_text(payload)
    staging.replace(out / 'results.json')
    print(payload, end='', flush=True)


def checkpoint_devices(path):
    import torch

    locations = set()
    def remember_location(storage, location):
        locations.add(location)
        return storage

    torch.load(path, map_location=remember_location, weights_only=True)
    return sorted(locations)


def power_weather(root, train_days, forecast_days, cache_root, workers, progress, dataset_factory=None):
    roster = load_roster(root)
    stations = pd.read_csv(root / 'static/power_stations.csv', dtype={'station_id': str}).set_index('station_id')
    coordinates = stations.loc[roster, ['latitude', 'longitude']].to_numpy(dtype=float)
    return roster, fetch_features(
        root, train_days, forecast_days, roster, coordinates, cache_root, workers, progress, dataset_factory,
    )


def fit_forecasts(root, out, train_days, dev_days, workers, epochs, batch, device, *,
                  dataset_factory=None, cache_root=None, seed=SEED, forecast_tag='development'):
    start = monotonic()
    def progress(message):
        print(f'[{monotonic() - start:.1f}s] {message}', flush=True)

    roster, (weather, columns, cache_hits, cache_version) = power_weather(
        root, train_days, dev_days, cache_root or out / 'features', workers, progress, dataset_factory,
    )
    weather_seconds = monotonic() - start

    assembly_start = monotonic()
    cut = len(train_days) * len(roster)
    train, development, means, scales = normalize(weather[:cut], weather[cut:])
    del weather
    labeled = labels(root, train_days, roster)
    actual = labeled['power'].to_numpy(dtype=float).reshape(cut, 96)
    valid_mask = np.isfinite(actual) & labeled['power_limit'].eq(1).to_numpy().reshape(cut, 96)
    assembly_seconds = monotonic() - assembly_start

    train_start = monotonic()
    network, predict, trajectory, device_label, peak = fit_power(
        train, actual, valid_mask, development, epochs, batch, device, progress, seed=seed,
    )
    train_seconds = monotonic() - train_start

    import torch

    inference_start = monotonic()
    power = predict()
    days = pd.MultiIndex.from_product([[day.isoformat() for day in dev_days], roster], names=['date', 'station_id']).to_frame(index=False)
    predictions = days.loc[days.index.repeat(96)].reset_index(drop=True)
    predictions['point_index'] = np.tile(np.arange(1, 97), len(days))
    predictions['prediction'] = power.ravel()
    energy = days.assign(energy=0.25 * power.sum(axis=1))
    predictions.to_csv(out / f'T1_{forecast_tag}.csv', index=False)
    energy.to_csv(out / f'T2_{forecast_tag}.csv', index=False)
    torch.save({'model': network.state_dict(), 'feature_columns': columns, 'mean': means.tolist(),
                'scale': scales.tolist(), 'seed': seed}, out / 'power_model.pt')
    inference_seconds = monotonic() - inference_start

    report = {'data_root': str(root), 'split_sha256': FROZEN_SPLIT, 'train_days': len(train_days), 'development_days': len(dev_days),
              'seed': seed, 'epochs': epochs, 'train_valid_points': int(valid_mask.sum()), 'weather_features': columns,
              'train_daily_rmse_mw': trajectory, 'device': device_label, 'peak_training_gpu_bytes': peak,
              'device_requested': device, 'batch_station_days': batch,
              'feature_workers': workers, 'feature_cache_hits': cache_hits, 'feature_cache_version': cache_version,
              'phase_seconds': {'weather_and_cache': weather_seconds, 'assembly_and_labels': assembly_seconds,
                                'training': train_seconds, 'prediction_and_export': inference_seconds}}
    (out / 'training_receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def run(root, split, out, workers, epochs, batch, device, dataset_factory=None, cache_root=None):
    roles = manifest_days(split)
    train_days, dev_days = roles['train'], roles['development']
    report = fit_forecasts(root, out, train_days, dev_days, workers, epochs, batch, device,
                           dataset_factory=dataset_factory, cache_root=cache_root)
    score_saved(root, dev_days, out, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--split', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/warm_date_split_v1.csv')
    parser.add_argument('--out', type=Path, default=Path('/workspace/qhl/runs/warm_shared_power_v1'))
    parser.add_argument('--cache-root', type=Path, default=Path('/workspace/qhl/runs/warm_power_feature_cache'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch-station-days', type=int, default=64)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--predict-b', action='store_true', help='Fit on train+A and export B forecasts.')
    modes.add_argument('--prepare-b-weather', action='store_true', help='Fill the B weather cache in a NetCDF4 environment.')
    modes.add_argument('--score-only', action='store_true', help='Score frozen predictions in --out without extracting features or training.')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.seed is not None and not args.predict_b:
        parser.error('--seed is for B predictions only.')
    if args.score_only:
        days = manifest_days(args.split)['development']
        receipt = args.out / 'training_receipt.json'
        report = json.loads(receipt.read_text()) if receipt.is_file() else {
            'data_root': str(args.data_root), 'split_sha256': FROZEN_SPLIT, 'development_days': len(days),
            'training_metadata': 'Original run did not save a training receipt.'}
        if report['data_root'] != str(args.data_root) or report['split_sha256'] != FROZEN_SPLIT:
            parser.error('Results directory belongs to a different data root or split.')
        report['rescored_from_saved_predictions'] = True
        try:
            report['checkpoint_original_devices'] = checkpoint_devices(args.out / 'power_model.pt')
        except (OSError, RuntimeError, ValueError, pickle.UnpicklingError, EOFError) as err:
            report['checkpoint_device_probe_error'] = type(err).__name__
        score_saved(args.data_root, days, args.out, report)
        return
    if not 1 <= args.workers <= 16 or min(args.epochs, args.batch_station_days) < 1:
        parser.error('Use 1 to 16 feature workers and positive epochs and batch size.')
    if args.prepare_b_weather:
        from run_companion_warm_t1 import date_roles

        train, forecast = date_roles(args.split, 'B')
        power_weather(args.data_root, train, forecast, args.cache_root, args.workers,
                      lambda message: print(f'[T2 B 天气] {message}', flush=True))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'results.json').exists():
        parser.error('Results already exist. Choose a new output directory for a new candidate.')
    if args.device == 'cuda':
        import torch

        if not torch.cuda.is_available():
            parser.error('PyTorch cannot access CUDA in this environment.')
        torch.zeros(1, device='cuda').item()
        print(f'PyTorch CUDA verified: {torch.cuda.get_device_name(0)}; CPU feature workers: {args.workers}', flush=True)
    if args.predict_b:
        from run_companion_warm_t1 import B_SEEDS, date_roles

        if args.seed not in B_SEEDS:
            parser.error('--predict-b requires one of the registered B seeds.')
        train, forecast = date_roles(args.split, 'B')
        if args.epochs != 16 or args.batch_station_days != 64:
            parser.error('Use the original MLP schedule of 16 epochs and batches of 64 station-days.')
        fit_forecasts(args.data_root, args.out, train, forecast, args.workers, args.epochs,
                      args.batch_station_days, args.device, cache_root=args.cache_root,
                      seed=args.seed, forecast_tag='sealed')
        return
    run(args.data_root, args.split, args.out, args.workers, args.epochs, args.batch_station_days, args.device,
        cache_root=args.cache_root)


if __name__ == '__main__':
    main()
