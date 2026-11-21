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


def fit_power(train, target, valid, devel, epochs, batch, device_name, progress):
    import torch
    from torch import nn

    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('PyTorch CUDA is unavailable. Use --device cpu only for a deliberate CPU run.')
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    device = torch.device(device_name)
    network = nn.Sequential(nn.Linear(train.shape[-1], 96), nn.LayerNorm(96), nn.SiLU(),
                            nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 1)).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0015, weight_decay=0.001)
    keep = valid.any(axis=1)
    if not np.any(keep):
        raise ValueError('No training days with valid points.')
    features = torch.as_tensor(train[keep], device=device)
    truth = torch.as_tensor(np.where(valid, target / POWER_CAP, 0)[keep], dtype=torch.float32, device=device)
    mask = torch.as_tensor(valid[keep], device=device)
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


def run(root, split, out, workers, epochs, batch, device, dataset_factory=None, cache_root=None):
    roles = manifest_days(split)
    train_days, dev_days = roles['train'], roles['development']
    start = monotonic()
    def progress(message):
        print(f'[{monotonic() - start:.1f}s] {message}', flush=True)

    roster = sorted(pd.read_csv(root / 'submission/T1_dataset/submission_template.csv',
                                usecols=['station_id'], dtype={'station_id': str})['station_id'].unique().tolist())
    second_roster = set(pd.read_csv(root / 'submission/T2_dataset/submission_template.csv',
                                   usecols=['station_id'], dtype={'station_id': str})['station_id'])
    if len(roster) != 12 or set(roster) != second_roster:
        raise ValueError('The T1 and T2 templates must have the same 12-station roster.')
    stations = pd.read_csv(root / 'static/power_stations.csv', dtype={'station_id': str}).set_index('station_id')
    coordinates = stations.loc[roster, ['latitude', 'longitude']].to_numpy(dtype=float)
    weather, columns, cache_hits, cache_version = fetch_features(
        root, train_days, dev_days, roster, coordinates, cache_root or out / 'features', workers, progress, dataset_factory
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
    network, predict, trajectory, device_label, peak = fit_power(train, actual, valid_mask, development, epochs, batch, device, progress)
    train_seconds = monotonic() - train_start

    import torch

    inference_start = monotonic()
    power = predict()
    days = pd.MultiIndex.from_product([[day.isoformat() for day in dev_days], roster], names=['date', 'station_id']).to_frame(index=False)
    predictions = days.loc[days.index.repeat(96)].reset_index(drop=True)
    predictions['point_index'] = np.tile(np.arange(1, 97), len(days))
    predictions['prediction'] = power.ravel()
    energy = days.assign(energy=0.25 * power.sum(axis=1))
    predictions.to_csv(out / 'T1_development.csv', index=False)
    energy.to_csv(out / 'T2_development.csv', index=False)
    torch.save({'model': network.state_dict(), 'feature_columns': columns, 'mean': means.tolist(),
                'scale': scales.tolist(), 'seed': SEED}, out / 'power_model.pt')
    inference_seconds = monotonic() - inference_start

    score_start = monotonic()
    # Development labels are opened only after both predictions and the shared model are saved.
    truth = labels(root, dev_days, roster)
    scores = {'T1': score_track('T1', truth, predictions, roster), 'T2': score_track('T2', truth, energy, roster)}
    report = {'data_root': str(root), 'split_sha256': FROZEN_SPLIT, 'train_days': len(train_days), 'development_days': len(dev_days),
              'seed': SEED, 'epochs': epochs, 'train_valid_points': int(valid_mask.sum()), 'weather_features': columns,
              'train_daily_rmse_mw': trajectory, 'device': device_label, 'peak_training_gpu_bytes': peak,
              'feature_workers': workers, 'feature_cache_hits': cache_hits, 'feature_cache_version': cache_version,
              'phase_seconds': {'weather_and_cache': weather_seconds, 'assembly_and_labels': assembly_seconds,
                                'training': train_seconds, 'prediction_and_export': inference_seconds,
                                'scoring': monotonic() - score_start}, 'scores': scores}
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    (out / 'results.json').write_text(payload)
    print(payload, end='', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--split', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/warm_date_split_v1.csv')
    parser.add_argument('--out', type=Path, default=Path('/workspace/qhl/runs/warm_shared_power_v1'))
    parser.add_argument('--cache-root', type=Path, default=Path('/workspace/qhl/runs/warm_power_feature_cache'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch-station-days', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or min(args.epochs, args.batch_station_days) < 1:
        parser.error('Use 1 to 8 feature workers and positive epochs and batch size.')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'results.json').exists():
        parser.error('Results already exist. Choose a new output directory for a new candidate.')
    if args.device == 'cuda':
        import torch

        if not torch.cuda.is_available():
            parser.error('PyTorch cannot access CUDA in this environment.')
        torch.zeros(1, device='cuda').item()
    run(args.data_root, args.split, args.out, args.workers, args.epochs, args.batch_station_days, args.device,
        cache_root=args.cache_root)


if __name__ == '__main__':
    main()
