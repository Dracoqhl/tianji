#!/usr/bin/env python3
"""Fit an independent T2 objective comparison on the fixed local date split."""

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from score_forecasts import MAX_PREDICTION, MIN_ENERGY_MWH, MIN_VALID_POINTS, score_track


TRAIN_START = date(2024, 5, 1)
INNER_START = date(2025, 11, 1)
VALIDATION_START = date(2026, 1, 21)
VALIDATION_END = date(2026, 4, 20)
REFERENCE_T2 = 69.9242
SOURCES = {
    'tianji': ('tianji_0p125.nc', np.arange(33), ('u10', 'v10', 'ws100')),
    'ec': ('ec_hres_0p10.nc', np.arange(33), ()),
    'aifs': ('aifs_0p25.nc', np.arange(0, 31, 6), ('u10', 'v10')),
}
POWER_HOURS = 4 + np.arange(96) / 4
KEYS = ['station_id', 'date']
MODES = ('log_mse', 'relative_mse')


def bracket(axis, points):
    if (len(axis) < 2 or not np.isfinite(axis).all() or not np.isfinite(points).all() or
            np.any(np.diff(axis) <= 0) or np.any(points < axis[0]) or np.any(points > axis[-1])):
        raise ValueError('Coordinates must be increasing and cover all requested points.')
    higher = np.clip(np.searchsorted(axis, points), 1, len(axis) - 1)
    lower = higher - 1
    fraction = (points - axis[lower]) / (axis[higher] - axis[lower])
    return lower, higher, fraction


def interpolate(left, right, fraction):
    mixed = (1 - fraction) * left + fraction * right
    return np.where(fraction == 0, left, np.where(fraction == 1, right, mixed))


def station_sampling(latitudes, longitudes, coordinates):
    if len(latitudes) < 2 or len(longitudes) < 2 or not np.all(np.diff(latitudes) < 0):
        raise ValueError('Expected decreasing latitude with at least two cells per axis.')
    south, north, lat_fraction = bracket(latitudes[::-1], coordinates[:, 0])
    south, north = len(latitudes) - 1 - south, len(latitudes) - 1 - north
    west, east, lon_fraction = bracket(longitudes, coordinates[:, 1])
    rows, row_lookup = np.unique(np.concatenate([south, north]), return_inverse=True)
    cols, col_lookup = np.unique(np.concatenate([west, east]), return_inverse=True)
    return rows, cols, np.split(row_lookup, 2), np.split(col_lookup, 2), lat_fraction, lon_fraction


def sample_variable(variable, sampling, time_bracket):
    if tuple(variable.dimensions) != ('time', 'lat', 'lon'):
        raise ValueError(f'Unexpected dimensions for {variable.name}: {variable.dimensions}.')
    rows, cols, (south, north), (west, east), lat_fraction, lon_fraction = sampling
    values = np.ma.asarray(variable[:, rows, cols], dtype=float).filled(np.nan)
    south_values = interpolate(values[:, south, west], values[:, south, east], lon_fraction)
    north_values = interpolate(values[:, north, west], values[:, north, east], lon_fraction)
    spatial = interpolate(south_values, north_values, lat_fraction)
    before, after, fraction = time_bracket
    return interpolate(spatial[before], spatial[after], fraction[:, None])


def add_statistics(features, name, time_station):
    rows = []
    for column in time_station.T:
        valid = column[np.isfinite(column)]
        if valid.size:
            rows.append([float(valid.mean()), float(valid.std()), *np.quantile(valid, [0.1, 0.5, 0.9]),
                         valid.size / len(column)])
        else:
            rows.append([np.nan] * 5 + [0.0])
    for label, values in zip(('mean', 'sd', 'p10', 'p50', 'p90', 'coverage'), np.asarray(rows).T):
        features[f'{name}_{label}'] = values


def weather_features(folder, stations, coordinates, dataset_factory):
    columns = {'station_id': stations, 'latitude': coordinates[:, 0], 'longitude': coordinates[:, 1]}
    speeds = []
    for name, (filename, expected_times, extras) in SOURCES.items():
        path = folder / 'nwp' / filename
        with dataset_factory(path, mode='r') as dataset:
            axis = {key: np.ma.asarray(dataset.variables[key][:], dtype=float).filled(np.nan)
                    for key in ('time', 'lat', 'lon')}
            if not np.array_equal(axis['time'], expected_times):
                raise ValueError(f'{path} has unexpected forecast hours.')
            sampling = station_sampling(axis['lat'], axis['lon'], coordinates)
            time_bracket = bracket(axis['time'], POWER_HOURS)
            fields = {key: sample_variable(dataset.variables[key], sampling, time_bracket)
                      for key in ('u100', 'v100', 'sp', 't2m', *extras)}
        speed = np.hypot(fields['u100'], fields['v100'])
        speeds.append(speed)
        add_statistics(columns, f'{name}_wind', speed)
        density = fields['sp'] / (287.05 * np.where(fields['t2m'] > 0, fields['t2m'], np.nan))
        add_statistics(columns, f'{name}_power_density', 0.5 * density * speed**3)
        for field in ('u100', 'v100', 't2m', 'ws100'):
            if field in fields:
                add_statistics(columns, f'{name}_{field}', fields[field])
        if 'u10' in fields:
            shear = speed - np.hypot(fields['u10'], fields['v10'])
            add_statistics(columns, f'{name}_shear', shear)
        add_statistics(columns, f'{name}_wind_change', np.abs(np.diff(speed, axis=0)))
        # Fixed bins summarize the shape of the forecast wind distribution over the day.
        for threshold in (4, 7, 10):
            finite = np.isfinite(speed)
            covered = finite.sum(axis=0)
            fraction = np.divide(((speed >= threshold) & finite).sum(axis=0), covered,
                                 out=np.full(len(stations), np.nan), where=covered > 0)
            columns[f'{name}_hour_fraction_wind_ge_{threshold}'] = fraction
    for first, second, left, right in [('tianji', 'ec', speeds[0], speeds[1]),
                                       ('tianji', 'aifs', speeds[0], speeds[2]),
                                       ('ec', 'aifs', speeds[1], speeds[2])]:
        add_statistics(columns, f'{first}_{second}_wind_gap', np.abs(left - right))
    return pd.DataFrame(columns)


def validation_folders(root):
    folders = []
    for offset in range((VALIDATION_END - TRAIN_START).days + 1):
        day = TRAIN_START + timedelta(days=offset)
        folder = root / 'train' / f'{day:%Y%m%d}_0000'
        if not folder.is_dir():
            raise ValueError(f'Missing fixed-split directory: {folder}')
        folders.append((day.isoformat(), folder))
    return folders


def feature_cache(root, folder_list, station_table, out, dataset_factory, progress):
    coordinates = station_table[['latitude', 'longitude']].to_numpy(dtype=float)
    stations = station_table['station_id'].tolist()
    version = hashlib.sha256(Path(__file__).read_bytes() + coordinates.tobytes() +
                             json.dumps(stations).encode() + str(root.resolve()).encode()).hexdigest()[:20]
    cache = out / 'features' / version
    cache.mkdir(parents=True, exist_ok=True)
    pieces = []
    misses = 0
    for number, (day, folder) in enumerate(folder_list, 1):
        signature = []
        for filename, _, _ in SOURCES.values():
            stat = (folder / 'nwp' / filename).stat()
            signature.extend((stat.st_size, stat.st_mtime_ns))
        fingerprint = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
        path = cache / f'{day}-{fingerprint}.csv'
        if path.is_file():
            frame = pd.read_csv(path, dtype={'station_id': str, 'date': str}, float_precision='round_trip')
        else:
            frame = weather_features(folder, stations, coordinates, dataset_factory)
            frame['date'] = day
            staging = path.with_suffix('.tmp')
            frame.to_csv(staging, index=False)
            staging.replace(path)
            misses += 1
        pieces.append(frame)
        if number == 1 or number % 30 == 0 or number == len(folder_list):
            progress(f'Features {number}/{len(folder_list)}, extracted {misses}, cached {number - misses}')
    return pd.concat(pieces, ignore_index=True), version


def load_daily_targets(paths, stations):
    frames = []
    for _, folder in paths:
        frame = pd.read_csv(folder / 'power_target.csv', dtype={'station_id': str, 'date': str},
                            usecols=['station_id', 'date', 'point_index', 'power', 'power_limit'])
        frames.append(frame.loc[frame['station_id'].isin(stations)])
    points = pd.concat(frames, ignore_index=True)
    valid = points['power_limit'].eq(1) & np.isfinite(points['power'].to_numpy(dtype=float))
    daily = points.loc[valid].groupby(KEYS, sort=False)['power'].agg(['size', 'mean'])
    daily['energy'] = daily['mean'] * 24.0
    gate = daily['size'].ge(MIN_VALID_POINTS)
    gate &= daily['energy'].ge(MIN_ENERGY_MWH) | np.isclose(daily['energy'], MIN_ENERGY_MWH, rtol=0, atol=1e-12)
    energy = daily.loc[gate, ['energy']].reset_index()
    return points, energy


def fit_weights(rows, mode):
    station_counts = rows.groupby('station_id', observed=True)['energy'].transform('size').to_numpy(dtype=float)
    weights = 1.0 / station_counts
    if mode == 'relative_mse':
        weights /= rows['energy'].to_numpy(dtype=float)**2
    return weights / weights.mean()


def output_energy(raw, mode):
    cap = MAX_PREDICTION['T2']
    values = np.exp(np.minimum(raw, np.log(cap))) if mode == 'log_mse' else raw
    return np.clip(values, 0.0, cap)


def diagnostics(forecast, scored_days):
    rows = forecast.merge(scored_days, on=KEYS, validate='one_to_one', suffixes=('_pred', '_true'))
    ratio = rows['energy_pred'] / rows['energy_true']
    rows = rows.assign(ratio=ratio, squared_error=(ratio - 1)**2)
    def aggregate(groups):
        return {str(name): {'days': len(group), 'mean_pred_over_true': float(group['ratio'].mean()),
                            'relative_rmse': float(np.sqrt(group['squared_error'].mean()))}
                for name, group in groups}

    bands = pd.cut(rows['energy_true'], [0, 60, 120, 240, np.inf], right=False)
    return {'by_station': aggregate(rows.groupby('station_id', observed=True)),
            'by_actual_energy_band_mwh': aggregate(rows.groupby(bands, observed=True))}


def train_one(lightgbm, mode, earlier, inner, full, feature_columns, rounds, threads):
    def dataset(rows):
        target = rows['energy']
        if mode == 'log_mse':
            target = np.log(target)
        return lightgbm.Dataset(rows[feature_columns], label=target,
                                weight=fit_weights(rows, mode), categorical_feature=['station_id'])

    station_codes = pd.factorize(inner['station_id'])[0]
    station_counts = np.bincount(station_codes)
    true_energy = inner['energy'].to_numpy(dtype=float)

    def readme_score(raw, _):
        squared = ((output_energy(raw, mode) - true_energy) / true_energy)**2
        station_rmse = np.sqrt(np.bincount(station_codes, weights=squared) / station_counts)
        return 't2_readme_score', 100.0 * (1 - station_rmse.mean()), True

    config = {'objective': 'regression', 'metric': 'None', 'verbosity': -1,
              'learning_rate': 0.04, 'num_leaves': 15, 'min_data_in_leaf': 40,
              'num_threads': threads, 'deterministic': True, 'force_col_wise': True, 'seed': 20261121}
    inner_model = lightgbm.train(config, dataset(earlier), num_boost_round=rounds,
                                 valid_sets=[dataset(inner)], feval=readme_score,
                                 callbacks=[lightgbm.early_stopping(35, verbose=False)])
    best = inner_model.best_iteration
    model = lightgbm.train(config, dataset(full), num_boost_round=best)
    return model, best


def run(root, out, threads, rounds, dataset_factory, lightgbm):
    start = monotonic()
    def progress(message):
        print(f'[{monotonic() - start:.1f}s] {message}', flush=True)

    roster = pd.read_csv(root / 'submission/T2_dataset/submission_template.csv',
                         usecols=['station_id'], dtype={'station_id': str})['station_id'].drop_duplicates().sort_values()
    if len(roster) != 12:
        raise ValueError('T2 template must define 12 stations.')
    stations = pd.read_csv(root / 'static/power_stations.csv', dtype={'station_id': str})
    stations = roster.to_frame().merge(stations, on='station_id', validate='one_to_one', how='left')
    if not np.isfinite(stations[['latitude', 'longitude']].to_numpy(dtype=float)).all():
        raise ValueError('Every T2 station must have finite coordinates.')
    folders = validation_folders(root)
    features, cache_version = feature_cache(root, folders, stations, out, dataset_factory, progress)
    if len(features) != len(folders) * len(roster) or features.duplicated(KEYS).any():
        raise ValueError('Weather features must match every date and station exactly once.')
    model_columns = [column for column in features if column != 'date']
    cutoff = VALIDATION_START.isoformat()
    train_paths = [(day, folder) for day, folder in folders if day < cutoff]
    _, train_energy = load_daily_targets(train_paths, roster)
    full = features.merge(train_energy, on=KEYS, validate='one_to_one')
    if len(full) != len(train_energy):
        raise ValueError('Weather features are missing a labeled training station-day.')
    valid = features.loc[features['date'].ge(cutoff)].copy()
    for frame in [full, valid]:
        frame['station_id'] = pd.Categorical(frame['station_id'], categories=roster)
    earlier = full.loc[full['date'].lt(INNER_START.isoformat())]
    inner = full.loc[full['date'].ge(INNER_START.isoformat())]
    if earlier['station_id'].nunique() != 12 or inner['station_id'].nunique() != 12:
        raise ValueError('Both training-only windows must contain eligible days from all 12 stations.')

    receipt = {'data_root': str(root), 'feature_cache_version': cache_version, 'train_days': len(train_paths),
               'validation_days': len(folders) - len(train_paths), 'earlier_rows': len(earlier),
               'inner_rows': len(inner), 'full_training_rows': len(full), 'feature_count': len(model_columns),
               'early_stopping_start': INNER_START.isoformat(), 'validation_start': cutoff,
               'validation_end': VALIDATION_END.isoformat(), 'threads': threads, 'max_rounds': rounds,
               'rank_reference_readme': REFERENCE_T2, 'models': {}}
    predictions = {}
    for mode in MODES:
        progress(f'Fitting {mode}')
        model, best = train_one(lightgbm, mode, earlier, inner, full, model_columns, rounds, threads)
        forecast = valid[KEYS].copy()
        forecast['energy'] = output_energy(model.predict(valid[model_columns], num_threads=threads), mode)
        forecast.to_csv(out / f'validation_{mode}.csv', index=False)
        model.save_model(str(out / f'{mode}.txt'))
        predictions[mode] = forecast
        receipt['models'][mode] = {'best_iteration': best}
        progress(f'{mode} frozen at iteration {best}')
    validation_paths = [(day, folder) for day, folder in folders if day >= cutoff]
    # The external validation labels are read only after both predictions have been saved.
    validation_points, valid_energy = load_daily_targets(validation_paths, roster)
    for mode, forecast in predictions.items():
        result = score_track('T2', validation_points, forecast, roster)
        result['delta_to_companion_local'] = result['readme_unclipped_score'] - REFERENCE_T2
        result['diagnostics'] = diagnostics(forecast, valid_energy)
        receipt['models'][mode].update(result)
    control, candidate = (receipt['models'][mode]['readme_unclipped_score'] for mode in MODES)
    receipt['paired_delta_relative_vs_log'] = candidate - control
    receipt['primary_passes_reference'] = candidate > REFERENCE_T2
    receipt['elapsed_seconds'] = round(monotonic() - start, 2)
    report = json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    (out / 'results.json').write_text(report)
    print(report, end='', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--out', type=Path, default=Path('/workspace/qhl/runs/independent_t2_v1'))
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8 or not 1 <= args.rounds <= 500:
        parser.error('Use between 1 and 8 threads and between 1 and 500 rounds.')
    import lightgbm
    from netCDF4 import Dataset

    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'results.json').exists():
        parser.error('Result already exists. Choose a new run directory for a different evaluation.')
    run(args.data_root, args.out, args.threads, args.rounds, Dataset, lightgbm)


if __name__ == '__main__':
    main()
