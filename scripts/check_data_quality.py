#!/usr/bin/env python3
"""Read-only training quality checks and sampled forecast layout validation."""

import argparse
from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd


NWP_STEPS = {
    'tianji_0p125.nc': np.arange(33),
    'ec_hres_0p10.nc': np.arange(33),
    'aifs_0p25.nc': np.arange(0, 31, 6),
}
POWER_FIELDS = ['station_id', 'point_index', 'power', 'wind', 'power_limit']
POWER_MAX = 24.1026
CAPACITY = 20.0855


def sample_paths(paths, count):
    if not paths or count <= 0:
        return []
    positions = np.linspace(0, len(paths) - 1, min(count, len(paths)), dtype=int)
    return [paths[i] for i in positions]


def numeric_column(frame, name):
    numbers = pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    return numbers


def summarize(values):
    finite = values[np.isfinite(values)]
    result = {'count': int(values.size), 'non_finite': int(values.size - finite.size)}
    if finite.size:
        percentiles = np.quantile(finite, [0, .01, .50, .99, 1])
        result.update({name: round(float(value), 5) for name, value in
                       zip(('min', 'p01', 'median', 'p99', 'max'), percentiles)})
    return result


def scan_power(days):
    counters = Counter()
    month_counts = {}
    valid_power, wind_values = [], []
    coverage = []
    for day in days:
        path = day / 'power_target.csv'
        if not path.is_file():
            counters['missing_power_files'] += 1
            continue
        df = pd.read_csv(path, usecols=POWER_FIELDS, dtype={'station_id': 'string'})
        power = numeric_column(df, 'power')
        wind = numeric_column(df, 'wind')
        limit = numeric_column(df, 'power_limit')
        valid = np.isfinite(power) & (limit == 1)
        counters.update({
            'label_rows': len(df), 'valid_power': int(valid.sum()),
            'valid_zero': int((valid & (power == 0)).sum()),
            'limited_zero': int(((limit == 0) & (power == 0)).sum()),
            'valid_above_scoring_capacity': int((valid & (power > CAPACITY)).sum()),
            'valid_above_submission_value_range': int((valid & (power > POWER_MAX)).sum()),
            'valid_negative': int((valid & (power < 0)).sum()),
            'valid_power_missing_or_invalid_wind': int((valid & ~np.isfinite(wind)).sum()),
            'nonmissing_invalid_power': int((df['power'].notna().to_numpy() & ~np.isfinite(power)).sum()),
        })
        valid_power.append(power[valid])
        wind_values.append(wind[valid])
        by_station = df.assign(valid=valid).groupby('station_id')['valid'].agg(['sum', 'size'])
        by_station.columns = ['valid', 'total']
        by_station['gated_days'] = by_station['valid'].ge(72).astype(int)
        by_station['observed_days'] = 1
        coverage.append(by_station)
        month = day.name[:4] + '-' + day.name[4:6]
        counts = month_counts.setdefault(month, Counter())
        counts.update({'points': len(df), 'valid': int(valid.sum()), 'station_days': len(by_station),
                       'scored_by_point_count': int(by_station['gated_days'].sum())})
    result = {'counts': counters, 'months_observed': len(month_counts)}
    if coverage:
        combined = pd.concat(coverage).groupby(level=0).sum()
        result['station_valid_point_share'] = summarize((combined['valid'] / combined['total']).to_numpy())
        result['station_72_point_day_share'] = summarize((combined['gated_days'] / combined['observed_days']).to_numpy())
        result['monthly_valid_point_share'] = {month: round(c['valid'] / c['points'], 5)
                                               for month, c in month_counts.items() if c['points']}
        result['valid_power_mw'] = summarize(np.concatenate(valid_power))
        result['wind_raw_units_at_valid_power_points'] = summarize(np.concatenate(wind_values))
    return result


def as_float(variable_values):
    return np.ma.asarray(variable_values, dtype=float).filled(np.nan)


def forecast_layout(dataset, expected):
    times = as_float(dataset.variables['time'][:])
    lat = as_float(dataset.variables['lat'][:])
    lon = as_float(dataset.variables['lon'][:])
    wind_dims = tuple(dataset.variables['u100'].dimensions) == ('time', 'lat', 'lon')
    wind_dims &= tuple(dataset.variables['v100'].dimensions) == ('time', 'lat', 'lon')
    checks = {'relative_hours_match': bool(np.array_equal(times, expected)),
              'latitude_descending': bool(lat.size >= 2 and np.all(np.diff(lat) < 0)),
              'longitude_ascending': bool(lon.size >= 2 and np.all(np.diff(lon) > 0)),
              'uv_dimensions_match': bool(wind_dims)}
    return checks, lat, lon


def local_values(variable, lat_indices, lon_indices):
    # netCDF4 slices latitude and longitude independently before paired station selection.
    rows, row_lookup = np.unique(lat_indices, return_inverse=True)
    columns, column_lookup = np.unique(lon_indices, return_inverse=True)
    slab = as_float(variable[:, rows, columns])
    return slab[:, row_lookup, column_lookup].ravel()


def scan_nwp(training, test, station_lat, station_lon, open_dataset):
    report = {}
    for filename, expected in NWP_STEPS.items():
        quality = {'training_layout': Counter(), 'test_layout': Counter()}
        training_values = {}
        for split, paths in [('training', training), ('test', test)]:
            counts = quality[split + '_layout']
            for path in paths:
                try:
                    with open_dataset(path / 'nwp' / filename, mode='r') as ds:
                        checks, lat, lon = forecast_layout(ds, expected)
                        counts['checked'] += 1
                        for name, passed in checks.items():
                            counts[name + '_failed'] += not passed
                        if not all(checks.values()):
                            continue
                        outside = ((station_lat < lat.min()) | (station_lat > lat.max()) |
                                   (station_lon < lon.min()) | (station_lon > lon.max()))
                        counts['station_out_of_grid_total'] += int(outside.sum())
                        if split == 'test':
                            continue
                        yi = np.abs(lat[:, None] - station_lat[None, :]).argmin(axis=0)
                        xi = np.abs(lon[:, None] - station_lon[None, :]).argmin(axis=0)
                        values = {}
                        for name in ('u100', 'v100', 'ws100', 't2m', 'tp'):
                            if name in ds.variables:
                                values[name] = local_values(ds.variables[name], yi, xi)
                        speed = np.hypot(values['u100'], values['v100'])
                        values['speed_from_uv'] = speed
                        if 'ws100' in values:
                            values['reported_vs_uv_speed_abs_difference'] = np.abs(values['ws100'] - speed)
                        for name, items in values.items():
                            training_values.setdefault(name, []).append(items)
                except (OSError, RuntimeError, KeyError, ValueError, IndexError) as exc:
                    counts['file_or_read_error_' + type(exc).__name__] += 1
        quality['training_nearest_station_values'] = {
            name: summarize(np.concatenate(chunks)) for name, chunks in training_values.items()
        }
        report[filename] = quality
    return report


def scan_obs(training, station_count):
    counters = Counter()
    for folder in training:
        path = folder / 'obs/obs_input.csv'
        if not path.is_file():
            counters['files_missing'] += 1
            continue
        df = pd.read_csv(path, dtype={'station_id': 'string', 'date': 'string'})
        needed = {'station_id', 'date', 'point_index', 'wind_speed', 'wind_direction', 'precipitation'}
        if not needed.issubset(df.columns):
            counters['files_with_missing_columns'] += 1
            continue
        expected_date = (datetime.strptime(folder.name[:8], '%Y%m%d') - timedelta(days=1)).date().isoformat()
        counters['files_checked'] += 1
        counters['rows'] += len(df)
        counters['wrong_row_count_files'] += len(df) != station_count * 4
        counters['wrong_station_count_files'] += df['station_id'].nunique() != station_count
        counters['unexpected_date_rows'] += int((~df['date'].eq(expected_date).fillna(False)).sum())
        counters['unexpected_hour_rows'] += int((~df['point_index'].isin([17, 18, 19, 20])).sum())
        counters['duplicate_key_rows'] += int(df.duplicated(['station_id', 'date', 'point_index']).sum())
        for name in ('wind_speed', 'wind_direction', 'precipitation'):
            values = numeric_column(df, name)
            counters[name + '_non_finite'] += int((~np.isfinite(values)).sum())
            counters[name + '_negative'] += int((values < 0).sum())
            if name == 'wind_direction':
                counters['wind_direction_above_360'] += int((values > 360).sum())
    return counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--nwp-days', type=int, default=12, help='Training dates spread over available dates (plus up to 3 documented dates).')
    args = parser.parse_args()
    if args.nwp_days < 1:
        parser.error('--nwp-days must be positive')
    root = args.data_root
    if not all((root / name).is_dir() for name in ('train', 'test', 'static')):
        parser.error(f'{root} must contain train, test and static')
    from netCDF4 import Dataset

    start = monotonic()
    days = sorted(path for path in (root / 'train').glob('????????_0000')
                  if path.is_dir() and path.name[:8].isdigit())
    test = sorted(p for p in (root / 'test').iterdir() if p.is_dir())
    special = {'20241210_0000', '20260129_0000', '20260329_0000'}
    training_sample = sorted(set(sample_paths(days, args.nwp_days)) |
                             {path for path in days if path.name in special})
    test_layout_sample = sample_paths(test, 2)
    stations = pd.read_csv(root / 'static/power_stations.csv')
    station_lat, station_lon = (numeric_column(stations, name) for name in ['latitude', 'longitude'])
    if not len(stations) or not (np.isfinite(station_lat) & np.isfinite(station_lon)).all():
        raise ValueError('Power station coordinates must be present and finite.')
    obs_stations = len(pd.read_csv(root / 'static/obs_stations.csv'))
    report = {
        'data_root': str(root),
        'training_days': len(days), 'training_weather_sample_days': len(training_sample),
        'test_layout_sample_days': len(test_layout_sample),
        'train_power_quality': scan_power(days),
        'sampled_training_obs_quality': scan_obs(training_sample, obs_stations),
        'nwp': scan_nwp(training_sample, test_layout_sample, station_lat, station_lon, Dataset),
        'elapsed_seconds': round(monotonic() - start, 2),
    }
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
