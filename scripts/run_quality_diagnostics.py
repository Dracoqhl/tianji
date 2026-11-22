#!/usr/bin/env python3
"""Read-only diagnostics for the chronological A and T3 early windows."""

import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import fcntl
from pathlib import Path
import json
from time import monotonic

import numpy as np
import pandas as pd

from advance_batch_b import keys_for
from batch_energy_short import roster_for, t3_windows
from check_data_quality import NWP_STEPS, as_float, sample_paths, scan_nwp
from run_companion_warm_t1 import A_SEED, SPLIT_PATH, chronological_dates, forecast_keys, sha256, write_json
from run_pressure_and_inventory import PRIOR, read_result, summarize_arm
from run_wind_attribution import read_arm
from score_forecasts import MIN_ENERGY_MWH, MIN_VALID_POINTS, MAX_PREDICTION


def sources(root, day):
    return root / 'train' / f'{day:%Y%m%d}_0000'


def frame_grid(path, stations, points, expected_date):
    frame = pd.read_csv(path, dtype={'station_id': str, 'date': str})
    frame = frame.loc[frame.station_id.isin(stations)].copy()
    if (not frame.date.eq(expected_date.isoformat()).all() or not frame.point_index.isin(points).all()):
        raise ValueError(f'Unexpected date or point index in {path}')
    index = pd.MultiIndex.from_product([stations, points], names=['station_id', 'point_index'])
    return frame.assign(present=True).set_index(index.names).reindex(index)


def values(frame, name, width):
    return pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=float, na_value=np.nan).reshape(-1, width)


def coordinates(path):
    table = pd.read_csv(path, dtype={'station_id': str}).set_index('station_id')
    if not table.index.is_unique or not np.isfinite(table[['latitude', 'longitude']]).all().all():
        raise ValueError(f'Invalid station coordinates in {path}')
    return table


def obs_support(folder, day, obs, neighbors):
    frame = frame_grid(folder / 'obs/obs_input.csv', obs.index, range(17, 21), day - timedelta(days=1))
    speed, direction, rain, temperature = (values(frame, name, 4) for name in
                                           ('wind_speed', 'wind_direction', 'precipitation', 'temperature'))
    speed_ok = np.isfinite(speed) & (speed >= 0)
    direction_ok = np.isfinite(direction) & (direction >= 0) & (direction <= 360)
    joint = speed_ok & direction_ok
    usable = joint | (speed_ok & (speed == 0))
    nearby = usable[neighbors]
    directions = np.where(joint, direction, 0)
    magnitude = np.where(joint, speed, 0)
    vector_u = -magnitude * np.sin(np.deg2rad(directions))
    vector_v = -magnitude * np.cos(np.deg2rad(directions))
    count = nearby.sum(axis=1)
    local_u = np.divide(vector_u[neighbors].sum(axis=1), count, out=np.full(count.shape, np.nan), where=count > 0)
    local_v = np.divide(vector_v[neighbors].sum(axis=1), count, out=np.full(count.shape, np.nan), where=count > 0)
    bins = {
        'rows': np.ones_like(joint), 'row_absent': ~frame.present.eq(True).to_numpy().reshape(-1, 4),
        'speed_missing': ~np.isfinite(speed), 'direction_missing': ~np.isfinite(direction),
        'only_speed_usable': speed_ok & ~direction_ok, 'only_direction_usable': direction_ok & ~speed_ok,
        'neither_usable': ~speed_ok & ~direction_ok, 'joint': joint, 'vector_usable': usable,
        'low_wind': speed_ok & (speed < 1), 'low_wind_direction_missing': (speed >= 0) & (speed < 1) & ~np.isfinite(direction),
        'temperature_missing': ~np.isfinite(temperature), 'precipitation_missing': ~np.isfinite(rain),
        'invalid_speed': np.isfinite(speed) & ~speed_ok, 'invalid_direction': np.isfinite(direction) & ~direction_ok,
    }
    counts = {str(site): {name: int(mask[i].sum()) for name, mask in bins.items()} |
              {'all_four_without_vector': int(not usable[i].any()), 'last_without_vector': int(not usable[i, -1])}
              for i, site in enumerate(obs.index)}
    return counts, {'nearby_count': count, 'natural_missing': ~nearby.all(axis=(1, 2)),
                    'last_incomplete': ~nearby[:, :, -1].all(axis=1),
                    'one_neighbor_all_four_missing': (~nearby.any(axis=2)).any(axis=1),
                    'no_local_vector_in_at_least_one_frame': (count == 0).any(axis=1),
                    'vector_speed': np.hypot(local_u, local_v)}


def nearest(power, obs):
    lat1, lon1 = np.deg2rad(power[['latitude', 'longitude']].to_numpy()).T[:, :, None]
    lat2, lon2 = np.deg2rad(obs[['latitude', 'longitude']].to_numpy()).T[:, None, :]
    distance_order = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    return np.argsort(distance_order, axis=1, kind='stable')[:, :5]


def power_state(frame):
    power, wind, flag = (values(frame, name, 96) for name in ('power', 'wind', 'power_limit'))
    present = frame.present.eq(True).to_numpy().reshape(-1, 96)
    finite = np.isfinite(power)
    valid = finite & (flag == 1)
    states = {'absent_row': ~present, 'missing_power': present & ~finite, 'missing_limit': present & finite & ~np.isfinite(flag),
              'limited': finite & (flag == 0), 'valid': valid,
              'unknown_limit': finite & np.isfinite(flag) & (flag != 0) & (flag != 1)}
    return power, wind, valid, states


def power_events(power, wind, valid, threshold):
    adjacent = valid[:, 1:] & valid[:, :-1]
    step = np.diff(power, axis=1)
    wind_step = np.diff(wind, axis=1)
    four = np.lib.stride_tricks.sliding_window_view(valid, 5, axis=1).all(axis=-1)
    hour_change = power[:, 4:] - power[:, :-4]
    hour_wind_change = wind[:, 4:] - wind[:, :-4]
    ramp = four & (np.abs(hour_change) >= threshold)
    ramp_supported = ramp & np.isfinite(hour_wind_change) & (np.sign(hour_change) * hour_wind_change > 0)
    spike = adjacent[:, :-1] & adjacent[:, 1:] & (np.abs(step[:, :-1]) >= threshold)
    spike &= (step[:, :-1] * step[:, 1:] < 0) & (np.abs(step[:, :-1] + step[:, 1:]) < threshold / 4)
    spike_wind = np.isfinite(wind_step[:, :-1]) & np.isfinite(wind_step[:, 1:])
    spike_wind &= (step[:, :-1] * wind_step[:, :-1] > 0) & (step[:, 1:] * wind_step[:, 1:] > 0)
    flat = four & (np.ptp(np.lib.stride_tricks.sliding_window_view(power, 5, axis=1), axis=-1) <= .01)
    return {'hour_ramp': (ramp, 5), 'hour_ramp_wind_supported': (ramp_supported, 5),
            'isolated_jump_return': (spike, 2),
            'isolated_jump_return_wind_supported': (spike & spike_wind, 2), 'zero_flat_hour': (flat & (power[:, 4:] == 0), 5),
            'positive_flat_hour': (flat & (power[:, 4:] > 0), 5)}


def power_audit(root, days, sites, threshold, label):
    monthly, hours, examples = defaultdict(Counter), defaultdict(Counter), defaultdict(list)
    changes = []
    station_days = []
    for i, day in enumerate(days, 1):
        grid = frame_grid(sources(root, day) / 'power_target.csv', sites, range(1, 97), day)
        power, wind, valid, states = power_state(grid)
        codes = (~np.isfinite(wind)).astype(int) + 2 * (~np.isfinite(power)).astype(int)
        codes += 4 * (~np.isfinite(values(grid, 'power_limit', 96)))
        present = grid.present.eq(True).to_numpy().reshape(-1, 96)
        states['valid_zero'] = valid & (power == 0)
        states['valid_power_outside_submission_range'] = valid & ((power < 0) | (power > MAX_PREDICTION['T1']))
        events = power_events(power, wind, valid, threshold) if threshold is not None else {}
        if threshold is None:
            contiguous = np.lib.stride_tricks.sliding_window_view(valid, 5, axis=1).all(axis=-1)
            changes.extend(np.abs(power[:, 4:] - power[:, :-4])[contiguous])
        for j, site in enumerate(sites):
            key = f'{site}/{day:%Y-%m}'
            count = int(valid[j].sum())
            monthly[key].update({'station_days': 1, 'points': 96, 'valid_72_to_95': int(72 <= count < 96),
                                 'complete_96': int(count == 96), 'zero_valid': int(count == 0)})
            for code, n in enumerate(np.bincount(codes[j, present[j]], minlength=8)):
                pattern = f'{code:03b}'[::-1]
                monthly[key][f'joint_missing_wind_power_limit_{pattern}'] += int(n)
            for state, mask in states.items():
                active = mask[j]
                missing_wind = ~np.isfinite(wind[j]) & active
                monthly[key].update({state: int(active.sum()), state + '_wind_missing': int(missing_wind.sum())})
                by_hour = active.reshape(24, 4).sum(axis=1)
                wind_by_hour = missing_wind.reshape(24, 4).sum(axis=1)
                for hour, (n, w) in enumerate(zip(by_hour, wind_by_hour)):
                    hours[f'{site}/{hour:02d}'].update({state: int(n), state + '_wind_missing': int(w)})
            for name, (mask, offset) in events.items():
                indices = np.flatnonzero(mask[j]) + offset
                monthly[key][name] += int(len(indices))
                for point in indices[:max(0, 40 - len(examples[name]))]:
                    center = int(point) - 1
                    sample = slice(max(0, center - 4), center + 2)
                    examples[name].append({'date': str(day), 'station_id': site, 'point_index': int(point),
                                           'window_start_point': max(1, center - 3), 'power': nullable(power[j, sample]),
                                           'wind': nullable(wind[j, sample]), 'valid': valid[j, sample].tolist()})
        station_days.append((day, power, valid))
        if i in (1, len(days)) or i % 90 == 0:
            print(f'[{label}] days {i}/{len(days)}', flush=True)
    return {'station_month_counts': monthly, 'station_hour_counts': hours, 'examples': examples}, station_days, changes


def nullable(array):
    return [float(x) if np.isfinite(x) else None for x in array]


def forecast_errors(station_days, forecast, obs_by_date, sites, threshold):
    groups = defaultdict(list)
    residual_groups = defaultdict(list)
    forecast = forecast.prediction.to_numpy().reshape(len(station_days), len(sites), 96)
    for i, (day, power, valid) in enumerate(station_days):
        support = obs_by_date[day]
        count = valid.sum(axis=1)
        all_groups = {'all': count >= MIN_VALID_POINTS, 'complete': count == 96,
                      'low_coverage': (count >= MIN_VALID_POINTS) & (count < 96),
                      'natural_complete': ~support['natural_missing']}
        all_groups.update({key: flag for key, flag in support.items() if flag.ndim == 1})
        err = forecast[i] - power
        for name, group in all_groups.items():
            for j in np.flatnonzero(group & (count >= MIN_VALID_POINTS)):
                groups[name].append((sites[j], float(np.sqrt(np.mean(err[j, valid[j]]**2)))))
        event_masks = power_events(power, np.full_like(power, np.nan), valid, threshold) if threshold is not None else {}
        for name, (mask, offset) in event_masks.items():
            if name.endswith('_wind_supported'):
                continue
            event_mask = mask & (count[:, None] >= MIN_VALID_POINTS)
            residual_groups[name].extend(err[:, offset - 1:offset - 1 + mask.shape[1]][event_mask])
    return {'groups': {name: {'station_days': len(rows),
                             'mean_station_day_rmse_mw': float(np.mean([rmse for _, rmse in rows])),
                             'by_station': {site: {'station_days': len(errors), 'mean_station_day_rmse_mw': float(np.mean(errors))}
                                            for site in sites
                                            if (errors := [rmse for key, rmse in rows if key == site])}}
                       for name, rows in groups.items()},
            'event_points': {name: {'points': len(errors), 'rmse_mw': float(np.sqrt(np.mean(np.square(errors))))}
                             for name, errors in residual_groups.items() if errors}}


def local_block(dataset, name, latitude, longitude):
    lat, lon = (as_float(dataset.variables[axis][:]) for axis in ('lat', 'lon'))
    rows = np.abs(lat[:, None] - latitude).argmin(axis=0)
    cols = np.abs(lon[:, None] - longitude).argmin(axis=0)
    field = as_float(dataset.variables[name][:])
    if field.ndim != 3 or tuple(dataset.variables[name].dimensions) != ('time', 'lat', 'lon'):
        raise ValueError(f'Unexpected gridded layout: {name}')
    return field, [field[:, max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(len(field), -1)
                   for y, x in zip(rows, cols)]


def weather_report(paths, power, dataset):
    lat, lon = power[['latitude', 'longitude']].to_numpy().T
    report = {'nwp': scan_nwp(paths, [], lat, lon, dataset), 'satellite': {}}
    for name, relative, var in [('GPM', 'gpm/gpm_input.nc', 'precipitation'),
                                ('FY4B', 'fy4b/fy4b_tbb13_input.nc', 'tbb13')]:
        layout, site_counts = Counter(), defaultdict(Counter)
        for path in paths:
            try:
                with dataset(path / relative) as data:
                    if not np.array_equal(as_float(data.variables['time'][:]), np.arange(-4, 0)):
                        layout['unexpected_time_axis'] += 1
                        continue
                    raw, blocks = local_block(data, var, lat, lon)
                    layout['files'] += 1
                    layout['domain_all_missing_frames'] += int((~np.isfinite(raw).any(axis=(1, 2))).sum())
            except (OSError, RuntimeError, KeyError, ValueError, IndexError) as error:
                layout[type(error).__name__] += 1
                continue
            for site, block in zip(power.index, blocks):
                finite = np.isfinite(block)
                common = (finite[1:] & finite[:-1]).mean(axis=1)
                site_counts[site].update({'station_days': 1, 'frames': 4, 'local_frames_with_any_data': int(finite.any(axis=1).sum()),
                                          'last_frame_empty': int(not finite[-1].any()), 'all_four_empty': int(not finite.any()),
                                          'adjacent_pairs': 3, 'pairs_with_80pct_common_support': int((common >= .8).sum()),
                                          'negative_pixels': int((block < 0).sum())})
        report['satellite'][name] = {'files': layout, 'station_counts': site_counts}
    return report


def tianji_site_wind(path, sites, open_dataset):
    lat, lon = sites[['latitude', 'longitude']].to_numpy().T
    with open_dataset(path) as data:
        time, grid_lat, grid_lon = (as_float(data.variables[x][:]) for x in ('time', 'lat', 'lon'))
        if not np.array_equal(time, NWP_STEPS[path.name]):
            raise ValueError(f'Unexpected time axis: {path}')
        rows = np.abs(grid_lat[:, None] - lat).argmin(axis=0)
        cols = np.abs(grid_lon[:, None] - lon).argmin(axis=0)
        yi, y = np.unique(rows, return_inverse=True)
        xi, x = np.unique(cols, return_inverse=True)
        u, v = (as_float(data.variables[name][:, yi, xi])[:, y, x] for name in ('u100', 'v100'))
        return np.hypot(u, v)


def add_nwp_context(root, examples, sites, dataset):
    forecasts = {}
    for samples in examples.values():
        for sample in samples:
            day = date.fromisoformat(sample['date'])
            if day not in forecasts:
                path = sources(root, day) / 'nwp/tianji_0p125.nc'
                try:
                    forecasts[day] = tianji_site_wind(path, sites, dataset)
                except (OSError, RuntimeError, KeyError, ValueError, IndexError):
                    forecasts[day] = None
            wind = forecasts[day]
            if wind is not None:
                position = sites.index.get_loc(sample['station_id'])
                start = sample['window_start_point'] - 1
                hours = 4 + np.arange(start, start + len(sample['power'])) / 4
                sample['tianji_wind_mps'] = nullable(np.interp(hours, np.arange(len(wind)), wind[:, position]))


def predicted_wind(root, days, sites, open_dataset):
    result = np.full((len(days), len(sites)), np.nan)
    for i, day in enumerate(days):
        path = sources(root, day) / 'nwp/tianji_0p125.nc'
        try:
            speed = tianji_site_wind(path, sites, open_dataset)[4:28]
            valid = np.isfinite(speed)
            result[i] = np.divide(np.where(valid, speed, 0).sum(axis=0), valid.sum(axis=0),
                                  out=np.full(len(sites), np.nan), where=valid.sum(axis=0) >= 18)
        except (OSError, RuntimeError, KeyError, ValueError, IndexError):
            pass
        if i == 0 or (i + 1) % 90 == 0 or i + 1 == len(days):
            print(f'[T2 mask] wind days {i + 1}/{len(days)}', flush=True)
    return result


def mask_replay(station_days, wind, sites):
    power, valid = (np.stack(items) for items in zip(*(record[1:] for record in station_days)))
    coverage = valid.sum(axis=-1)
    rng = np.random.default_rng(A_SEED)
    rows = []
    for j, site in enumerate(sites):
        full = np.flatnonzero((coverage[:, j] == 96) & np.isfinite(wind[:, j]))
        partial = np.flatnonzero((coverage[:, j] >= MIN_VALID_POINTS) & (coverage[:, j] < 96) & np.isfinite(wind[:, j]))
        for i in partial:
            past = full[full < i]
            if not len(past):
                continue
            distances = np.abs(wind[past, j] - wind[i, j])
            target = past[distances.argmin()]
            distance = float(distances.min())
            if distance > 1.0:
                continue
            truth = power[target, j]
            pattern = valid[i, j]
            actual = float(truth.mean() * 24)
            structured = float(truth[pattern].mean() * 24)
            random = [float(truth[rng.choice(96, size=int(pattern.sum()), replace=False)].mean() * 24)
                      for _ in range(16)]
            above = actual >= MIN_ENERGY_MWH
            rows.append({'station_id': site, 'mask_date': str(station_days[i][0]), 'replay_date': str(station_days[target][0]),
                         'valid_points': int(pattern.sum()), 'wind_distance_mps': distance,
                         'full_mwh': actual, 'structured_error_mwh': structured - actual,
                         'random_mean_error_mwh': float(np.mean(random) - actual),
                         'random_mean_abs_error_mwh': float(np.mean(np.abs(np.array(random) - actual))),
                         'structured_threshold_flip': int((structured >= MIN_ENERGY_MWH) != above),
                         'random_threshold_flip_rate': float(np.mean((np.array(random) >= MIN_ENERGY_MWH) != above))})
    def summary(records):
        return {'matched_masks': len(records),
                'distinct_replay_days': len({(r['station_id'], r['replay_date']) for r in records}),
                **({key: float(np.mean([r[key] for r in records])) for key in
                    ('structured_error_mwh', 'random_mean_error_mwh', 'random_mean_abs_error_mwh',
                     'structured_threshold_flip', 'random_threshold_flip_rate')} if records else {}),
                **({'structured_mean_abs_error_mwh': float(np.mean([abs(r['structured_error_mwh']) for r in records]))}
                   if records else {})}
    return {'partial_station_days': int(((coverage >= MIN_VALID_POINTS) & (coverage < 96)).sum()),
            'finite_nwp_wind_station_days': int(np.isfinite(wind).sum()), 'overall': summary(rows),
            'by_mask_month': {m: summary([r for r in rows if r['mask_date'].startswith(m)])
                              for m in sorted({r['mask_date'][:7] for r in rows})},
            'by_station': {site: summary([r for r in rows if r['station_id'] == site]) for site in sites},
            'rows': rows}


def locked(runs, directory, days, sites, track):
    folder = runs / directory
    manifest = folder / 'predictions_frozen.json'
    keys = keys_for(days, sites, track) if track == 'T2' else forecast_keys(days, sites)
    return read_arm(folder, 'baseline', keys, track), sha256(manifest)


def t2_errors(station_days, forecast):
    keys = [(str(day), site) for day, power, _ in station_days
            for site in forecast.station_id.unique()]
    ordered = forecast.set_index(['date', 'station_id']).loc[keys]['energy'].to_numpy()
    power, valid = (np.stack(items) for items in zip(*(record[1:] for record in station_days)))
    count = valid.sum(axis=-1)
    actual = 24 * np.divide(np.where(valid, power, 0).sum(axis=-1), count,
                            out=np.zeros_like(count, dtype=float), where=count > 0)
    eligible = (count >= MIN_VALID_POINTS) & ((actual >= MIN_ENERGY_MWH) |
                                               np.isclose(actual, MIN_ENERGY_MWH, atol=1e-12, rtol=0))
    relative = np.divide(ordered.reshape(actual.shape) - actual, actual, out=np.zeros_like(actual), where=eligible)
    groups = {'all': eligible, '72_to_95': eligible & (count < 96), '96': eligible & (count == 96),
              'low_24_to_60_mwh': eligible & (actual < 60)}
    return {name: {'station_days': int(mask.sum()), 'relative_rmse': float(np.sqrt(np.mean(relative[mask]**2)))
                   if mask.any() else None} for name, mask in groups.items()}


def obs_audit(root, days, obs, power, neighbors):
    by_station_month, by_farm_month, by_day = defaultdict(Counter), defaultdict(Counter), {}
    failures = {}
    for i, day in enumerate(days, 1):
        try:
            station, support = obs_support(sources(root, day), day, obs, neighbors)
        except (OSError, ValueError, KeyError) as exc:
            failures[str(day)] = str(exc)
            shape = (len(power), 4)
            support = {'nearby_count': np.zeros(shape, dtype=int), 'vector_speed': np.full(shape, np.nan)}
            support.update({key: np.ones(len(power), dtype=bool) for key in (
                'natural_missing', 'last_incomplete', 'one_neighbor_all_four_missing',
                'no_local_vector_in_at_least_one_frame')})
        else:
            for site, counts in station.items():
                by_station_month[f'{site}/{day:%Y-%m}'].update(counts)
        for j, site in enumerate(power.index):
            by_farm_month[f'{site}/{day:%Y-%m}'].update({
                'station_days': 1, 'nearby_vector_station_hours': int(support['nearby_count'][j].sum()),
                'possible_nearby_station_hours': 4 * len(neighbors[j]),
                **{key: int(mask[j]) for key, mask in support.items() if mask.ndim == 1}})
        by_day[day] = support
        if i in (1, len(days)) or i % 90 == 0:
            print(f'[OBS] days {i}/{len(days)}', flush=True)
    return {'days': len(days), 'failed_files': failures, 'obs_station_month': by_station_month,
            'wind_farm_month': by_farm_month}, by_day


def window_support(by_day, days, sites, all_sites):
    index = all_sites.index.get_indexer(sites)
    return {day: {name: data[index] for name, data in by_day[day].items()} for day in days}


def ramp_threshold(changes):
    return max(.01, float(np.quantile(changes, .95))) if len(changes) else None


def inventory(runs):
    reports = {}
    for name in PRIOR:
        folder = runs / name
        record = read_result(folder / 'summary.json')
        summary = summarize_arm(record) if 'status' in record else {
            track: summarize_arm(item) for track, item in record.items() if isinstance(item, dict)}
        reports[name] = {'summary': summary, 'progress': read_result(folder / 'progress.json')}
    return reports


def window_diagnostics(root, runs, out, track, name, sites, fit_days, score_days, folder, support, coordinates, dataset):
    _, _, changes = power_audit(root, fit_days, sites, None, f'{track} {name} training threshold')
    threshold = ramp_threshold(changes)
    fit, train_data, _ = power_audit(root, fit_days, sites, threshold, f'{track} {name} training')
    audit, score_data, _ = power_audit(root, score_days, sites, threshold, f'{track} {name} evaluation')
    add_nwp_context(root, fit['examples'], coordinates, dataset)
    add_nwp_context(root, audit['examples'], coordinates, dataset)
    result = {'dates': {'train': list(map(str, fit_days)), 'evaluation': list(map(str, score_days))},
              'train_hourly_ramp_p95_mw': threshold, 'train': fit, 'evaluation': audit}
    try:
        forecast, digest = locked(runs, folder, score_days, sites, track)
        result['forecast'] = {'manifest_sha256': digest, 'folder': folder,
                              **forecast_errors(score_data, forecast, support, sites, threshold)}
    except (OSError, KeyError, ValueError) as exc:
        result['forecast'] = {'status': 'unavailable', 'detail': str(exc)}
    write_json(out / f'{track}_{name}_power.json', result)
    return train_data, score_data, result


def totals(by_group):
    result = Counter()
    for counts in by_group.values():
        result.update(counts)
    return result


def execute(args):
    from netCDF4 import Dataset

    root, runs, out = args.data_root, args.runs, args.out
    start = monotonic()
    first, recent = chronological_dates(args.split, 'A')
    long = roster_for(root, 'T1')
    short = roster_for(root, 'T3')
    t3 = list(t3_windows())[:2]
    sites = coordinates(root / 'static/power_stations.csv').loc[long + short]
    obs = coordinates(root / 'static/obs_stations.csv')
    receipt = {'inputs': {'data_root': str(root.resolve()), 'split_sha256': sha256(args.split),
                          'source_sha256': sha256(Path(__file__)),
                          'weather_sample_days_per_prefix': args.weather_days, 'no_model_fits': True},
               'status': 'running', 'stages': {}}

    def checkpoint(name, details):
        receipt['stages'][name] = details
        receipt['elapsed_seconds'] = round(monotonic() - start, 2)
        write_json(out / 'batch_overview.json', receipt)

    print('[1/5] Prior experiment reports', flush=True)
    checkpoint('previous', inventory(runs))
    print('[2/5] Observation missingness and nearby wind-vector support', flush=True)
    neighbors = nearest(sites, obs)
    obs_report, support = obs_audit(root, sorted(set(first + recent + [day for _, a, b in t3 for day in a + b])),
                                    obs, sites, neighbors)
    obs_report['nearest_observation_stations'] = {site: obs.index[row].tolist()
                                                   for site, row in zip(sites.index, neighbors)}
    write_json(out / 'obs_quality.json', obs_report)
    checkpoint('observations', {'report': 'obs_quality.json', 'days': obs_report['days'],
                                'files_failed': len(obs_report['failed_files']),
                                'observation_totals': totals(obs_report['obs_station_month']),
                                'wind_farm_totals': totals(obs_report['wind_farm_month'])})

    print('[3/5] T1/T3 power states, ramp diagnostics and locked A forecasts', flush=True)
    windows = [('T1', 'recent_A', long, first, recent, 'pressure_profile_gpu_v1/T1/A/forecasts'),
               *(('T3', name, short, fit, days, f'pressure_profile_gpu_v1/T3/{name}/{A_SEED}/forecasts')
                 for name, fit, days in t3)]
    for track, name, roster, fit, days, folder in windows:
        train_data, validation_data, report = window_diagnostics(
            root, runs, out, track, name, roster, fit, days, folder, window_support(support, days, roster, sites), sites.loc[roster], Dataset)
        if track == 'T1':
            t2_training, t2_validation = train_data, validation_data
        checkpoint(f'{track}_{name}', {'report': f'{track}_{name}_power.json',
                                       'ramp_threshold_mw': report['train_hourly_ramp_p95_mw'],
                                       'fit_totals': totals(report['train']['station_month_counts']),
                                       'evaluation_totals': totals(report['evaluation']['station_month_counts']),
                                       'forecast': report['forecast']})

    print('[4/5] T2 structured mask replay and A coverage errors', flush=True)
    if set(roster_for(root, 'T2')) != set(long):
        raise ValueError('T1 and T2 station rosters differ.')
    wind = predicted_wind(root, first, sites.loc[long], Dataset)
    replay = mask_replay(t2_training, wind, long)
    try:
        forecast, digest = locked(runs, 'steamlink_followup_gpu_v1/T2/A/forecasts', recent, long, 'T2')
        replay['A'] = {'manifest_sha256': digest, 'coverage_errors': t2_errors(t2_validation, forecast)}
    except (OSError, KeyError, ValueError) as exc:
        replay['A'] = {'status': 'unavailable', 'detail': str(exc)}
    write_json(out / 't2_mask_replay.json', replay)
    checkpoint('T2', {'report': 't2_mask_replay.json', **{k: replay[k] for k in ('partial_station_days', 'overall', 'A')}})

    print('[5/5] NWP source quality and consecutive satellite-frame coverage', flush=True)
    for name, days, roster in [('long_train', first, long), (t3[0][0], t3[0][1], short), (t3[1][0], t3[1][1], short)]:
        paths = sample_paths([sources(root, day) for day in days], args.weather_days)
        report = weather_report(paths, sites.loc[roster], Dataset)
        report['sample_dates'] = [path.name[:8] for path in paths]
        file = f'weather_{name}.json'
        write_json(out / file, report)
        checkpoint(f'weather_{name}', {'report': file, 'sample_days': len(paths),
                                      'NWP': {k: v['training_layout'] for k, v in report['nwp'].items()},
                                      'satellite': {k: v['files'] for k, v in report['satellite'].items()}})
    receipt['status'] = 'complete'
    write_json(out / 'batch_overview.json', receipt)
    print(f'[Report] {out / "batch_overview.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--runs', type=Path, default=Path('/workspace/qhl/runs'))
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weather-days', type=int, default=16, choices=range(1, 65), metavar='1–64')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        overview = args.out / 'batch_overview.json'
        write_json(overview, {'status': 'starting', 'stages': {}})
        try:
            execute(args)
        except Exception as error:
            receipt = json.loads(overview.read_text())
            receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
            write_json(overview, receipt)
            raise


if __name__ == '__main__':
    main()
