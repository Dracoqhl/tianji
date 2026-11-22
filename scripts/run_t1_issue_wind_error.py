#!/usr/bin/env python3
"""Test observational issue-time wind residuals against the adopted T1 v3 recipe."""
import argparse
import fcntl
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, chronological_dates, forecast_keys, sha256, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from run_steamlink_followup import corrected_sources, finite_mean, mapped, old_t1
from run_t1_t3_batch import paired_report
from run_t1_wind_aux import prior_baseline
from run_wind_attribution import observed_curve, read_arm, wind_labels
from same_issue_weather import archived_issue_weather
from score_forecasts import MAX_PREDICTION, MIN_VALID_POINTS
from screen_companion_t1_stations import block_comparisons, show_progress
from t1_issue_wind_error import (FOLD_DAYS, farm_residual, forecast_inputs, initial_residual_fit, near_weights,
                                 observation_table, training_truth, wind_response)
from warm_shared_power import labels, load_roster

RUNS = Path('/workspace/qhl/runs')


def dynamic_baseline(args, phase, keys, seed):
    root = args.dynamic_run / 'T1'
    if phase == 'A':
        folder = root / 'A' / 'forecasts'
        locked = sha256(folder / 'predictions_frozen.json')
        gate = read_json(root / 'A' / 'score.json')['gate']
        if gate['prediction_lock'] != locked or not gate['promoted']:
            raise ValueError('The T1 v3 A forecasts are not linked to their approved gate.')
    else:
        base = root / 'B'
        marker = base / 'predictions_frozen.json'
        score = read_json(base / 'b_score.json')
        folder = base / str(seed) / 'pair'
        locked = sha256(folder / 'predictions_frozen.json')
        if (not score['acceptance']['accepted'] or score['prediction_lock_sha256'] != sha256(marker)
                or read_json(marker)[str(seed)] != locked
                or read_json(folder / 'predictions_frozen.json')['sources']['gate']
                != read_json(root / 'A' / 'score.json')['gate']):
            raise ValueError('The T1 v3 accepted B reference failed its lock verification.')
    baseline = read_arm(folder, 'candidate', keys, 'T1')
    return baseline, locked


def fit_weather(args, train, days, roster):
    from netCDF4 import Dataset

    sites, coordinates = observation_table(args.data_root)
    farms = pd.read_csv(args.data_root / 'static/power_stations.csv', dtype={'station_id': str}).set_index('station_id')
    positions = farms.loc[roster, ['latitude', 'longitude']].to_numpy(dtype=float)
    near, weight = near_weights(coordinates, positions)
    count = len(train)
    obs, input_hash = forecast_inputs(args.data_root, args.out / 'issue_inputs', [*train, *days],
                                       sites, coordinates, Dataset)
    t0 = training_truth(args.data_root, train, sites)
    truth = wind_labels(args.data_root, train, roster)
    shape = (count, len(roster), 96)
    wind = truth.wind.to_numpy(dtype=float).reshape(shape)
    power = truth.power.to_numpy(dtype=float).reshape(shape)
    valid = truth.power_limit.eq(1).to_numpy().reshape(shape)
    curves = [observed_curve(wind[:, site], power[:, site], valid[:, site]) for site in range(len(roster))]
    weather, columns, stamp = archived_issue_weather(args.workdir, args.data_root, [*train, *days], roster, print)
    weather = weather.reshape(count+len(days), len(roster), 96, len(columns))

    def unadjusted(stop, first, last):
        return finite_mean(corrected_sources(weather, columns, wind, slice(0, stop), slice(first, last)))

    predictions, corrections, info = [], [], []
    first_fold = count - 3 * FOLD_DAYS
    if first_fold <= 0:
        raise ValueError('The issue-error method requires three training-only 90-day blocks with an earlier prefix.')
    for first in range(first_fold, count, FOLD_DAYS):
        last = first+FOLD_DAYS
        print(f'[T1 训练内时间外] {train[first]} to {train[last-1]}', flush=True)
        correction, samples = initial_residual_fit(obs[:first], t0[:first], obs[first:last])
        predictions.append(unadjusted(first, first, last))
        corrections.append(farm_residual(correction, near, weight))
        info.append({'prefix_last': str(train[first-1]), 'validation': [str(train[first]), str(train[last-1])],
                     'issue_wind_pairs': samples})

    correction, full_samples = initial_residual_fit(obs[:count], t0, obs[count:])
    future = farm_residual(correction, near, weight)
    reference = unadjusted(count, count, None)
    adjusted, response = wind_response(wind[first_fold:], np.concatenate(predictions), np.concatenate(corrections),
                                        future, reference)
    return {'baseline_wind': reference, 'adjusted_wind': adjusted,
            'baseline_branch': mapped(reference, curves), 'candidate_branch': mapped(adjusted, curves),
            'metadata': {'source_grid_cache': stamp, 'issue_inputs': input_hash, 'folds': info,
                         'full_issue_wind_pairs': full_samples, 'horizon_coefficients': response,
                         'nearby_weather_sites': {name: [sites[i] for i in near[j]] for j, name in enumerate(roster)},
                         'issue_residual_available_station_days': int(np.isfinite(future).all(axis=-1).sum())}}


def pair_for(args, phase, seed, keys, weather):
    baseline, lock = dynamic_baseline(args, phase, keys, seed)
    v2, v2_lock = prior_baseline(args, phase, keys, seed)
    previous, old_lock = old_t1(args, phase, keys, seed)
    fixed = keys.assign(prediction=weather['baseline_branch'].ravel())
    if not np.allclose(blend(previous, fixed, .2, 'T1').prediction, v2.prediction, atol=1e-6, rtol=0):
        raise ValueError('The unchanged wind branch does not reproduce locked T1 v2.')
    # With both inputs already bounded, the 80/20 convex mix is unclipped.
    tree = (baseline.prediction.to_numpy() - .20*fixed.prediction.to_numpy()) / .80
    if not ((tree >= -1e-6) & (tree <= MAX_PREDICTION['T1'] + 1e-6)).all():
        raise ValueError('The adopted v3 forecast is incompatible with the frozen wind branch.')
    update = .20 * (weather['candidate_branch'].ravel()-weather['baseline_branch'].ravel())
    candidate = keys.assign(prediction=(baseline.prediction + update).clip(0, MAX_PREDICTION['T1']))
    return {'baseline': baseline, 'candidate': candidate}, {'v3_lock': lock, 'v2_lock': v2_lock, 'old_lock': old_lock}


def wind_diagnostics(truth, weather):
    baseline = weather['baseline_wind']
    corrected = weather['adjusted_wind']
    actual = truth.wind.to_numpy(dtype=float).reshape(baseline.shape)
    scored = truth.power_limit.eq(1).to_numpy().reshape(baseline.shape) & np.isfinite(
        truth.power.to_numpy(dtype=float).reshape(baseline.shape))
    eligible = scored & (scored.sum(axis=-1, keepdims=True) >= MIN_VALID_POINTS)
    common = eligible & np.isfinite(actual) & (actual >= 0) & np.isfinite(baseline) & np.isfinite(corrected)
    reports = {}
    for name, span in [('all', slice(None)), *[(str(i+1), slice(i, i+24)) for i in range(0, 96, 24)]]:
        mask = common[..., span]
        if mask.any():
            old = float(np.abs(baseline[..., span][mask]-actual[..., span][mask]).mean())
            new = float(np.abs(corrected[..., span][mask]-actual[..., span][mask]).mean())
        else:
            old, new = None, None
        reports[name] = {'paired_points': int(mask.sum()), 'baseline_mae': old, 'candidate_mae': new,
                         'delta': new-old if old is not None else None}
    return reports


def execute(args):
    from netCDF4 import __version__ as netcdf_version
    start = monotonic()
    root = args.out
    roster = load_roster(args.data_root)
    code = ('run_t1_issue_wind_error.py', 't1_issue_wind_error.py', 'run_steamlink_followup.py',
            't1_dynamic_wind.py', 'run_wind_attribution.py')
    freeze_json(root / 'plan.json', {
        'paths': {key: str(getattr(args, key).resolve()) for key in
                  ('data_root', 'workdir', 'wind_run', 'power_run', 'dynamic_run')},
        'split': sha256(args.split), 'netcdf_version': netcdf_version, 'fold_days': FOLD_DAYS,
        'script_hashes': {name: sha256(Path(__file__).with_name(name)) for name in code},
    })
    train, days = chronological_dates(args.split, 'A')
    show_progress(root, 'T1 起报误差 A', 0, 3, '读缓存，构建训练内预测')
    weather = fit_weather(args, train, days, roster)
    show_progress(root, 'T1 起报误差 A', 1, 3, '替换 v3 的风速功率分支')
    keys = forecast_keys(days, roster)
    frames, stamp = pair_for(args, 'A', A_SEED, keys, weather)
    frames, lock = freeze_window(root / 'T1/A/forecasts', 'T1', keys, frames,
                                 {'reference': stamp, 'fit': weather['metadata']})
    show_progress(root, 'T1 起报误差 A', 2, 3, '计分与天气增量')
    truth = wind_labels(args.data_root, days, roster)
    comparison = block_comparisons(truth, frames, roster, ('candidate',))
    scores = {arm: value[PRIMARY_SCORE] for arm, value in comparison['scores'].items()}
    wind = wind_diagnostics(truth, weather)
    delta = scores['candidate']-scores['baseline']
    wind_delta = wind['all']['delta']
    gate = {'power_delta': delta, 'wind_mae_delta': wind_delta, 'prediction_lock': lock,
            'promoted': delta > 1e-8 and wind_delta is not None and wind_delta < -1e-8}
    freeze_json(root / 'T1/A/score.json', {'report': comparison, 'wind': wind, 'gate': gate})
    summary = {'A': {'scores': scores, 'wind': wind, 'gate': gate,
                     'station_pairs': comparison['station_pairs']['candidate']}}
    write_json(root / 'summary.json', summary)
    show_progress(root, 'T1 起报误差 A', 3, 3, f'分差 {delta:+.6f}，自动进入 B={gate["promoted"]}')
    if gate['promoted']:
        a_train, a_days = train, days
        train, days = chronological_dates(args.split, 'B')
        if train != a_train + a_days:
            raise ValueError('B must use the training prefix plus all A dates.')
        keys = forecast_keys(days, roster)
        print('[T1 B] 拟合一次确定性订正，复用 v3 三种子', flush=True)
        weather = fit_weather(args, train, days, roster)
        folder = root / 'T1/B'
        pairs, signatures = {}, {}
        for j, seed in enumerate(B_SEEDS):
            show_progress(root, 'T1 起报误差 B', j, len(B_SEEDS), f'seed={seed}')
            frames, stamp = pair_for(args, 'B', seed, keys, weather)
            pairs[seed], signatures[str(seed)] = freeze_window(folder / str(seed) / 'pair', 'T1', keys, frames,
                  {'reference': stamp, 'fit': weather['metadata'], 'gate': gate})
        all_locked = freeze_json(folder / 'predictions_frozen.json', signatures)
        report = paired_report(folder, 'T1', labels(args.data_root, days, roster), roster, keys, pairs, all_locked)
        summary['B'] = {'scores': {arm: value[PRIMARY_SCORE] for arm, value in report['ensemble']['scores'].items()},
                        'acceptance': report['acceptance'], 'identical_seed_groups': report['identical_seed_groups'],
                        'seed_deltas': {seed: value['delta'] for seed, value in report['pairs'].items()}}
        show_progress(root, 'T1 起报误差 B', len(B_SEEDS), len(B_SEEDS), '配对计分完成')
    summary['seconds'] = monotonic()-start
    write_json(root / 'summary.json', summary)
    print(f'结果摘要 {root / "summary.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RUNS / 't1_issue_wind_error_cpu_v1')
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--wind-run', type=Path, default=RUNS / 'wind_attribution_gpu_v1')
    parser.add_argument('--power-run', type=Path, default=RUNS / 't1_t3_fast_gpu_v1')
    parser.add_argument('--dynamic-run', type=Path, default=RUNS / 't1_dynamic_gradient_gpu_v1')
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)


if __name__ == '__main__':
    main()
