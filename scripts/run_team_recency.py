#!/usr/bin/env python3
"""Pair the teammates' fixed recency weights against the current T1/T3 baselines."""
import argparse
from datetime import date
import fcntl
from functools import partial
import hashlib
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic
import traceback

import numpy as np

from advance_batch_b import read_frozen_forecast
from batch_energy_short import fit_short_forecasts
from confirm_working_baselines import keep_sites
from experiment_companion_t1 import assert_matched, read_json
from run_companion_warm_t1 import B_SEEDS, SPLIT_PATH, sha256, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from run_pressure_and_inventory import read_result
from run_pressure_profile import t1_stage, t3_fit_contract, t3_stage
from run_steamlink_followup import confirmed_t3, native_forecast, old_t1
from run_t1_issue_wind_error import dynamic_baseline
from run_wind_attribution import STATION
from same_issue_weather import archive_stamp
from screen_companion_t1_stations import show_progress
from t1_dynamic_wind import NAMES as DYNAMIC_COLUMNS, append_dynamic


RUNS = Path('/workspace/qhl/runs')
ARM = 'team_recency'
RECENCY = {'T1': {'floor': .50, 'half_life_days': 180},
           'T3': {'floor': .75, 'half_life_days': 20}}


def recency_weights(days, roster, positions, original, *, floor, half_life_days):
    """Apply date decay to the given fit rows and preserve each station's weight sum."""
    if not days or any(type(day) is not date for day in days) or list(days) != sorted(set(days)):
        raise ValueError('Recency weights require ordered, unique calendar training days.')
    if not roster or len(roster) != len(set(roster)) or not 0 < floor <= 1 or half_life_days <= 0:
        raise ValueError('Invalid station roster or recency parameters.')
    positions = np.asarray(positions)
    if (positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer) or not positions.size
            or positions.min() < 0 or positions.max() >= len(days) * len(roster) * 96
            or np.unique(positions).size != positions.size):
        raise ValueError('Weights require distinct fit rows in the current training grid.')
    site_day = positions // 96
    day_idx, site_idx = np.divmod(site_day, len(roster))
    ages = np.array([(days[-1] - day).days for day in days])
    factors = floor + (1.0 - floor) * np.exp2(-ages / half_life_days)
    base = np.ones(positions.size, dtype=float) if original is None else np.asarray(original, dtype=float)
    if base.shape != positions.shape or not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError('Original weights must be finite, positive and aligned to fit rows.')
    weights = base * factors[day_idx]

    def mass(items):
        return np.bincount(site_idx, weights=items, minlength=len(roster))

    original_sums, raw_sums = mass(base), mass(weights)
    scale = np.divide(original_sums, raw_sums, out=np.ones_like(raw_sums), where=raw_sums > 0)
    weights *= scale[site_idx]
    totals = mass(weights)
    if not np.isfinite(weights).all() or (weights <= 0).any() or not np.allclose(totals, original_sums, rtol=1e-10):
        raise ValueError('Recency weighting failed to preserve positive station mass.')
    counts = np.bincount(site_idx, minlength=len(roster))
    age_sum = mass(ages[day_idx] * weights)
    original_age_sum = mass(ages[day_idx] * base)
    squared_sum = mass(weights**2)
    return weights, {
        'anchor': str(days[-1]), 'days': len(days), 'floor': floor, 'half_life_days': half_life_days,
        'fit_rows': positions.size, 'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest(),
        'min_weight': float(weights.min()), 'max_weight': float(weights.max()),
        'by_station': {
            station: {'rows': int(counts[i]), 'original_weight_sum': float(original_sums[i]),
                      'weighted_sum': float(totals[i]), 'normalizer': float(scale[i]),
                      'original_mean_age_days': float(original_age_sum[i] / original_sums[i]),
                      'weighted_mean_age_days': float(age_sum[i] / totals[i]),
                      'weight_effective_points': float(totals[i]**2 / squared_sum[i])}
            for i, station in enumerate(roster) if counts[i]
        },
    }


def t1_pairs(args, phase, seed, keys, reference):
    baseline, lock = dynamic_baseline(args, phase, keys, seed)
    routed, route_lock = old_t1(args, phase, keys, seed)
    wind = keys.assign(prediction=reference.ravel())
    source = args.dynamic_run / 'T1' / phase / str(seed) / 'dynamic_gradient'
    original_path = source / 'training_receipt.json'
    original = read_json(original_path)
    original_tree = read_frozen_forecast(source / original['prediction_file'], original['prediction_sha256'], keys, 'T1')
    restored = blend(keep_sites(original_tree, routed, [STATION]), wind, .2, 'T1')
    if not np.allclose(restored.prediction, baseline.prediction, rtol=0, atol=1e-6):
        raise ValueError('The tree, route and wind forecasts did not reproduce the frozen T1 baseline.')

    def dynamic(data_root, _cache, matrix, names, days, roster, positions):
        return append_dynamic(data_root, args.dynamic_run / 'native_weather', matrix, names, days, roster, positions)

    tree, receipt = native_forecast(
        args, phase, seed, keys, component=ARM, feature_append=dynamic, columns=DYNAMIC_COLUMNS,
        training_weight=partial(recency_weights, **RECENCY['T1']))
    check = read_json(args.out / 'T1' / phase / str(seed) / ARM / 'training_receipt.json')
    assert_matched(original, check, ('seed', 'data_root', 'feature_workdir', 'chronological_dates',
        'valid_training_sample_sha256', 'config_sha256', 'companion_cache_metadata_sha256',
        'early_stopping_days_inside_train', 'early_stopping_start_date', 'early_stopping_valid_rows',
        'device', 'threads', 'feature_names', 'backend_version'))
    weight = check['experts'][0]['training_weight']
    candidate = blend(keep_sites(tree, routed, [STATION]), wind, .2, 'T1')
    return {'baseline': baseline, 'candidate': candidate}, {
        'baseline_forecast_lock': lock, 'unchanged_route_lock': route_lock, 'v3_tree_receipt': sha256(original_path),
        'candidate_tree_receipt': receipt, 'training_weight_sha256': {k: v['weights_sha256'] for k, v in weight.items()}}


def t3_pairs(args, window, train, days, short, all_sites, model_source, seed):
    keys, tree, ridge, baseline, lock = confirmed_t3(args, window, train, days, short, seed)
    old, contract = t3_fit_contract(args, window, seed)
    source_config = args.config or args.companion_code / 'configs/default.json'
    signature = {'dates': [list(map(str, train)), list(map(str, days))], 'seed': seed, 'device': args.device,
                 'config': sha256(source_config), 'cache': archive_stamp(args.workdir, [*train, *days]),
                 'fit_contract': contract, 'baseline': lock, 'recency': RECENCY['T3']}
    expected = read_json(args.short_tree / 'results.json')['settings']
    assert_matched(expected, {'data_root': str(args.data_root.resolve()), 'split_sha256': sha256(args.split),
                             'device': args.device, 'threads': args.threads,
                             'companion_config_sha256': signature['config']},
                   ('data_root', 'split_sha256', 'device', 'threads', 'companion_config_sha256'))
    target = args.out / 'T3' / window / str(seed)
    target.mkdir(parents=True, exist_ok=True)
    fitted = target / 'tree'
    marker = fitted / 'fit.json'
    if not marker.is_file():
        staging = Path(mkdtemp(dir=target, prefix='.tree-attempt-'))
        *schema, params = model_source
        _, _, fit = fit_short_forecasts(
            args, staging, train, days, all_sites, short, (*schema, dict(params, random_state=seed)),
            modes=('short_only',), training_weight=partial(recency_weights, **RECENCY['T3']))
        assert_matched(old, fit, ('train_days', 'score_days', 'inner_stop_start', 'cache', 'sample_keys'))
        assert_matched(old['fit']['short_only'], fit['fit']['short_only'],
                       ('fit_points', 'target_points', 'early_stop_points'))
        freeze_json(staging / 'fit.json', {'signature': signature, 'fit': fit})
        staging.rename(fitted)
    info = read_json(marker)
    if info['signature'] != signature:
        raise ValueError('The T3 recency cache has a different fit contract.')
    improved = read_frozen_forecast(fitted / 'short_only.csv', info['fit']['predictions_sha256']['short_only'], keys, 'T3')
    arms = {'baseline': baseline, 'candidate': blend(improved, ridge, .25, 'T3'),
            'short_tree': tree, 'weighted_short_tree': improved}
    frames, digest = freeze_window(target / 'forecasts', 'T3', keys, arms, {
        'baseline_forecast_lock': lock, 'original_fit_contract': contract, 'candidate_tree_receipt': sha256(marker)})
    return keys, frames, digest


def execute(args):
    start = monotonic()
    code = Path(__file__).parent
    sources = ('run_team_recency.py', 'run_pressure_profile.py', 'run_companion_warm_t1.py',
               'batch_energy_short.py', 'run_steamlink_followup.py', 'tree_model.py')
    freeze_json(args.out / 'plan.json', {
        'paths': {name: str(getattr(args, name).resolve()) for name in ('data_root', 'workdir', 'companion_code',
            'dynamic_run', 'power_run', 'short_tree', 'transfer_run', 't3_confirm')},
        'split_sha256': sha256(args.split), 'device': args.device, 'threads': args.threads, 'tracks': args.tracks,
        'seeds': list(B_SEEDS), 'parameters': RECENCY,
        'code_sha256': {name: sha256(code / name) for name in sources},
    })
    audit = read_result(args.quality_report)
    results = {'quality_audit': {'path': str(args.quality_report),
                                 'status': audit.get('status', 'unknown')}}
    stages = {'T3': partial(t3_stage, predict_pair=t3_pairs, title='T3 近期加权'),
              'T1': partial(t1_stage, predict_pair=t1_pairs, title='T1 近期加权')}
    ordered = [track for track in stages if track in args.tracks]
    for i, track in enumerate(ordered):
        show_progress(args.out, '近期样本加权', i, len(ordered), track)
        (args.out / track).mkdir(exist_ok=True)
        try:
            results[track] = {'status': 'complete', **stages[track](args)}
        except Exception as error:
            traceback.print_exc()
            results[track] = {'status': 'failed', 'error': str(error)}
        write_json(args.out / 'summary.json', results)
    show_progress(args.out, '近期样本加权', len(ordered), len(ordered), '结果已保存')
    results['seconds'] = monotonic() - start
    write_json(args.out / 'summary.json', results)
    print(f'结果摘要 {args.out / "summary.json"}', flush=True)
    return int(any(results[track]['status'] != 'complete' for track in ordered))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out': RUNS / 'team_recency_gpu_v1', 'data-root': Path('/workspace/qhl/data'),
        'workdir': Path('/workspace/cpy/code/workdir'), 'companion-code': Path('/workspace/cpy/code'),
        'dynamic-run': RUNS / 't1_dynamic_gradient_gpu_v1', 'power-run': RUNS / 't1_t3_fast_gpu_v1',
        'short-tree': RUNS / 'three_track_gpu_batch_v1/T3', 'transfer-run': RUNS / 'same_issue_followup_v1/T3',
        't3-confirm': RUNS / 't3_transfer_confirmation_gpu_v1', 'split': SPLIT_PATH,
        'quality-report': RUNS / 'quality_diagnostics_cpu_v1/batch_overview.json'}
    for field, default in defaults.items():
        parser.add_argument(f'--{field}', type=Path, default=default)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--tracks', nargs='+', choices=('T1', 'T3'), default=['T1', 'T3'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(execute(args))


if __name__ == '__main__':
    main()
