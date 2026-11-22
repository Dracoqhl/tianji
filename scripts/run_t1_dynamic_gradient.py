#!/usr/bin/env python3
"""Compare the frozen T1 v2 to a six-column dynamical weather augmentation."""
import argparse
import fcntl
from pathlib import Path
from time import monotonic

import numpy as np

from compare_warm_t1 import PRIMARY_SCORE
from confirm_working_baselines import keep_sites
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, chronological_dates, forecast_keys, sha256, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from run_steamlink_followup import native_forecast, old_t1, ramp_timing, wind_alternatives
from run_t1_t3_batch import paired_report, point_diagnostics
from run_t1_wind_aux import prior_baseline
from run_wind_attribution import STATION
from screen_companion_t1_stations import block_comparisons, show_progress
from t1_dynamic_wind import NAMES, RADIUS_M, append_dynamic
from warm_shared_power import labels, load_roster

RUNS = Path('/workspace/qhl/runs')
ARM = 'dynamic_gradient'


def forecasts(args, phase, seed, keys, wind_power):
    baseline, prior_lock = prior_baseline(args, phase, keys, seed)
    prior, old_lock = old_t1(args, phase, keys, seed)
    fixed = keys.assign(prediction=wind_power.ravel())
    if not np.allclose(blend(prior, fixed, .2, 'T1').prediction, baseline.prediction, atol=1e-6, rtol=0):
        raise ValueError('The unchanged wind-power branch does not reproduce T1 v2.')
    tree, model = native_forecast(args, phase, seed, keys, component=ARM, feature_append=append_dynamic, columns=NAMES)
    tree = keep_sites(tree, prior, [STATION])
    return {'baseline': baseline, 'candidate': blend(tree, fixed, .2, 'T1')}, {
        'v2_reference': prior_lock, 'previous_route': old_lock, 'candidate_tree': model,
    }


def execute(args):
    start = monotonic()
    root = args.out
    roster = load_roster(args.data_root)
    scripts = (Path(__file__), Path(__file__).with_name('t1_dynamic_wind.py'),
               Path(__file__).with_name('steamlink_native_wind.py'),
               Path(__file__).with_name('run_steamlink_followup.py'))
    freeze_json(root / 'plan.json', {
        'paths': {k: str(getattr(args, k).resolve()) for k in ('data_root', 'workdir', 'wind_run', 'power_run',
                                                              'companion_code')},
        'split': sha256(args.split), 'device': args.device, 'threads': args.threads, 'seeds': list(B_SEEDS),
        'feature_names': NAMES, 'radius_m': RADIUS_M, 'code': {script.name: sha256(script) for script in scripts},
    })
    show_progress(root, 'T1 动态梯度 A', 0, 3, '读取固定风速功率分支')
    train, days = chronological_dates(args.split, 'A')
    keys = forecast_keys(days, roster)
    wind, _, thresholds, detail = wind_alternatives(args, train, days, roster, selected=ARM)
    show_progress(root, 'T1 动态梯度 A', 1, 3, '构建六列，拟合全天树')
    frames, receipt = forecasts(args, 'A', A_SEED, keys, wind)
    frames, lock = freeze_window(root / 'T1/A/forecasts', 'T1', keys, frames, {'source': receipt, 'fit': detail})
    show_progress(root, 'T1 动态梯度 A', 2, 3, '计分与爬坡诊断')
    truth = labels(args.data_root, days, roster)
    scores = block_comparisons(truth, frames, roster, ('candidate',))
    values = {arm: row[PRIMARY_SCORE] for arm, row in scores['scores'].items()}
    delta = values['candidate'] - values['baseline']
    gate = {'promoted': delta > 1e-8, 'delta': delta, 'prediction_lock': lock}
    ramp = {'pointwise': point_diagnostics(truth, frames, thresholds[0]),
            'hour_timing': ramp_timing(truth, frames, thresholds[1])}
    monthly = {}
    for month in sorted(set(keys.date.str[:7])):
        pair = {key: value.loc[value.date.str[:7].eq(month)].reset_index(drop=True) for key, value in frames.items()}
        monthly[month] = block_comparisons(truth.loc[truth.date.str[:7].eq(month)], pair, roster, ('candidate',))
    freeze_json(root / 'T1/A/score.json', {'report': scores, 'monthly': monthly, 'gate': gate, 'ramp': ramp})
    summary = {'A': {'scores': values, 'gate': gate, 'ramp': ramp,
                     'station_pairs': scores['station_pairs']['candidate']}}
    write_json(root / 'summary.json', summary)
    print(f'[T1 A] {values["baseline"]:.6f} -> {values["candidate"]:.6f}, 差 {delta:+.6f}', flush=True)
    show_progress(root, 'T1 动态梯度 A', 3, 3, f'自动进入 B={gate["promoted"]}')
    if gate['promoted']:
        a_train, a_days = train, days
        train, days = chronological_dates(args.split, 'B')
        if train != a_train + a_days:
            raise ValueError('B training dates must add all A dates.')
        keys = forecast_keys(days, roster)
        wind, _, _, detail = wind_alternatives(args, train, days, roster, selected=ARM)
        directory = root / 'T1/B'
        pairs, sources = {}, {}
        for index, seed in enumerate(B_SEEDS):
            show_progress(root, 'T1 动态梯度 B', index, len(B_SEEDS), f'seed={seed}')
            frames, receipt = forecasts(args, 'B', seed, keys, wind)
            pairs[seed], sources[str(seed)] = freeze_window(directory / str(seed) / 'pair', 'T1', keys, frames,
                {'source': receipt, 'gate': gate, 'fit': detail})
        digest = freeze_json(directory / 'predictions_frozen.json', sources)
        report = paired_report(directory, 'T1', labels(args.data_root, days, roster), roster, keys, pairs, digest)
        summary['B'] = {'acceptance': report['acceptance'],
                        'scores': {arm: score[PRIMARY_SCORE] for arm, score in report['ensemble']['scores'].items()},
                        'seed_deltas': {key: item['delta'] for key, item in report['pairs'].items()},
                        'identical_seed_groups': report['identical_seed_groups']}
        show_progress(root, 'T1 动态梯度 B', len(B_SEEDS), len(B_SEEDS), '配对计分完成')
    summary['seconds'] = monotonic() - start
    write_json(root / 'summary.json', summary)
    print(f'结果摘要 {root / "summary.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RUNS / 't1_dynamic_gradient_gpu_v1')
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--wind-run', type=Path, default=RUNS / 'wind_attribution_gpu_v1')
    parser.add_argument('--power-run', type=Path, default=RUNS / 't1_t3_fast_gpu_v1')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)


if __name__ == '__main__':
    main()
