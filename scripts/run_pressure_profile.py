#!/usr/bin/env python3
"""Evaluate fixed pressure-layer weather columns in T1 v3 and the T3 short-history tree."""
import argparse
import fcntl
from functools import partial
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic
import traceback

import numpy as np

from advance_batch_b import read_frozen_forecast
from batch_energy_short import fit_short_forecasts, roster_for, short_source, t3_windows
from compare_warm_t1 import PRIMARY_SCORE
from confirm_working_baselines import keep_sites, score_t3
from experiment_companion_t1 import assert_matched, read_json
from pressure_wind_profile import NAMES as PRESSURE_COLUMNS, append_pressure
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, chronological_dates, forecast_keys, sha256, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from run_steamlink_followup import confirmed_t3, native_forecast, old_t1, ramp_timing, wind_alternatives
from run_t1_issue_wind_error import dynamic_baseline
from run_t1_t3_batch import paired_report, point_diagnostics
from run_wind_attribution import STATION
from same_issue_weather import archive_stamp
from screen_companion_t1_stations import block_comparisons, show_progress
from t1_dynamic_wind import NAMES as DYNAMIC_COLUMNS, append_dynamic
from warm_shared_power import labels, load_roster

RUNS = Path('/workspace/qhl/runs')
ENTRY = Path(__file__)
ARM = 'pressure_profile'


def t1_forecasts(args, phase, seed, keys, reference):
    baseline, lock = dynamic_baseline(args, phase, keys, seed)
    old_v1, route_lock = old_t1(args, phase, keys, seed)
    fixed = keys.assign(prediction=reference.ravel())
    old_dir = args.dynamic_run / 'T1' / phase / str(seed) / 'dynamic_gradient'
    old_receipt = old_dir / 'training_receipt.json'
    info = read_json(old_receipt)
    dynamic = read_frozen_forecast(old_dir / info['prediction_file'], info['prediction_sha256'], keys, 'T1')
    replay = blend(keep_sites(dynamic, old_v1, [STATION]), fixed, .2, 'T1')
    if not np.allclose(replay.prediction, baseline.prediction, rtol=0, atol=1e-6):
        raise ValueError('Could not reproduce the accepted dynamic-gradient blend.')

    def augmented(data_root, cache, matrix, names, days, roster, positions):
        matrix, names = append_dynamic(data_root, args.dynamic_run / 'native_weather',
                                       matrix, names, days, roster, positions)
        return append_pressure(data_root, args.out / 'pressure_weather', matrix, names, days, roster, positions)

    tree, receipt = native_forecast(args, phase, seed, keys, component=ARM, feature_append=augmented,
                                   columns=[*DYNAMIC_COLUMNS, *PRESSURE_COLUMNS])
    check = read_json(args.out / 'T1' / phase / str(seed) / ARM / 'training_receipt.json')
    assert_matched(info, check, ('seed', 'data_root', 'feature_workdir', 'chronological_dates',
                                  'valid_training_sample_sha256', 'config_sha256',
                                  'companion_cache_metadata_sha256', 'early_stopping_days_inside_train', 'device'))
    if check['feature_names'] != info['feature_names'] + PRESSURE_COLUMNS:
        raise ValueError('T1 must add only the registered pressure columns to the v3 tree.')
    candidate = blend(keep_sites(tree, old_v1, [STATION]), fixed, .2, 'T1')
    return {'baseline': baseline, 'candidate': candidate}, {'locked_v3': lock, 'route': route_lock,
        'v3_tree': sha256(old_receipt), 'candidate_tree': receipt}


def t1_stage(args):
    roster = load_roster(args.data_root)
    root = args.out / 'T1'
    train, days = chronological_dates(args.split, 'A')
    show_progress(root, 'T1 压层 A', 0, 3, '重建 v3 风速功率分支')
    ref, _, thresholds, weather = wind_alternatives(args, train, days, roster, selected=ARM)
    keys = forecast_keys(days, roster)
    show_progress(root, 'T1 压层 A', 1, 3, '构建压层日缓存并拟合全天树')
    frames, source = t1_forecasts(args, 'A', A_SEED, keys, ref)
    frames, lock = freeze_window(root / 'A' / 'forecasts', 'T1', keys, frames, {'source': source, 'wind': weather})
    show_progress(root, 'T1 压层 A', 2, 3, '计分与爬坡诊断')
    truth = labels(args.data_root, days, roster)
    scores = block_comparisons(truth, frames, roster, ('candidate',))
    main = {arm: row[PRIMARY_SCORE] for arm, row in scores['scores'].items()}
    delta = main['candidate'] - main['baseline']
    gate = {'delta': delta, 'promoted': delta > 1e-8, 'forecast_lock': lock}
    diagnostics = {'pointwise': point_diagnostics(truth, frames, thresholds[0]),
                   'hour_timing': ramp_timing(truth, frames, thresholds[1])}
    freeze_json(root / 'A' / 'score.json', {'scores': scores, 'gate': gate, 'ramp': diagnostics})
    result = {'A': main, 'gate': gate, 'station_pairs': scores['station_pairs']['candidate'], 'ramp': diagnostics}
    show_progress(root, 'T1 压层 A', 3, 3, f'基线 {main["baseline"]:.6f}，候选 {main["candidate"]:.6f}，差 {delta:+.6f}')
    if not gate['promoted']:
        return result
    b_train, b_days = chronological_dates(args.split, 'B')
    if b_train != train + days:
        raise ValueError('T1 B must add all A days to training.')
    keys = forecast_keys(b_days, roster)
    ref, _, _, weather = wind_alternatives(args, b_train, b_days, roster, selected=ARM)
    pairs, locks = {}, {}
    folder = root / 'B'
    for index, seed in enumerate(B_SEEDS):
        show_progress(root, 'T1 压层 B', index, len(B_SEEDS), f'seed={seed}')
        frames, source = t1_forecasts(args, 'B', seed, keys, ref)
        pairs[seed], locks[str(seed)] = freeze_window(folder / str(seed) / 'pair', 'T1', keys, frames,
            {'source': source, 'wind': weather, 'gate': gate})
    digest = freeze_json(folder / 'predictions_frozen.json', locks)
    report = paired_report(folder, 'T1', labels(args.data_root, b_days, roster), roster, keys, pairs, digest)
    result['B'] = summarize_b(report)
    show_progress(root, 'T1 压层 B', len(B_SEEDS), len(B_SEEDS), '配对计分完成')
    return result


def t3_fit_contract(args, window, seed):
    if seed == A_SEED:
        path = args.short_tree / 'results.json'
        return read_json(path)['windows'][window], sha256(path)
    path = args.t3_confirm / 'T3' / str(seed) / 'predictions_frozen.json'
    receipt = read_json(path)['sources']['fit']['tree']
    if receipt['seed'] != seed:
        raise ValueError('T3 fit receipt uses a different seed.')
    return receipt, sha256(path)


def t3_forecasts(args, window, train, days, short, all_sites, model_source, seed):
    keys, tree, ridge, baseline, lock = confirmed_t3(args, window, train, days, short, seed)
    target = args.out / 'T3' / window / str(seed)
    target.mkdir(parents=True, exist_ok=True)
    old, contract = t3_fit_contract(args, window, seed)
    source_config = args.config or args.companion_code / 'configs/default.json'
    signature = {'dates': [list(map(str, train)), list(map(str, days))], 'seed': seed, 'device': args.device,
                 'config': sha256(source_config), 'cache': archive_stamp(args.workdir, [*train, *days]),
                 'fit_contract': contract, 'baseline': lock, 'pressure_columns': PRESSURE_COLUMNS}
    expected = read_json(args.short_tree / 'results.json')['settings']
    assert_matched(expected, {'data_root': str(args.data_root.resolve()), 'split_sha256': sha256(args.split),
                             'device': args.device, 'threads': args.threads,
                             'companion_config_sha256': signature['config']},
                   ('data_root', 'split_sha256', 'device', 'threads', 'companion_config_sha256'))
    fitted = target / 'tree'
    marker = fitted / 'fit.json'
    if not marker.is_file():
        stage = Path(mkdtemp(dir=target, prefix='.tree-attempt-'))
        *schema, params = model_source
        _, _, fit = fit_short_forecasts(args, stage, train, days, all_sites, short,
                                      (*schema, dict(params, random_state=seed)), modes=('short_only',),
                                      extra_features=partial(append_pressure, args.data_root, args.out / 'pressure_weather'))
        pairs = ('train_days', 'score_days', 'inner_stop_start', 'cache')
        assert_matched(old, fit, pairs)
        samples = ('fit_points', 'target_points', 'early_stop_points')
        assert_matched(old['fit']['short_only'], fit['fit']['short_only'], samples)
        freeze_json(stage / 'fit.json', {'signature': signature, 'fit': fit})
        stage.rename(fitted)
    info = read_json(marker)
    if info['signature'] != signature:
        raise ValueError('The T3 pressure-profile cache has a different fit contract.')
    improved = read_frozen_forecast(fitted / 'short_only.csv', info['fit']['predictions_sha256']['short_only'], keys, 'T3')
    result = {'baseline': baseline, 'candidate': blend(improved, ridge, .25, 'T3'),
              'short_tree': tree, 'pressure_tree': improved}
    frames, digest = freeze_window(target / 'forecasts', 'T3', keys, result, {
        'old_fit': contract, 'reference': lock, 'candidate': sha256(marker)})
    return keys, frames, digest


def summarize_b(report):
    return {'acceptance': report['acceptance'],
            'scores': {name: row[PRIMARY_SCORE] for name, row in report['ensemble']['scores'].items()},
            'seed_deltas': {seed: row['delta'] for seed, row in report['pairs'].items()},
            'identical_seed_groups': report['identical_seed_groups']}


def t3_stage(args):
    root = args.out / 'T3'
    short = roster_for(args.data_root, 'T3')
    all_sites, source = short_source(args, load_roster(args.data_root) + short)
    windows = list(t3_windows())
    result = {'A': {}}
    for index, (window, train, days) in enumerate(windows[:2]):
        show_progress(root, 'T3 压层早窗', index, 2, window)
        _, frames, digest = t3_forecasts(args, window, train, days, short, all_sites, source, A_SEED)
        scores = score_t3(labels(args.data_root, days, short), short, frames)
        freeze_json(root / window / 'score.json', {**scores, 'prediction_lock': digest})
        result['A'][window] = {'scores': {name: row[PRIMARY_SCORE] for name, row in scores['scores'].items()},
                               'delta': scores['delta'], 'station_deltas': scores['station_deltas']}
        print(f'[T3 {window}] {result["A"][window]["scores"]}，融合差 {scores["delta"]:+.6f}', flush=True)
    average = float(np.mean([part['delta'] for part in result['A'].values()]))
    gate = {'mean_early_delta': average, 'promoted': average > 1e-8}
    result['gate'] = gate
    freeze_json(root / 'a_score.json', result)
    show_progress(root, 'T3 压层早窗', 2, 2, f'准入 {gate}')
    if gate['promoted']:
        window, train, days = windows[-1]
        folder = root / window
        pairs, locks = {}, {}
        for index, seed in enumerate(B_SEEDS):
            show_progress(root, 'T3 压层 50/10', index, len(B_SEEDS), f'seed={seed}')
            keys, arms, source_lock = t3_forecasts(args, window, train, days, short, all_sites, source, seed)
            pairs[seed], locks[str(seed)] = freeze_window(folder / str(seed) / 'pair', 'T3', keys,
                {arm: arms[arm] for arm in ('baseline', 'candidate')}, {'gate': gate, 'forecast_lock': source_lock})
        digest = freeze_json(folder / 'predictions_frozen.json', locks)
        report = paired_report(folder, 'T3', labels(args.data_root, days, short), short, keys, pairs, digest)
        result['B'] = summarize_b(report)
        show_progress(root, 'T3 压层 50/10', len(B_SEEDS), len(B_SEEDS), '配对计分完成')
    return result


def execute(args):
    start = monotonic()
    names = (ENTRY.name, 'pressure_wind_profile.py', 'batch_energy_short.py', 't1_dynamic_wind.py',
             'run_steamlink_followup.py', 'run_companion_warm_t1.py')
    freeze_json(args.out / 'plan.json', {
        'paths': {name: str(getattr(args, name).resolve()) for name in ('data_root', 'workdir', 'companion_code',
            'dynamic_run', 'power_run', 'wind_run', 't3_confirm', 'short_tree', 'transfer_run')},
        'split': sha256(args.split), 'device': args.device, 'threads': args.threads, 'seeds': list(B_SEEDS),
        'columns': PRESSURE_COLUMNS, 'code': {name: sha256(ENTRY.with_name(name)) for name in names},
    })
    results = {}
    for i, (track, stage) in enumerate((('T3', t3_stage), ('T1', t1_stage))):
        show_progress(args.out, '压层风批次', i, 2, track)
        (args.out / track).mkdir(exist_ok=True)
        try:
            results[track] = {'status': 'complete', **stage(args)}
        except Exception as error:
            traceback.print_exc()
            results[track] = {'status': 'failed', 'error': str(error)}
        write_json(args.out / 'summary.json', results)
    show_progress(args.out, '压层风批次', 2, 2, '结果已保存')
    results['seconds'] = monotonic()-start
    write_json(args.out / 'summary.json', results)
    print(f'结果摘要 {args.out / "summary.json"}', flush=True)
    return int(any(results[key]['status'] != 'complete' for key in ('T1','T3')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out': RUNS / 'pressure_profile_gpu_v1', 'data-root': Path('/workspace/qhl/data'),
        'workdir': Path('/workspace/cpy/code/workdir'), 'companion-code': Path('/workspace/cpy/code'),
        'dynamic-run': RUNS / 't1_dynamic_gradient_gpu_v1', 'power-run': RUNS / 't1_t3_fast_gpu_v1',
        'wind-run': RUNS / 'wind_attribution_gpu_v1', 'short-tree': RUNS / 'three_track_gpu_batch_v1/T3',
        'transfer-run': RUNS / 'same_issue_followup_v1/T3', 't3-confirm': RUNS / 't3_transfer_confirmation_gpu_v1',
        'split': SPLIT_PATH}
    for field, default in defaults.items():
        parser.add_argument(f'--{field}', type=Path, default=default)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1,5), default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(execute(args))


if __name__ == '__main__':
    main()
