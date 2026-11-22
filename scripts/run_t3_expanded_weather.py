#!/usr/bin/env python3
"""Evaluate cached pre-issue observations and satellite context for frozen T3 transfer."""
import argparse
import fcntl
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic

import numpy as np

from advance_batch_b import read_frozen_forecast
from batch_energy_short import roster_for, t3_windows
from compare_t2_mlp import select_mlp_runtime
from compare_warm_t1 import PRIMARY_SCORE
from confirm_working_baselines import average_seeds, score_t3
from experiment_companion_t1 import read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, date_roles, sha256, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from run_t1_t3_batch import call, paired_report
from run_t3_gradual_adaptation import checkpoint_contract
from same_issue_weather import CONTEXT_FIELDS, NWP_FIELDS, archived_issue_weather
from screen_companion_t1_stations import show_progress
from screen_short_transfer import fit_models, pretrain_dates
from warm_shared_power import labels, load_roster

RUNS = Path('/workspace/qhl/runs')
ENTRY = Path(__file__).resolve()
EXPANDED = 'expanded_weather'
RIDGE = 'frozen_context_ridge'
EXTRA = {
    'obs': (
        'obs_wind_dir_sin_idw_3', 'obs_wind_dir_cos_idw_3', 'obs_wind_speed_mean_4h',
        'obs_wind_speed_trend_4h', 'obs_temperature_trend_4h', 'obs_precipitation_mean_4h',
        'obs_missing_frac',
    ),
    'sat': (
        'gpm_mean_4h', 'gpm_missing_3', 'fy4b_trend_4h', 'fy4b_min_4h', 'fy4b_missing_4h',
        'fy4b_cold_240_3',
    ),
}
COLUMNS = [*NWP_FIELDS, *CONTEXT_FIELDS, *(name for fields in EXTRA.values() for name in fields)]


def source_for(args, days):
    allowed, _ = date_roles(args.split, 'A')
    return pretrain_dates(allowed, days)


def run_worker(args):
    roster = roster_for(args.data_root, 'T3')
    _, train, days = next(item for item in t3_windows() if item[0] == args.window)
    regular = load_roster(args.data_root)
    source = source_for(args, days)
    keys, _, _, _, _, _, provenance = checkpoint_contract(args, args.window, train, days, roster, args.seed)
    availability = {}

    def read(workdir, root, dates, sites, progress):
        values, columns, version = archived_issue_weather(workdir, root, dates, sites, progress, issue_fields=EXTRA)
        if columns != COLUMNS:
            raise ValueError('Expanded transfer weather columns changed.')
        extra = values[:, 0, len(NWP_FIELDS)+len(CONTEXT_FIELDS):]
        scope = 'target' if sites == roster else 'source'
        availability[scope] = {'station_days': len(extra), 'cache_version': version,
                               'finite_share': dict(zip(columns[-extra.shape[1]:], np.isfinite(extra).mean(axis=0).tolist()))}
        return values, columns, version

    forecasts, fit = fit_models(args, source, regular, roster, train, days, args.out, seed=args.seed,
                                include_scratch=False, weather_reader=read)
    if set(availability) != {'target', 'source'}:
        raise ValueError('Missing target or source input quality coverage.')
    freeze_window(args.out, 'T3', keys, {EXPANDED: keys.assign(prediction=forecasts[RIDGE].ravel())},
                  {'reference': provenance, 'columns': COLUMNS, 'source_dates': list(map(str, source)),
                   'coverage': availability, 'checkpoint_sha256': sha256(args.out / 'frozen_encoder.pt'), 'fit': fit})


def forecasts(args, window, train, days, roster, seed, interpreter):
    keys, tree, original, baseline, _, _, provenance = checkpoint_contract(args, window, train, days, roster, seed)
    directory = args.out / window / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    trained = directory / 'expanded'
    marker = trained / 'predictions_frozen.json'
    if not marker.is_file():
        staged = Path(mkdtemp(dir=directory, prefix='.weather-attempt-'))
        options = ['--worker', '--window', window, '--seed', seed, '--out', staged]
        for field in ('data_root', 'workdir', 'short_tree', 'transfer_run', 't3_confirm', 'split', 'device'):
            options.extend((f'--{field.replace("_", "-")}', getattr(args, field)))
        call(interpreter, ENTRY, *options)
        staged.rename(trained)
    record = read_json(marker)
    stamp = record['sources']
    if (stamp['reference'] != provenance or stamp['columns'] != COLUMNS
            or stamp['source_dates'] != list(map(str, source_for(args, days)))):
        raise ValueError('Expanded T3 forecast uses a different checkpoint, window or feature schema.')
    forecast = read_frozen_forecast(trained / f'{EXPANDED}.csv', record['hashes'][EXPANDED], keys, 'T3')
    output = {'baseline': baseline, 'candidate': blend(tree, forecast, .25, 'T3'),
              'compact_transfer_single': original, 'expanded_transfer_single': forecast, 'short_tree': tree}
    frames, digest = freeze_window(directory / 'forecasts', 'T3', keys, output,
                                   {'expanded_lock': sha256(marker), 'reference': provenance})
    return keys, frames, digest, stamp['coverage']


def execute(args):
    start = monotonic()
    args.mlp_weather_python = None
    roster = roster_for(args.data_root, 'T3')
    interpreter = select_mlp_runtime(args, require_weather=False)[0]
    code = (ENTRY.name, 'same_issue_weather.py', 'screen_short_transfer.py', 'run_t3_gradual_adaptation.py',
            'warm_shared_power.py', 'screen_same_issue_context.py')
    freeze_json(args.out / 'plan.json', {
        'paths': {name: str(getattr(args, name).resolve()) for name in
                  ('data_root', 'workdir', 'transfer_run', 'short_tree', 't3_confirm')},
        'code': {name: sha256(ENTRY.with_name(name)) for name in code}, 'columns': COLUMNS,
        'split': sha256(args.split), 'device': args.device, 'seed': A_SEED, 'b_seeds': list(B_SEEDS),
        'epoch_budget': 16, 'fraction': .25,
    })
    windows = list(t3_windows())
    early = {}
    for index, (name, train, days) in enumerate(windows[:2]):
        show_progress(args.out, 'T3 扩展气象早窗', index, 2, name)
        _, frames, digest, coverage = forecasts(args, name, train, days, roster, A_SEED, interpreter)
        report = score_t3(labels(args.data_root, days, roster), roster, frames)
        freeze_json(args.out / name / 'score.json', {**report, 'prediction_lock': digest, 'coverage': coverage})
        scores = {arm: item[PRIMARY_SCORE] for arm, item in report['scores'].items()}
        early[name] = {'scores': scores, 'delta': report['delta'], 'station_deltas': report['station_deltas'],
                       'prediction_lock': digest, 'coverage': coverage}
        print(f'[T3 {name}] {scores}，融合差 {report["delta"]:+.6f}', flush=True)
    average = float(np.mean([value['delta'] for value in early.values()]))
    gate = {'promoted': average > 1e-8, 'mean_early_delta': average}
    freeze_json(args.out / 'a_score.json', {'early': early, 'gate': gate})
    summary = {'early': early, 'gate': gate}
    write_json(args.out / 'summary.json', summary)
    show_progress(args.out, 'T3 扩展气象早窗', 2, 2, f'准入 {gate}')
    if gate['promoted']:
        name, train, days = windows[-1]
        folder = args.out / name
        pairs, locks, singles = {}, {}, {}
        for index, seed in enumerate(B_SEEDS):
            show_progress(args.out, 'T3 扩展气象 50/10', index, len(B_SEEDS), f'seed={seed}')
            keys, frames, digest, _ = forecasts(args, name, train, days, roster, seed, interpreter)
            pairs[seed], locks[str(seed)] = freeze_window(folder / str(seed) / 'pair', 'T3', keys,
                {arm: frames[arm] for arm in ('baseline', 'candidate')}, {'forecast_lock': digest, 'gate': gate})
            singles[seed] = frames
        digest = freeze_json(folder / 'predictions_frozen.json', locks)
        truth = labels(args.data_root, days, roster)
        report = paired_report(folder, 'T3', truth, roster, keys, pairs, digest)
        fits = {arm: keys.assign(prediction=np.mean([frames[arm].prediction.to_numpy() for frames in singles.values()],
                                                   axis=0))
                for arm in ('compact_transfer_single', 'expanded_transfer_single', 'short_tree')}
        single = score_t3(truth, roster, {**average_seeds(keys, pairs, 'T3'), **fits})
        freeze_json(folder / 'single_scores.json', {'prediction_lock': digest, 'scores': single['scores']})
        summary['B'] = {'acceptance': report['acceptance'], 'seed_deltas': {
            str(seed): row['delta'] for seed, row in report['pairs'].items()},
            'scores': {arm: row[PRIMARY_SCORE] for arm, row in single['scores'].items()},
            'station_deltas': report['ensemble']['station_deltas'], 'identical_seed_groups': report['identical_seed_groups']}
        show_progress(args.out, 'T3 扩展气象 50/10', len(B_SEEDS), len(B_SEEDS), '计分完成')
    summary['seconds'] = monotonic() - start
    write_json(args.out / 'summary.json', summary)
    print(f'结果摘要 {args.out / "summary.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RUNS / 't3_expanded_weather_gpu_v1')
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--short-tree', type=Path, default=RUNS / 'three_track_gpu_batch_v1/T3')
    parser.add_argument('--transfer-run', type=Path, default=RUNS / 'same_issue_followup_v1/T3')
    parser.add_argument('--t3-confirm', type=Path, default=RUNS / 't3_transfer_confirmation_gpu_v1')
    parser.add_argument('--mlp-run', type=Path, default=RUNS / 'warm_shared_power_v1')
    parser.add_argument('--mlp-python', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--window', choices=[item[0] for item in t3_windows()])
    parser.add_argument('--seed', type=int, choices=B_SEEDS, default=A_SEED)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if args.window is None:
            parser.error('--worker requires --window')
        run_worker(args)
    else:
        with (args.out / '.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(args)


if __name__ == '__main__':
    main()
