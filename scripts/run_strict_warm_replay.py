#!/usr/bin/env python3
"""Matched 365-day fit and May-August historical test, without reading later labels."""
import argparse
from datetime import date, timedelta
import fcntl
from pathlib import Path
import subprocess
from tempfile import mkdtemp
from time import monotonic
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd

from advance_batch_b import keys_for, read_frozen_forecast
from batch_energy_short import energy_features, fit_energy, load_daily_targets, roster_for, weather_day_targets
from compare_t2_mlp import mix, select_mlp_runtime
from compare_warm_t1 import PRIMARY_SCORE
from confirm_working_baselines import keep_sites
from experiment_companion_t1 import assert_matched, read_json
from run_companion_warm_t1 import A_SEED, EARLY_STOP_DAYS, SPLIT_PATH, forecast_keys, run, sha256, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from run_next_matched_batch import amplitude_and_shift, warm_training_weight
from run_steamlink_followup import station_calibration, t2_scores, wind_alternatives
from run_wind_attribution import STATION
from same_issue_weather import archive_stamp, feature_spec
from screen_companion_t1_stations import block_comparisons, show_progress
from score_forecasts import DAY_KEYS, MAX_PREDICTION
from t1_dynamic_wind import NAMES as DYNAMIC_COLUMNS, append_dynamic
from warm_shared_power import fit_forecasts, labels, load_roster, power_weather
from warm_weather_regime import NAMES as WARM_COLUMNS, append_warm_weather


RUNS = Path('/workspace/qhl/runs')
ARMS = ('baseline', 'warm_weather', 'warm_weight')


def replay_dates(split):
    listed = [date.fromisoformat(value) for value in pd.read_csv(split, dtype=str)['date']]
    start, middle, end = date(2024, 5, 1), date(2025, 5, 1), date(2025, 9, 1)
    train = [start + timedelta(days=i) for i in range((middle - start).days)]
    days = [middle + timedelta(days=i) for i in range((end - middle).days)]
    if len(set(listed)) != len(listed) or not set(train + days).issubset(listed):
        raise ValueError('The registered data manifest does not cover the complete warm replay.')
    return train, days


def scope(args, train, days):
    return {'train': list(map(str, train)), 'forecast': list(map(str, days)), 'seed': A_SEED,
            'device': args.device, 'threads': args.threads, 'data_root': str(args.data_root.resolve())}


def t1_tree(args, train, days, keys, arm):
    directory = args.out / 'T1' / 'tree'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / arm
    marker = target / 'training_receipt.json'

    def features(matrix, names, dates, sites, positions):
        matrix, names = append_dynamic(args.data_root, args.dynamic_run / 'native_weather',
                                       matrix, names, dates, sites, positions)
        if arm == 'warm_weather':
            return append_warm_weather(args.workdir, matrix, names, dates, sites, positions)
        return matrix, names

    if not marker.is_file():
        pending = Path(mkdtemp(dir=directory, prefix=f'.{arm}-attempt-'))
        options = SimpleNamespace(**vars(args))
        options.out, options.seed, options.phase = pending, A_SEED, 'A'
        options.train_block, options.smoke, options.check_only = None, False, False
        options.backend, options.components, options.point_partition = 'xgboost', [], 'full'
        options.early_stop_days, options.diagnostic = EARLY_STOP_DAYS, None
        run(options, extra_features=features, date_partition=(train, days),
            training_weight=warm_training_weight if arm == 'warm_weight' else None)
        pending.rename(target)
    info = read_json(marker)
    dates = scope(args, train, days)
    expected = {'training': dates['train'], 'forecast': dates['forecast']}
    if info['chronological_dates'] != expected or info['seed'] != A_SEED:
        raise ValueError('T1 tree receipt uses a different warm replay date plan.')
    if arm == 'warm_weight' and not info['experts'][0].get('training_weight'):
        raise ValueError('T1 seasonal weight receipt is missing.')
    return read_frozen_forecast(target / info['prediction_file'], info['prediction_sha256'], keys, 'T1'), info, sha256(marker)


def windows(days):
    yield 'may_jul', [day for day in days if day.month <= 7]
    for month in (5, 6, 7, 8):
        yield f'2025-{month:02d}', [day for day in days if day.month == month]


def subset(frame, days):
    return frame.loc[frame.date.isin(list(map(str, days)))]


def t1_predict(args, train, days):
    folder = args.out / 'T1'
    sites = load_roster(args.data_root)
    keys = forecast_keys(days, sites)
    wind, _, _, weather = wind_alternatives(args, train, days, sites, selected='baseline')
    power = keys.assign(prediction=wind.ravel())
    result, receipt = {}, {}
    for arm in ARMS:
        print(f'[T1 warm] fit or reuse {arm}', flush=True)
        tree, metadata, receipt[arm] = t1_tree(args, train, days, keys, arm)
        if arm == 'baseline':
            base_tree, base_meta = tree, metadata
        else:
            assert_matched(base_meta, metadata, ('seed', 'chronological_dates', 'companion_cache_metadata_sha256',
                'early_stopping_days_inside_train', 'early_stopping_valid_rows', 'device', 'threads',
                'feature_spec_sha256', 'config_sha256', 'valid_training_sample_sha256', 'companion_source_sha256'))
            added = WARM_COLUMNS if arm == 'warm_weather' else []
            if metadata['feature_names'] != base_meta['feature_names'] + added:
                raise ValueError(f'T1 warm {arm}: feature mismatch.')
            tree = keep_sites(tree, base_tree, [STATION])
        if not set(DYNAMIC_COLUMNS).issubset(metadata['feature_names']):
            raise ValueError('T1 replay lost dynamic gradient columns.')
        result[arm] = blend(tree, power, .2, 'T1')
    forecasts, lock = freeze_window(folder / 'forecasts', 'T1', keys, result, {'trees': receipt, 'weather': weather})
    return sites, keys, forecasts, lock


def t1_report(args, days, bundle):
    sites, keys, forecasts, lock = bundle
    truth = labels(args.data_root, days, sites)
    scores = {}
    for name, dates in [('all', days), *windows(days)]:
        detail = block_comparisons(subset(truth, dates),
            {arm: subset(frame, dates) for arm, frame in forecasts.items()}, sites, ARMS[1:])
        summary = {arm: row[PRIMARY_SCORE] for arm, row in detail['scores'].items()}
        scores[name] = {'scores': summary, 'delta': {a: summary[a]-summary['baseline'] for a in ARMS[1:]},
                        'station_pairs': detail['station_pairs'],
                        'scored_station_days': detail['scores']['baseline']['scored_station_days']}
    result = {'prediction_lock': lock, 'windows': scores, 'shift_oracles': amplitude_and_shift(truth, forecasts, keys)}
    freeze_json(args.out / 'T1' / 'score.json', result)
    print(f'[T1 warm] {scores["all"]["scores"]}', flush=True)
    return result


def fit_scope(train, days, calibration):
    tail = len(train)//5
    return (train[:-tail], train[-tail:]) if calibration else (train, days)


def mlp_worker(args, train, days):
    fit, forecast = fit_scope(train, days, args.calibration)
    if args.worker == 'weather':
        power_weather(args.data_root, fit, forecast, args.mlp_cache, 2, print)
    else:
        fit_forecasts(args.data_root, args.worker_output, fit, forecast, 2, 16, 64, args.device,
                      cache_root=args.mlp_cache, seed=A_SEED)


def mlp_curve(args, train, days, sites, runtime, calibration, folder):
    use, score = fit_scope(train, days, calibration)
    directory = folder / 'mlp'
    plan = scope(args, use, score)
    if not directory.exists():
        temp = Path(mkdtemp(dir=folder, prefix='.mlp-attempt-'))
        interpreter, weather_python = runtime
        options = [str(Path(__file__).resolve()), '--data-root', str(args.data_root), '--split', str(args.split),
                   '--mlp-cache', str(args.mlp_cache), '--device', args.device, '--threads', str(args.threads),
                   '--worker-output', str(temp)]
        if calibration:
            options += ['--calibration']
        if interpreter != weather_python:
            subprocess.run([str(weather_python), '-u', *options, '--worker', 'weather'], check=True)
        subprocess.run([str(interpreter), '-u', *options, '--worker', 'mlp'], check=True)
        file = temp / 'T2_development.csv'
        freeze_json(temp / 'receipt.json', {'plan': plan, 'forecast': sha256(file),
                                           'training': sha256(temp / 'training_receipt.json')})
        temp.rename(directory)
    meta = read_json(directory / 'receipt.json')
    if meta['plan'] != plan or meta['training'] != sha256(directory / 'training_receipt.json'):
        raise ValueError('Warm T2 MLP cache has different dates or receipt.')
    keys = keys_for(score, sites, 'T2')
    frame = read_frozen_forecast(directory / 'T2_development.csv', meta['forecast'], keys, 'T2')
    return frame, sha256(directory / 'receipt.json')


def t2_component(args, train, days, sites, runtime, calibration):
    use, score = fit_scope(train, days, calibration)
    folder = args.out / 'T2' / ('calibration' if calibration else 'full')
    folder.mkdir(parents=True, exist_ok=True)
    mlp, mlp_lock = mlp_curve(args, train, days, sites, runtime, calibration, folder)
    tree = folder / 'tree'
    fit, fit_info = energy_features(args, sites, use, feature_spec(args.workdir))
    future, future_info = energy_features(args, sites, score, feature_spec(args.workdir))
    if fit_info['columns'] != future_info['columns']:
        raise ValueError('Warm replay T2 feature mismatch.')
    sig = {**scope(args, use, score), 'features': [fit_info['sha256'], future_info['sha256']]}
    if not (tree / 'receipt.json').is_file():
        temp = Path(mkdtemp(dir=folder, prefix='.tree-attempt-'))
        _, actual = load_daily_targets(weather_day_targets(args.data_root, use), sites)
        table = fit.merge(actual, on=DAY_KEYS, validate='one_to_one')
        if len(table) != len(actual):
            raise ValueError('Warm T2 fit table lost eligible rows.')
        raw, _, receipt = fit_energy(args, fit_info['columns'], table, use, future, 'relative_mse', temp, seed=A_SEED)
        target = temp / 'forecast.csv'
        future[DAY_KEYS].assign(energy=raw).to_csv(target, index=False)
        freeze_json(temp / 'receipt.json', {'signature': sig, 'fit': receipt, 'forecast': sha256(target)})
        temp.rename(tree)
    meta = read_json(tree / 'receipt.json')
    if meta['signature'] != sig:
        raise ValueError('Warm T2 tree cache has different inputs.')
    keys = keys_for(score, sites, 'T2')
    direct = read_frozen_forecast(tree / 'forecast.csv', meta['forecast'], keys, 'T2')
    return keys, mix(mlp, direct), {'mlp': mlp_lock, 'tree': sha256(tree/'receipt.json')}


def energy_arms(raw, factors):
    multiplier = raw.station_id.map(factors).fillna(1.)
    protected = multiplier.where(raw.energy.lt(180.), 1. + .5 * (multiplier - 1.))
    return {'baseline': raw,
            'station_calibrated': raw.assign(energy=(raw.energy * multiplier).clip(0, MAX_PREDICTION['T2'])),
            'high_energy_guard': raw.assign(energy=(raw.energy * protected).clip(0, MAX_PREDICTION['T2']))}


def t2_predict(args, train, days):
    sites = roster_for(args.data_root, 'T2')
    if sites != load_roster(args.data_root):
        raise ValueError('Warm replay T1 and T2 station rosters do not align.')
    runtime = select_mlp_runtime(args)
    print('[T2 warm] full forecast', flush=True)
    keys, raw, full = t2_component(args, train, days, sites, runtime, False)
    print('[T2 warm] training-only calibration', flush=True)
    _, past, historical = t2_component(args, train, days, sites, runtime, True)
    _, cal_days = fit_scope(train, days, True)
    _, actual = load_daily_targets(weather_day_targets(args.data_root, cal_days), sites)
    factors, counts = station_calibration(past, actual)
    forecasts, lock = freeze_window(args.out / 'T2/forecasts', 'T2', keys, energy_arms(raw, factors),
        {'fit': full, 'training_calibration': historical, 'factors': factors, 'scored_training_days': counts,
         'high_energy_guard': {'predicted_threshold': 180., 'shrinkage_toward_one': .5}})
    return sites, forecasts, lock


def t2_report(args, days, bundle):
    sites, forecasts, lock = bundle
    scores = {}
    for name, dates in [('all', days), *windows(days)]:
        details, groups = t2_scores(args, dates, sites, {arm: subset(frame, dates) for arm, frame in forecasts.items()})
        summary = {arm: row[PRIMARY_SCORE] for arm, row in details.items()}
        scores[name] = {'scores': summary, 'delta': {a: value - summary['baseline'] for a, value in summary.items()},
                        'scored_station_days': details['baseline']['scored_station_days'], 'groups': groups}
    result = {'prediction_lock': lock, 'windows': scores}
    freeze_json(args.out / 'T2' / 'score.json', result)
    print(f'[T2 warm] {scores["all"]["scores"]}', flush=True)
    return result


def run_replay(args, train, days):
    start = monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_json(args.out / 'plan.json', {'dates': scope(args, train, days), 'tracks': args.tracks,
            'split_sha256': sha256(args.split), 'cache': archive_stamp(args.workdir, train + days),
            'tree_config': sha256(args.config or args.companion_code / 'configs/default.json') if 'T1' in args.tracks else None,
            'code': {name: sha256(Path(__file__).with_name(name)) for name in (
                'run_strict_warm_replay.py', 'run_companion_warm_t1.py', 'warm_weather_regime.py',
                'warm_shared_power.py', 'run_next_matched_batch.py', 'batch_energy_short.py')}})
        stages = {'T1': (t1_predict, t1_report), 'T2': (t2_predict, t2_report)}
        stages = {track: fns for track, fns in stages.items() if track in args.tracks}
        result, frozen = {}, {}
        for i, (track, (predict, _)) in enumerate(stages.items()):
            show_progress(args.out, '暖季回放', i, 2 * len(stages), f'{track} 预测')
            try:
                frozen[track] = predict(args, train, days)
                result[track] = {'status': 'forecasts_locked'}
            except Exception as exc:
                traceback.print_exc()
                result[track] = {'status': 'failed', 'error': str(exc)}
            write_json(args.out / 'summary.json', result)
        for i, (track, (_, score)) in enumerate(stages.items()):
            show_progress(args.out, '暖季回放', len(stages) + i, 2 * len(stages), f'{track} 计分')
            if track in frozen:
                try:
                    result[track] = {'status': 'complete', **score(args, days, frozen[track])}
                except Exception as exc:
                    traceback.print_exc()
                    result[track] = {'status': 'failed', 'error': str(exc)}
            write_json(args.out / 'summary.json', result)
        result['seconds'] = round(monotonic() - start, 2)
        write_json(args.out / 'summary.json', result)
        show_progress(args.out, '暖季回放', 2 * len(stages), 2 * len(stages), '已保存')
        print(f'结果摘要 {args.out / "summary.json"}', flush=True)
        return int(any(result[t]['status'] != 'complete' for t in stages))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out': RUNS/'strict_warm_replay_gpu_v1', 'data-root': Path('/workspace/qhl/data'),
        'workdir': Path('/workspace/cpy/code/workdir'), 'companion-code': Path('/workspace/cpy/code'),
        'dynamic-run': RUNS/'t1_dynamic_gradient_gpu_v1', 'split': SPLIT_PATH,
        'mlp-run': RUNS/'warm_shared_power_v1', 'mlp-cache': RUNS/'warm_power_feature_cache'}
    for key, value in defaults.items():
        parser.add_argument(f'--{key}', type=Path, default=value)
    for key in ('config', 'mlp-python', 'mlp-weather-python', 'worker-output'):
        parser.add_argument(f'--{key}', type=Path)
    parser.add_argument('--tracks', nargs='+', choices=('T1','T2'), default=['T1','T2'])
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--worker', choices=('weather','mlp'))
    parser.add_argument('--calibration', action='store_true')
    args = parser.parse_args()
    train, days = replay_dates(args.split)
    if args.worker:
        mlp_worker(args, train, days)
        return 0
    return run_replay(args, train, days)


if __name__ == '__main__':
    raise SystemExit(main())
