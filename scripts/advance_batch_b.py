#!/usr/bin/env python3
"""Promote positive matched A candidates to B, then predict, lock and score."""

import argparse
import fcntl
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from batch_energy_short import ENERGY_MODELS, energy_arm_report, energy_features, fit_energy, save_csv, weather_day_targets
from compare_warm_t1 import PRIMARY_SCORE, b_statistics, compare_pair
from experiment_companion_t1 import (
    assert_matched, code_version, event, fit_or_reuse, identical_seed_groups, phase_versions, read_json,
)
from independent_t2 import load_daily_targets
from run_companion_warm_t1 import B_SEEDS, SPLIT_PATH, cache_paths, cache_version, date_roles, forecast_keys, sha256, write_json
from score_forecasts import DAY_KEYS, MAX_PREDICTION, POINT_KEYS, align_predictions
from screen_companion_t1_stations import A_REPORT, FREEZE, STATION_RECIPE_FIELDS, combine_forecasts, fit_options, show_progress
from tree_model import check_backend
from warm_shared_power import labels, load_roster


ENERGY_ARMS = ('t1_integral', *(variant for mode in ENERGY_MODELS for variant in (mode, f'{mode}_calibrated')))
ARMS = ('baseline', 'candidate')


def choose_gate(scores, reference, allowed):
    if reference not in allowed or not set(allowed).issubset(scores):
        raise ValueError('The A report is missing registered forecast arms.')
    points = {name: scores[name][PRIMARY_SCORE] for name in allowed}
    if not all(isfinite(value) for value in points.values()):
        raise ValueError('Only finite A scores can be used for B promotion.')
    candidate = max((name for name in allowed if name != reference), key=points.__getitem__)
    delta = points[candidate] - points[reference]
    return {'baseline': reference, 'candidate': candidate, 'a_baseline': points[reference],
            'a_candidate': points[candidate], 'a_delta': delta,
            'status': 'promoted' if delta > 0 else 'not_positive'}


def a_decisions(batch, allowed_a_dates):
    decisions = {}
    t1 = batch / 'T1' / A_REPORT
    t2 = batch / 'T2' / 'results.json'
    if t1.is_file():
        freeze_path = batch / 'T1' / FREEZE
        report = read_json(t1)
        freeze = read_json(freeze_path)
        if (report['freeze_sha256'] != sha256(freeze_path) or report['routes'] != freeze['routes']
                or report['training']['baseline']['forecast_days'] != len(allowed_a_dates)):
            raise ValueError('T1 A result differs from its frozen route or A grid.')
        gate = choose_gate(report['scores'], 'baseline', ARMS)
        if gate['status'] == 'promoted' and not report['routes']:
            raise ValueError('A positive T1 comparison must reference a nonempty frozen route.')
        decisions['T1'] = dict(gate, routes=report['routes'], a_report_sha256=sha256(t1),
                               recipe=freeze['recipe'])
    else:
        decisions['T1'] = {'status': 'awaiting_a'}
    if t2.is_file():
        report = read_json(t2)
        window = report['windows']['A']
        if window['window'] != [day.isoformat() for day in allowed_a_dates]:
            raise ValueError('T2 report uses a different A window.')
        gate = choose_gate(window['scores'], report['reference'], ENERGY_ARMS)
        decisions['T2'] = dict(gate, a_report_sha256=sha256(t2), recipe=report['settings'])
    else:
        decisions['T2'] = {'status': 'awaiting_a'}
    decisions['T3'] = {'status': 'fixed_50_10'}
    return decisions


def keys_for(days, roster, track):
    if track == 'T1':
        return forecast_keys(days, roster)
    return pd.MultiIndex.from_product((roster, [day.isoformat() for day in days]), names=DAY_KEYS).to_frame(index=False)


def read_frozen_forecast(path, digest, keys, track):
    if sha256(path) != digest:
        raise ValueError(f'{path}: B forecast hash mismatch.')
    value = 'prediction' if track == 'T1' else 'energy'
    fields = POINT_KEYS if track == 'T1' else DAY_KEYS
    frame = pd.read_csv(path, dtype={'station_id': str, 'date': str}, float_precision='round_trip')
    return keys.assign(**{value: align_predictions(keys, frame, fields, value, MAX_PREDICTION[track])})


def pooled_curve(args, seed, keys, versions):
    directory = args.out / 'pooled_curves' / str(seed)
    return fit_or_reuse(args, directory, keys, 'B', seed, [], versions)[:2]


def make_t1(args, train_days, days, roster, gate):
    keys = keys_for(days, roster, 'T1')
    versions = phase_versions(args, train_days, days)
    routes = gate['routes']

    def fit(seed, directory):
        receipt, baseline = pooled_curve(args, seed, keys, versions)
        assert_matched(gate['recipe'], receipt, STATION_RECIPE_FIELDS)
        candidates, receipts = {}, {'baseline': receipt}
        for name in sorted(set(routes.values())):
            components, partition = fit_options(name)
            other, forecast, _ = fit_or_reuse(args, directory / name, keys, 'B', seed, components, versions,
                                              point_partition=partition)
            assert_matched(receipt, other)
            candidates[name], receipts[name] = forecast, other
        return {'baseline': baseline, 'candidate': combine_forecasts(baseline, candidates, routes)}, receipts

    return fit


def warm_energy_inputs(args, roster, train_days, days, source):
    version = source['features']
    path = args.batch / 'T2' / 'energy_features' / f'{version["version"]}.csv'
    spec_path = args.workdir / 'stats/feature_spec_train.json'
    if (sha256(spec_path) != version['feature_spec_sha256']
            or cache_version(cache_paths(args.workdir, train_days)) != version['weather_cache_version']):
        raise ValueError('Daily train+A feature sources differ from the A reference.')
    if path.is_file():
        if sha256(path) != version['sha256']:
            raise ValueError('Daily training features differ from the A reference.')
        train = pd.read_csv(path, dtype={'station_id': str, 'date': str}, float_precision='round_trip')
        train_meta = version
    else:
        train, train_meta = energy_features(args, roster, train_days, spec_path)
    wanted = pd.MultiIndex.from_frame(keys_for(train_days, roster, 'T2'))
    actual = pd.MultiIndex.from_frame(train[DAY_KEYS])
    if actual.has_duplicates or len(actual) != len(wanted) or len(wanted.difference(actual)):
        raise ValueError('Daily cache differs from the registered 628-day train+A grid.')
    forecast, b_meta = energy_features(args, roster, days, spec_path)
    if b_meta['columns'] != version['columns'] or list(train) != [*b_meta['columns'], 'date', 'station_id']:
        raise ValueError('Daily training and B feature schemas differ.')
    return train, forecast, b_meta['columns'], {'train_plus_a': train_meta, 'b_forecast': b_meta}


def make_t2(args, train_days, days, roster, gate):
    source = read_json(args.batch / 'T2' / 'results.json')
    train, forecast, features, meta = warm_energy_inputs(args, roster, train_days, days, source)
    _, eligible = load_daily_targets(weather_day_targets(args.data_root, train_days), roster)
    training = train.merge(eligible, on=DAY_KEYS, validate='one_to_one')
    if len(training) != len(eligible):
        raise ValueError('Daily train+A features do not cover the eligible T2 samples.')
    requested = {gate[arm] for arm in ARMS}

    def fit(seed, directory):
        fits, predictions = {'daily_cache': meta}, {}
        for mode in ENERGY_MODELS:
            if requested.intersection((mode, f'{mode}_calibrated')):
                raw, cal, fits[mode] = fit_energy(args, features, training, train_days, forecast, mode, directory,
                                                  seed=seed)
                predictions[mode] = forecast[DAY_KEYS].assign(energy=raw)
                predictions[f'{mode}_calibrated'] = forecast[DAY_KEYS].assign(energy=cal)
        if 't1_integral' in requested:
            versions = phase_versions(args, train_days, days)
            curve_receipt, curve = pooled_curve(args, seed, keys_for(days, roster, 'T1'), versions)
            daily = curve.groupby(DAY_KEYS, sort=False, as_index=False).prediction.sum()
            daily['energy'] = (daily.pop('prediction') / 4).clip(0, MAX_PREDICTION['T2'])
            predictions['t1_integral'] = daily
            fits['t1_integral'] = curve_receipt
        return {arm: predictions[gate[arm]] for arm in ARMS}, fits

    return fit


def lock_predictions(args, track, gate, recipe, train, days, roster, *, build_fit=None, refresh_unstarted=False):
    directory = args.out / track
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / 'plan.json'
    settings = {'track': track, 'gate': gate, 'runtime': recipe, 'seeds': list(B_SEEDS),
                'train_days': len(train), 'prediction_days': len(days),
                'forecast_cache_version': cache_version(cache_paths(args.workdir, days)),
                'fit_cache_version': cache_version(cache_paths(args.workdir, train))}
    if plan_path.exists():
        previous = read_json(plan_path)
        if previous != settings:
            equivalent = {**previous, 'runtime': {**previous['runtime'], 'code_sha256': recipe.get('code_sha256')}}
            outputs = [path for path in directory.iterdir()
                       if path != plan_path and not path.name.startswith('plan.unstarted.')]
            if not refresh_unstarted or equivalent != settings or outputs:
                raise ValueError(f'{track}: the B recipe changed after freezing.')
            archive = directory / f'plan.unstarted.{sha256(plan_path)[:12]}.json'
            write_json(archive, previous)
            write_json(plan_path, settings)
            print(f'[{track} B] 已保留未开训的旧计划 {archive.name}', flush=True)
    if not plan_path.exists():
        write_json(plan_path, settings)
    plan_digest = sha256(plan_path)
    keys = keys_for(days, roster, track)
    all_inputs = {}
    fit = None
    show_progress(args.out, f'{track} B 预测', 0, len(B_SEEDS), '核对缓存和预测计划')
    for step, seed in enumerate(B_SEEDS, 1):
        stage = directory / str(seed)
        path = stage / 'receipt.json'
        if not path.is_file():
            print(f'[{track} B {step}/{len(B_SEEDS)}] seed={seed} 开始拟合', flush=True)
            if fit is None:
                if build_fit is None:
                    build_fit = make_t1 if track == 'T1' else make_t2
                fit = build_fit(args, train, days, roster, gate)
            stage.mkdir(parents=True, exist_ok=True)
            predictions, details = fit(seed, stage)
            hashes = {arm: save_csv(stage / f'{arm}.csv', predictions[arm]) for arm in ARMS}
            for arm, digest in hashes.items():
                read_frozen_forecast(stage / f'{arm}.csv', digest, keys, track)
            write_json(path, {'seed': seed, 'plan_sha256': plan_digest, 'forecasts': hashes, 'training': details})
        record = read_json(path)
        if record['seed'] != seed or record['plan_sha256'] != plan_digest or set(record['forecasts']) != set(ARMS):
            raise ValueError(f'{track}/{seed}: B receipt does not match the frozen plan.')
        for arm in ARMS:
            read_frozen_forecast(stage / f'{arm}.csv', record['forecasts'][arm], keys, track)
        all_inputs[str(seed)] = {'receipt_sha256': sha256(path), 'forecasts': record['forecasts']}
        show_progress(args.out, f'{track} B 预测', step, len(B_SEEDS), f'{seed} 两组预测已保存核验')
    manifest_path = directory / 'predictions_frozen.json'
    manifest = {'plan_sha256': plan_digest, 'seeds': all_inputs, 'expected_rows': len(keys)}
    if manifest_path.is_file() and read_json(manifest_path) != manifest:
        raise ValueError(f'{track}: B forecasts changed after locking.')
    if not manifest_path.is_file():
        write_json(manifest_path, manifest)
    return sha256(manifest_path)


def before_scoring(directory, track, keys, manifest_digest):
    manifest_path = directory / 'predictions_frozen.json'
    if sha256(manifest_path) != manifest_digest:
        raise ValueError(f'{track}: B prediction lock changed.')
    manifest = read_json(manifest_path)
    if (manifest['plan_sha256'] != sha256(directory / 'plan.json')
            or manifest['expected_rows'] != len(keys) or set(manifest['seeds']) != {str(seed) for seed in B_SEEDS}):
        raise ValueError(f'{track}: B prediction lock is incomplete.')
    predictions = {}
    for seed in B_SEEDS:
        stage = directory / str(seed)
        locked = manifest['seeds'][str(seed)]
        if sha256(stage / 'receipt.json') != locked['receipt_sha256'] or set(locked['forecasts']) != set(ARMS):
            raise ValueError(f'{track}/{seed}: B fit receipt or forecast set changed.')
        predictions[seed] = {arm: read_frozen_forecast(stage / f'{arm}.csv', locked['forecasts'][arm], keys, track)
                             for arm in ARMS}
    return predictions


def paired_scores(args, track, gate, days, roster, predictions, manifest_digest):
    if track == 'T1':
        truth = labels(args.data_root, days, roster)

        def score(arms):
            return compare_pair(truth, arms['baseline'], arms['candidate'], roster)

    else:
        truth, eligible = load_daily_targets(weather_day_targets(args.data_root, days), roster)

        def score(arms):
            scores = energy_arm_report(truth, eligible, arms, roster)
            return {'scores': scores, 'delta': {metric: scores['candidate'][metric] - scores['baseline'][metric]
                                                for metric in (PRIMARY_SCORE, 'pdf_clipped_score')}}

    pairs = {seed: score(arms) for seed, arms in predictions.items()}
    value = 'prediction' if track == 'T1' else 'energy'
    ensemble = {}
    for arm in ARMS:
        values = np.mean([predictions[seed][arm][value].to_numpy() for seed in B_SEEDS], axis=0)
        ensemble[arm] = predictions[B_SEEDS[0]][arm].assign(**{value: np.clip(values, 0, MAX_PREDICTION[track])})
    final = score(ensemble)
    deltas = [pairs[seed]['delta'][PRIMARY_SCORE] for seed in B_SEEDS]
    paired = b_statistics(deltas, final['delta'][PRIMARY_SCORE])
    result = {'track': track, 'gate': gate, 'prediction_lock_sha256': manifest_digest, 'seeds': list(B_SEEDS),
              'pairs': pairs, 'ensemble': final,
              'identical_seed_groups': {
                  arm: identical_seed_groups({seed: predictions[seed][arm][value].to_numpy() for seed in B_SEEDS})
                  for arm in ARMS
              }}
    if track == 'T1':
        result['acceptance'] = paired
    else:
        result['paired_summary'] = {key: paired[key] for key in ('mean_delta', 'sample_sd', 'paired_t_95_interval',
                                                               'identical_seed_deltas', 'ensemble_delta')}
    return result


def execute(args):
    train, days = date_roles(args.split, 'B')
    _, a_days = date_roles(args.split, 'A')
    if len(train) != 628 or len(days) != 92:
        raise ValueError('Expected the registered 628/92 B split.')
    decisions = a_decisions(args.batch, a_days)
    gate_path = args.out / 'auto_gate.json'
    if gate_path.is_file() and read_json(gate_path) != decisions:
        raise ValueError('A enrollment changed after freezing B.')
    for track, row in decisions.items():
        detail = ''
        if 'a_delta' in row:
            detail = f' {row["candidate"]} 对 {row["baseline"]}，A 差 {row["a_delta"]:+.6f}'
        print(f'[B 准入] {track} {row["status"]}{detail}', flush=True)
    if any(row['status'] == 'awaiting_a' for row in decisions.values()):
        raise ValueError('Finish both warm A reports before freezing the B enrollment.')
    if not gate_path.is_file():
        write_json(gate_path, decisions)
    promoted = {track: row for track, row in decisions.items() if row['status'] == 'promoted'}
    if not promoted:
        return
    config = args.config or args.companion_code / 'configs/default.json'
    runtime = {'backend': 'xgboost', 'device': args.device, 'threads': args.threads,
               'backend_version': check_backend('xgboost', args.device),
               'data_root': str(args.data_root.resolve()), 'split_sha256': sha256(args.split),
               'workdir': str(args.workdir.resolve()), 'companion_config_sha256': sha256(config),
               'feature_spec_sha256': sha256(args.workdir / 'stats/feature_spec_train.json'),
               'code_sha256': {**code_version(), Path(__file__).name: sha256(Path(__file__)),
                               'batch_energy_short.py': sha256(Path(fit_energy.__code__.co_filename))}}
    roster = load_roster(args.data_root)
    manifests = {}
    for track, gate in promoted.items():
        fields = ['data_root', 'split_sha256', 'backend', 'device', 'threads', 'backend_version',
                  'feature_spec_sha256']
        if track == 'T2':
            fields.extend(('workdir', 'companion_config_sha256'))
        assert_matched(runtime, gate['recipe'], fields)
        manifests[track] = lock_predictions(args, track, gate, runtime, train, days, roster)

    # Load and verify every promoted arm before opening the validation labels of any track.
    ready = {track: before_scoring(args.out / track, track, keys_for(days, roster, track), digest)
             for track, digest in manifests.items()}
    event(args.out, 'B', status='all_predictions_verified', tracks=list(ready))
    for track, predictions in ready.items():
        path = args.out / track / 'score.json'
        if path.is_file():
            result = read_json(path)
            if result['prediction_lock_sha256'] != manifests[track]:
                raise ValueError(f'{track}: existing B score refers to a different prediction lock.')
        else:
            result = paired_scores(args, track, promoted[track], days, roster, predictions, manifests[track])
            write_json(path, result)
        delta = result['ensemble']['delta'][PRIMARY_SCORE]
        base, candidate = (result['ensemble']['scores'][arm][PRIMARY_SCORE] for arm in ARMS)
        print(f'[B] {track} 基线 {base:.6f}，候选 {candidate:.6f}，差 {delta:+.6f}。报告 {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=Path('/workspace/qhl/runs/three_track_gpu_batch_v1'))
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--local-wind-cache', type=Path, default=Path('/workspace/qhl/runs/companion_t1_local_wind'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error('Use 1 to 4 CPU threads.')
    args.backend = 'xgboost'
    args.out = args.batch / 'B'
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args)
        except (Exception, KeyboardInterrupt) as error:
            event(args.out, 'B', status='failed', detail=str(error))
            raise
        event(args.out, 'B', status='complete')


if __name__ == '__main__':
    main()
