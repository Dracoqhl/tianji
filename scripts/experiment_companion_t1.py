#!/usr/bin/env python3
"""Run the registered companion T1 A screen and the two separate B stages."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from companion_t1_features import COMPONENTS
from companion_upwind_terrain import COMPONENT as TERRAIN
from compare_warm_t1 import MINIMUM_GAIN, PRIMARY_SCORE, b_statistics, compare_pair, diagnostic_errors
from run_companion_warm_t1 import A_SEED, B_SEEDS, DEFAULT_THREADS, EARLY_STOP_DAYS, GPU_THREADS, SPLIT_PATH
from run_companion_warm_t1 import cache_paths, cache_version, date_roles, forecast_keys, sha256, write_json
from run_companion_warm_t1 import block_early_stop_days, training_block_dates
from score_forecasts import MAX_PREDICTION, POINT_KEYS, align_predictions, score_track
from tree_model import check_backend
from warm_shared_power import FROZEN_SPLIT, labels, load_roster


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / 'run_companion_warm_t1.py'
REFERENCE_A = 85.80089361353807
A_FREEZE = 'a_freeze.json'
B_FREEZE = 'b_predictions_frozen.json'
MATCH_FIELDS = ('phase', 'train_block', 'backend', 'device', 'backend_version', 'data_root', 'feature_workdir',
                'feature_spec_sha256', 'split_sha256', 'config_sha256', 'companion_source_sha256',
                'threads', 'train_days', 'forecast_days',
                'early_stopping_days_inside_train', 'early_stopping_start_date', 'early_stopping_valid_rows',
                'companion_cache_metadata_sha256', 'valid_training_sample_sha256')
RECIPE_FIELDS = ('backend', 'device', 'backend_version', 'data_root', 'feature_spec_sha256',
                 'split_sha256', 'config_sha256', 'companion_source_sha256', 'threads',
                 'early_stopping_days_inside_train')


def read_json(path):
    return json.loads(path.read_text())


def event(out, stage, **fields):
    with (out / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps({'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                                 'stage': stage, **fields}, allow_nan=False) + '\n')


def code_version():
    files = ('experiment_companion_t1.py', RUNNER.name, 'companion_t1_features.py', 'compare_warm_t1.py',
             'warm_shared_power.py', 'independent_t2.py', 'score_forecasts.py', 'tree_model.py',
             'companion_spatial_features.py', 'companion_upwind_terrain.py')
    return {name: sha256(SCRIPT_DIR / name) for name in files}


def setup(args):
    config_path = args.config or args.companion_code / 'configs/default.json'
    return {
        'data_root': str(args.data_root.resolve()), 'workdir': str(args.workdir.resolve()),
        'companion_code': str(args.companion_code.resolve()), 'config_sha256': sha256(config_path),
        'local_wind_cache': str(args.local_wind_cache.resolve()), 'split_sha256': sha256(args.split),
        'backend': args.backend, 'device': args.device, 'backend_version': check_backend(args.backend, args.device),
        'threads': args.threads, 'early_stop_days': EARLY_STOP_DAYS, 'code_sha256': code_version(),
        'a_seed': A_SEED, 'b_seeds': list(B_SEEDS), 'components': list(COMPONENTS), 'minimum_gain': MINIMUM_GAIN,
    }


def load_run(directory, expected_keys, phase, seed, components, *, backend, device, block_scope=None,
             point_partition='full'):
    receipt_path = directory / 'training_receipt.json'
    receipt = read_json(receipt_path)
    expected_file = 'companion_T1_development.csv' if phase == 'A' else 'companion_T1_sealed.csv'
    expected = {'phase': phase, 'seed': seed, 'components': components, 'split_sha256': FROZEN_SPLIT,
                'backend': backend, 'device': device, 'smoke': False,
                'train_block': None, 'point_partition': point_partition,
                'train_days': 586 if phase == 'A' else 628, 'forecast_days': 42 if phase == 'A' else 92,
                'prediction_file': expected_file, 'prediction_rows': len(expected_keys)}
    if block_scope:
        expected.update(block_scope)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{receipt_path}: training receipt differs from the registered run.')
    path = directory / receipt['prediction_file']
    if sha256(path) != receipt['prediction_sha256']:
        raise ValueError(f'{path}: forecast hash mismatch.')
    original = pd.read_csv(path, dtype={'station_id': str, 'date': str}, float_precision='round_trip')
    forecast = expected_keys.assign(prediction=align_predictions(
        expected_keys, original, POINT_KEYS, 'prediction', MAX_PREDICTION['T1'],
    ))
    return receipt, forecast, sha256(receipt_path)


def assert_matched(first, second, fields=MATCH_FIELDS):
    differences = [key for key in fields if first[key] != second[key]]
    if differences:
        raise ValueError(f'The compared training receipts differ in matched inputs: {differences}.')


def fit_or_reuse(args, directory, keys, phase, seed, components, versions, diagnostic=None, *, train_block=None,
                 point_partition='full'):
    block_scope = None
    early_stop_days = EARLY_STOP_DAYS
    if train_block is not None:
        train, holdout = training_block_dates(args.split, *train_block)
        early_stop_days = block_early_stop_days(train)
        block_scope = {'train_block': [day.isoformat() for day in train_block],
                       'train_days': len(train), 'forecast_days': len(holdout),
                       'early_stopping_days_inside_train': early_stop_days}
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(mkdtemp(dir=directory.parent, prefix=f'.{directory.name}.attempt-'))
        command = [sys.executable, '-u', str(RUNNER), '--phase', phase, '--seed', str(seed),
                   '--companion-code', str(args.companion_code), '--workdir', str(args.workdir),
                   '--data-root', str(args.data_root), '--out', str(staging), '--split', str(args.split),
                   '--local-wind-cache', str(args.local_wind_cache), '--threads', str(args.threads),
                   '--backend', args.backend, '--device', args.device,
                   '--point-partition', point_partition,
                   '--early-stop-days', str(early_stop_days)]
        if block_scope:
            command.extend(('--train-block', *block_scope['train_block']))
        if args.config:
            command.extend(('--config', str(args.config)))
        if diagnostic:
            command.extend(('--diagnostic', diagnostic))
        if TERRAIN in components:
            command.extend(('--terrain-lookup', str(args.terrain_lookup)))
        command.extend(('--components', *components))
        event(args.out, phase, status='fit_started', run=str(directory.relative_to(args.out)),
              seed=seed, components=components, point_partition=point_partition,
              log=str(staging.relative_to(args.out) / 'train.log'))
        with (staging / 'train.log').open('w') as log, subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        ) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            if process.wait():
                raise RuntimeError(f'T1 training failed for {directory.name}. See {staging / "train.log"}.')
        # Only publish a directory that contains a verified receipt and complete forecasts.
        load_run(staging, keys, phase, seed, components, backend=args.backend, device=args.device,
                 block_scope=block_scope, point_partition=point_partition)
        staging.rename(directory)
        event(args.out, phase, status='fit_complete', run=str(directory.relative_to(args.out)), seed=seed,
              log=str(directory.relative_to(args.out) / 'train.log'))
    receipt, forecast, digest = load_run(directory, keys, phase, seed, components,
                                         backend=args.backend, device=args.device, block_scope=block_scope,
                                         point_partition=point_partition)
    if receipt['companion_cache_metadata_sha256'] != versions:
        raise ValueError(f'{directory}: companion cache files changed during or after this run.')
    if TERRAIN in components and receipt[TERRAIN]['lookup_sha256'] != sha256(args.terrain_lookup):
        raise ValueError(f'{directory}: upwind terrain lookup differs from the fit receipt.')
    return receipt, forecast, digest


def verified_a_freeze(out):
    path = out / A_FREEZE
    if not path.is_file():
        raise ValueError('Complete the A screen and freeze its selected components before phase B.')
    frozen = read_json(path)
    decisions = frozen['reports']
    expected_names = ['baseline.json', *(f'{i:02d}_{name}.json' for i, name in enumerate(COMPONENTS, 1))]
    if set(decisions) != set(expected_names):
        raise ValueError('The A freeze must reference the baseline and the four registered comparisons.')
    active = []
    for name, expected_hash in decisions.items():
        if sha256(out / 'a' / name) != expected_hash:
            raise ValueError(f'A report changed after freezing: {name}.')
    for component, name in zip(COMPONENTS, expected_names[1:]):
        report = read_json(out / 'a' / name)
        if report['component'] != component or report['candidate_components'] != [*active, component]:
            raise ValueError(f'Registered A comparison order disagrees with {name}.')
        if report['delta'][PRIMARY_SCORE] >= MINIMUM_GAIN:
            active.append(component)
    if active != frozen['candidate_components'] or frozen['baseline_components'] != []:
        raise ValueError('Frozen B configuration disagrees with the A decisions.')
    return frozen


def dates_and_keys(args, phase):
    train_days, days = date_roles(args.split, phase)
    roster = load_roster(args.data_root)
    return train_days, days, roster, forecast_keys(days, roster)


def phase_versions(args, train_days, predict_days):
    return {name: cache_version(cache_paths(args.workdir, days))
            for name, days in (('training', train_days), ('prediction', predict_days))}


def report_gains(component, result, retained):
    delta = result['delta'][PRIMARY_SCORE]
    blocks = ', '.join(f'{month}: {value["delta"][PRIMARY_SCORE]:+.6f}'
                       for month, value in result['blocks'].items())
    print(f'[A] {component}: {delta:+.6f}; {blocks}; retained={retained}', flush=True)


def screen_a(args):
    settings = setup(args)
    if (args.out / A_FREEZE).exists():
        frozen = verified_a_freeze(args.out)
        if frozen['setup'] != settings:
            raise ValueError('A is already frozen with different settings. Use a fresh run directory.')
        print(json.dumps(frozen, indent=2))
        return
    settings_path = args.out / 'setup.json'
    if settings_path.exists() and read_json(settings_path) != settings:
        raise ValueError('Experiment setup changed between attempts. Use a fresh run directory.')
    write_json(settings_path, settings)

    train_days, days, roster, keys = dates_and_keys(args, 'A')
    versions = phase_versions(args, train_days, days)
    directory = args.out / 'a'
    print(f'[A] Fit and verify the baseline feature configuration using {args.backend}/{args.device}', flush=True)
    baseline_receipt, baseline, baseline_digest = fit_or_reuse(
        args, directory / 'baseline', keys, 'A', A_SEED, [], versions,
    )
    truth = labels(args.data_root, days, roster)
    score = score_track('T1', truth, baseline, roster)
    if (len(baseline) != 48384 or score['scored_stations'] != 12 or score['scored_station_days'] != 399):
        raise ValueError(f'A baseline keys or scoring population differs from the registered cohort: {score}.')
    baseline_report = directory / 'baseline.json'
    write_json(baseline_report, {'scores': score, 'historical_lightgbm_reference': REFERENCE_A,
                                'training_receipt_sha256': baseline_digest, 'training': baseline_receipt})
    reports = {baseline_report.name: sha256(baseline_report)}
    print(f'[A] Refitted baseline {score[PRIMARY_SCORE]:.6f}, 12 stations, 399 scored station-days', flush=True)
    active, control, control_receipt = [], baseline, baseline_receipt
    for index, component in enumerate(COMPONENTS, 1):
        print(f'[A {index}/4] Test {component} on top of {active}', flush=True)
        candidate_components = [*active, component]
        experiment = directory / f'{index:02d}_{component}'
        receipt, candidate, digest = fit_or_reuse(
            args, experiment, keys, 'A', A_SEED, candidate_components, versions, diagnostic=component,
        )
        assert_matched(control_receipt, receipt)
        result = compare_pair(truth, control, candidate, roster, development=True)
        retained = result['delta'][PRIMARY_SCORE] >= MINIMUM_GAIN
        values = None
        if component != 'station_day_weight':
            diagnostic = receipt['diagnostic']
            path = experiment / 'diagnostic.npy'
            if (diagnostic['component'] != component or diagnostic['file'] != path.name or
                    sha256(path) != diagnostic['sha256']):
                raise ValueError(f'{experiment}: diagnostic feature values differ from the receipt.')
            # Diagnostic arrays were saved in the same order as the original CSV.
            original_keys = pd.read_csv(experiment / receipt['prediction_file'], usecols=POINT_KEYS,
                                        dtype={'station_id': str, 'date': str})
            values = (original_keys.assign(value=np.load(path, allow_pickle=False)).set_index(POINT_KEYS)['value']
                      .reindex(pd.MultiIndex.from_frame(candidate[POINT_KEYS])).to_numpy())
        result.update({'component': component, 'control_components': active, 'candidate_components': candidate_components,
                       'retained': retained, 'seed': A_SEED, 'candidate_training_receipt_sha256': digest,
                       'control_training': control_receipt, 'candidate_training': receipt,
                       'diagnostic_rmse_groups': diagnostic_errors(truth, control, candidate, component, values)})
        report_path = experiment.with_suffix('.json')
        write_json(report_path, result)
        reports[report_path.name] = sha256(report_path)
        report_gains(component, result, retained)
        event(args.out, 'A', status='scored', component=component, delta=result['delta'][PRIMARY_SCORE], retained=retained)
        if retained:
            active, control, control_receipt = candidate_components, candidate, receipt

    frozen = {'setup': settings, 'reports': reports, 'baseline_components': [], 'candidate_components': active,
              'recipe': {key: baseline_receipt[key] for key in RECIPE_FIELDS}}
    write_json(args.out / A_FREEZE, frozen)
    print(f'[A] Frozen B candidate components: {active}', flush=True)


def b_runs(out, active):
    for seed in B_SEEDS:
        for arm, components in (('baseline', []), ('candidate', active)):
            directory = out / 'b' / str(seed) / arm
            yield seed, arm, components, directory


def predict_b(args):
    frozen = verified_a_freeze(args.out)
    if frozen['setup'] != setup(args):
        raise ValueError('B settings must match the frozen A experiment.')
    if (args.out / B_FREEZE).exists():
        print(f'[B predict] Already frozen at {args.out / B_FREEZE}', flush=True)
        return
    train_days, days, _, keys = dates_and_keys(args, 'B')
    versions = phase_versions(args, train_days, days)
    records = {}
    control = None
    for index, (seed, arm, components, directory) in enumerate(b_runs(args.out, frozen['candidate_components']), 1):
        print(f'[B predict {index}/6] {arm}, seed {seed}, components {components}', flush=True)
        receipt, _, digest = fit_or_reuse(args, directory, keys, 'B', seed, components, versions)
        assert_matched(frozen['recipe'], receipt, RECIPE_FIELDS)
        if control is None:
            control = receipt
        assert_matched(control, receipt)
        records[str(directory.relative_to(args.out))] = {
            'training_receipt_sha256': digest, 'prediction_sha256': receipt['prediction_sha256'],
        }
    write_json(args.out / B_FREEZE, {'a_freeze_sha256': sha256(args.out / A_FREEZE), 'forecasts': records})
    event(args.out, 'B-predict', status='frozen', predictions=len(records))
    print(f'[B predict] Six full forecasts frozen in {args.out / B_FREEZE}', flush=True)


def identical_seed_groups(forecasts):
    groups = []
    for seed, values in forecasts.items():
        for group in groups:
            if np.array_equal(forecasts[group[0]], values):
                group.append(seed)
                break
        else:
            groups.append([seed])
    return groups


def score_b(args):
    frozen = verified_a_freeze(args.out)
    manifest_path = args.out / B_FREEZE
    if not manifest_path.is_file():
        raise ValueError('Run B prediction successfully before opening B labels.')
    manifest = read_json(manifest_path)
    if (manifest['a_freeze_sha256'] != sha256(args.out / A_FREEZE) or
            frozen['setup']['data_root'] != str(args.data_root.resolve()) or
            frozen['setup']['split_sha256'] != sha256(args.split) or
            frozen['setup']['code_sha256'] != code_version()):
        raise ValueError('Scoring inputs differ from the frozen B setup.')
    _, days, roster, keys = dates_and_keys(args, 'B')
    expected_runs = list(b_runs(args.out, frozen['candidate_components']))
    if set(manifest['forecasts']) != {str(run[3].relative_to(args.out)) for run in expected_runs}:
        raise ValueError('B prediction freeze must contain exactly the six registered forecasts.')
    receipts, forecasts = {}, {'baseline': {}, 'candidate': {}}
    for seed, arm, components, directory in expected_runs:
        receipt, forecast, digest = load_run(directory, keys, 'B', seed, components,
                                             backend=frozen['recipe']['backend'], device=frozen['recipe']['device'])
        record = manifest['forecasts'][str(directory.relative_to(args.out))]
        if record != {'training_receipt_sha256': digest, 'prediction_sha256': receipt['prediction_sha256']}:
            raise ValueError(f'{directory}: training receipt or prediction changed after the B freeze.')
        assert_matched(frozen['recipe'], receipt, RECIPE_FIELDS)
        if receipts:
            assert_matched(next(iter(receipts.values())), receipt)
        receipts[f'{seed}/{arm}'] = receipt
        forecasts[arm][seed] = forecast['prediction'].to_numpy()
    report_path = args.out / 'b_score.json'
    if report_path.exists():
        prior = read_json(report_path)
        if prior['b_prediction_freeze_sha256'] != sha256(manifest_path):
            raise ValueError('Existing B report refers to a different prediction freeze.')
        print(f'[B score] Already scored at {report_path}. Accepted={prior["acceptance"]["accepted"]}', flush=True)
        return

    event(args.out, 'B-score', status='opening_frozen_labels', predictions=6, days=len(days))
    truth = labels(args.data_root, days, roster)
    pairs = {str(seed): compare_pair(truth, keys.assign(prediction=forecasts['baseline'][seed]),
                                    keys.assign(prediction=forecasts['candidate'][seed]), roster)
             for seed in B_SEEDS}
    ensembles = {arm: keys.assign(prediction=np.clip(np.mean(list(values.values()), axis=0), 0, MAX_PREDICTION['T1']))
                 for arm, values in forecasts.items()}
    ensemble_scores = compare_pair(truth, ensembles['baseline'], ensembles['candidate'], roster)
    deltas = [pairs[str(seed)]['delta'][PRIMARY_SCORE] for seed in B_SEEDS]
    statistics = b_statistics(deltas, ensemble_scores['delta'][PRIMARY_SCORE])
    duplicates = {arm: identical_seed_groups(values) for arm, values in forecasts.items()}
    decision = frozen['candidate_components'] if statistics['accepted'] else frozen['baseline_components']
    report = {'b_prediction_freeze_sha256': sha256(manifest_path), 'seeds': list(B_SEEDS),
              'backend': frozen['recipe']['backend'], 'device': frozen['recipe']['device'],
              'candidate_components': frozen['candidate_components'], 'adopted_components': decision,
              'per_seed': pairs, 'paired_seed_deltas': deltas, 'three_seed_ensemble': ensemble_scores,
              'acceptance': statistics, 'identical_forecast_seed_groups': duplicates,
              'repeated_seed_result': any(len(group) > 1 for groups in duplicates.values() for group in groups),
              'training': receipts}
    write_json(report_path, report)
    event(args.out, 'B-score', status='scored', accepted=statistics['accepted'], adopted=decision)
    print(json.dumps({'paired_seed_deltas': deltas, 'acceptance': statistics, 'adopted_components': decision,
                      'identical_forecast_seed_groups': duplicates, 'report': str(report_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('a', 'b-predict', 'b-score'))
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--local-wind-cache', type=Path, default=Path('/workspace/qhl/runs/companion_t1_local_wind'))
    parser.add_argument('--threads', type=int)
    parser.add_argument('--backend', choices=('lightgbm', 'xgboost'), default='lightgbm')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.threads is None:
        args.threads = GPU_THREADS if args.device == 'cuda' else DEFAULT_THREADS
    if args.out is None:
        suffix = 'v2' if args.backend == 'lightgbm' else f'{args.backend}_{args.device}'
        args.out = Path(f'/workspace/qhl/runs/companion_t1_incremental_{suffix}')
    if args.threads < 1:
        parser.error('--threads must be positive.')
    if args.device == 'cuda' and args.threads > GPU_THREADS:
        parser.error(f'Use at most {GPU_THREADS} CPU threads with CUDA.')
    args.out.mkdir(parents=True, exist_ok=True)
    stages = {'a': screen_a, 'b-predict': predict_b, 'b-score': score_b}
    with (args.out / '.stage.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('An experiment command is already using this run directory.')
        event(args.out, args.stage, status='started', backend=args.backend, device=args.device)
        try:
            stages[args.stage](args)
        except (Exception, KeyboardInterrupt) as error:
            event(args.out, args.stage, status='failed', detail=str(error))
            raise


if __name__ == '__main__':
    main()
