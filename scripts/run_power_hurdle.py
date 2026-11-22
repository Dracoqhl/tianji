#!/usr/bin/env python3
"""Compare two-part GPU power trees on the frozen T1 and T3 baselines."""
import argparse
import fcntl
from functools import partial
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic
import traceback
from types import SimpleNamespace

import numpy as np

from advance_batch_b import read_frozen_forecast
from batch_energy_short import fit_short_forecasts, roster_for, short_source, t3_windows
from confirm_working_baselines import average_seeds
from experiment_companion_t1 import assert_matched, read_json
from run_companion_warm_t1 import B_SEEDS, EARLY_STOP_DAYS, SPLIT_PATH, chronological_dates, forecast_keys, run, sha256, write_json
from run_forecast_blends import blend, freeze_json
from run_fusion_mechanisms import finish_pending, stage, t3_shape_blend, T3_SHAPE
from run_intraday_residuals import t1_v4_reference
from run_phase_anchor import T1_ARMS as PHASE_ARMS, phase_forecasts
from run_pressure_profile import t3_fit_contract
from run_t1_issue_wind_error import dynamic_baseline
from run_t3_nearby_donors import SHAPE, v3_reference
from run_wind_attribution import STATION, read_arm
from same_issue_weather import archive_stamp
from score_forecasts import DAY_KEYS, POINT_KEYS, MAX_PREDICTION, MIN_VALID_POINTS
from screen_companion_t1_stations import show_progress
from t1_dynamic_wind import NAMES as DYNAMIC_COLUMNS, append_dynamic
from tree_model import best_iteration, fitted_device, predict_tree, save_tree_model, train_tree, xgboost_parameters
from warm_shared_power import labels, load_roster


RUNS = Path('/workspace/qhl/runs')
ARMS = {'hurdle_half': .5, 'hurdle_full': 1.}


def fit_hurdle(inputs, target, inner, forecast, names, params, device, folder):
    if (params.get('objective', 'huber') != 'huber' or not np.isfinite(target).all()
            or (target < 0).any() or not inner.any() or inner.all()):
        raise ValueError('Two-part power needs nonnegative power and a valid inner split.')
    positive = target > 0
    if np.unique(positive[~inner]).size != 2 or not (positive & inner).any():
        raise ValueError('Insufficient zero/positive points for chronological two-part fitting.')
    options = {'categorical_features': ['station_idx'], 'backend': 'xgboost', 'device': device}
    audit = {}

    def head(name, output, eligible, objective):
        fit, stop = eligible & ~inner, eligible & inner
        config = {**params, 'objective': objective}
        print(f'[Power {name}] {int(fit.sum())} fit, {int(stop.sum())} early-stop points', flush=True)
        selected = train_tree(inputs[fit], output[fit], names, config, X_valid=inputs[stop], y_valid=output[stop],
                              early_stopping_rounds=int(params.get('early_stopping_rounds', 100)), **options)
        trees = best_iteration(selected, backend='xgboost', fallback=params['n_estimators'])
        del selected
        picked = slice(None) if eligible.all() else eligible
        final = train_tree(inputs[picked], output[picked], names, {**config, 'n_estimators': trees}, **options)
        path = folder / f'{name}.joblib'
        save_tree_model(final, path)
        values = predict_tree(final, forecast, backend='xgboost').astype(float)
        audit[name] = {'trees': trees, 'selection_rows': int(fit.sum()), 'stopping_rows': int(stop.sum()),
                       'refit_rows': int(eligible.sum()), 'model_file': path.name, 'model_sha256': sha256(path),
                       'effective_device': fitted_device(final.booster), 'parameters': xgboost_parameters(config, device)}
        return values

    probability = head('occurrence', positive.astype(np.float32), np.ones(len(target), dtype=bool), 'binary')
    strength = head('positive_power', target, positive, 'huber')
    if (not np.isfinite(probability).all() or not np.isfinite(strength).all()
            or (probability < 0).any() or (probability > 1).any()):
        raise ValueError('Invalid two-part forecast.')
    return probability * np.maximum(strength, 0), {
        'trees': audit['positive_power']['trees'], 'fit_parameters': {name: row['parameters'] for name, row in audit.items()},
        'heads': audit, 'train_zero_points': int((~positive).sum()), 'early_stop_zero_points': int((inner & ~positive).sum()),
        'mean_forecast_p_positive': float(probability.mean())}


def t1_tree(args, phase, train, days, seed, keys):
    root = args.dynamic_run / 'T1' / phase / str(seed) / 'dynamic_gradient'
    old_path = root / 'training_receipt.json'
    old = read_json(old_path)
    direct = read_frozen_forecast(root / old['prediction_file'], old['prediction_sha256'], keys, 'T1')
    target = args.out / 'T1' / phase / str(seed) / 'hurdle'
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = target / 'training_receipt.json'
    if not marker.is_file():
        folder = Path(mkdtemp(dir=target.parent, prefix='.hurdle-attempt-'))
        options = SimpleNamespace(**vars(args))
        options.out, options.phase, options.seed = folder, phase, seed
        options.backend, options.components, options.point_partition = 'xgboost', [], 'full'
        options.train_block, options.smoke, options.check_only, options.diagnostic = None, False, False, None
        options.early_stop_days = EARLY_STOP_DAYS
        dynamic = partial(append_dynamic, args.data_root, args.dynamic_run / 'native_weather')
        run(options, extra_features=dynamic, date_partition=(train, days), fit_expert=fit_hurdle)
        folder.rename(target)
    receipt = read_json(marker)
    assert_matched(old, receipt, ('seed', 'data_root', 'chronological_dates', 'companion_cache_metadata_sha256',
                                  'device', 'threads', 'backend_version', 'early_stopping_valid_rows',
                                  'early_stopping_start_date', 'feature_names', 'config_sha256',
                                  'valid_training_sample_sha256', 'companion_source_sha256'))
    if not set(DYNAMIC_COLUMNS).issubset(receipt['feature_names']):
        raise ValueError('T1 two-part tree must retain the accepted dynamic gradient inputs.')
    candidate = read_frozen_forecast(target / receipt['prediction_file'], receipt['prediction_sha256'], keys, 'T1')
    return direct, candidate, {'direct': sha256(old_path), 'two_part': sha256(marker)}


def t1_predict(args, phase, train, days, sites, seed, names):
    keys = forecast_keys(days, sites)
    base, reference = t1_v4_reference(args, phase, keys, seed)
    v3, _ = dynamic_baseline(args, phase, keys, seed)
    direct, branch, source = t1_tree(args, phase, train, days, seed, keys)
    fit = args.phase_run / 'T1/phase_fit.json'
    sites_fit = read_json(fit)['sites']
    delta = (.8 * (branch.prediction - direct.prediction)).where(keys.station_id.ne(STATION), 0.)
    arms = {'baseline': base}
    for name in names:
        changed = v3.assign(prediction=(v3.prediction + ARMS[name] * delta).clip(0, MAX_PREDICTION['T1']))
        arms[name] = phase_forecasts(changed, sites_fit, (PHASE_ARMS[1],))[PHASE_ARMS[1]]
    return keys, arms, {'v4': reference, 'trees': source, 'smoothing': sha256(fit)}


def t3_study(args):
    features = None

    def predict(window, train, days, sites, seed, names):
        nonlocal features
        keys, direct, ridge, base, reference = v3_reference(args, window, train, days, sites, seed)
        if features is None:
            features = short_source(args, [*load_roster(args.data_root), *sites])
        roster, model_source = features
        *schema, params = model_source
        old, origin = t3_fit_contract(args, window, seed)
        config = sha256(args.config or args.companion_code / 'configs/default.json')
        assert_matched(read_json(args.short_tree / 'results.json')['settings'],
                       {'data_root': str(args.data_root.resolve()), 'split_sha256': sha256(args.split),
                        'device': args.device, 'threads': args.threads, 'companion_config_sha256': config},
                       ('data_root', 'split_sha256', 'device', 'threads', 'companion_config_sha256'))
        signature = {'window': window, 'fit': list(map(str, train)), 'forecast': list(map(str, days)), 'seed': seed,
                     'reference': reference, 'old_tree': origin, 'config': config,
                     'weather': archive_stamp(args.workdir, train + days), 'feature_names': schema[2]}
        folder = args.out / 'T3' / window / str(seed)
        folder.mkdir(parents=True, exist_ok=True)
        saved = folder / 'hurdle'
        marker = saved / 'fit.json'
        if not marker.is_file():
            pending = Path(mkdtemp(dir=folder, prefix='.hurdle-attempt-'))
            _, _, fitted = fit_short_forecasts(args, pending, train, days, roster, sites,
                                                (*schema, dict(params, random_state=seed)),
                                                modes=('short_only',), fit_expert=fit_hurdle)
            assert_matched(old, fitted, ('train_days', 'score_days', 'inner_stop_start', 'sample_keys', 'cache'))
            assert_matched(old['fit']['short_only'], fitted['fit']['short_only'],
                           ('fit_points', 'target_points', 'early_stop_points'))
            freeze_json(pending / 'fit.json', {'signature': signature, 'result': fitted})
            pending.rename(saved)
        record = read_json(marker)
        if record['signature'] != signature:
            raise ValueError('Existing two-part T3 model uses a different fit contract.')
        two_part = read_frozen_forecast(saved / 'short_only.csv', record['result']['predictions_sha256']['short_only'],
                                        keys, 'T3')
        arms = {'baseline': base}
        for name in names:
            changed = blend(direct, two_part, ARMS[name], 'T3')
            arms[name] = t3_shape_blend(changed, ridge, T3_SHAPE[SHAPE])
        return keys, arms, {'v3': reference, 'tree': sha256(marker)}

    return predict, tuple(ARMS)


def group_errors(truth, frames):
    valid = truth.loc[truth.power_limit.eq(1) & np.isfinite(truth.power), POINT_KEYS + ['power']]
    valid = valid.loc[valid.groupby(DAY_KEYS).power.transform('size').ge(MIN_VALID_POINTS)]
    errors = {}
    for name, frame in frames.items():
        rows = valid.merge(frame, on=POINT_KEYS, validate='one_to_one')
        populations = {'zero': rows.power.eq(0), 'positive': rows.power.gt(0), 'above_12_mw': rows.power.ge(12)}
        square = (rows.prediction - rows.power)**2
        errors[name] = {group: {'points': int(use.sum()), 'rmse_mw': float(np.sqrt(square[use].mean()))
                                if use.any() else None} for group, use in populations.items()}
    return errors


def diagnostics(args, track, score):
    root = args.out / track
    sites = load_roster(args.data_root) if track == 'T1' else roster_for(args.data_root, 'T3')
    if track == 'T1':
        windows = [(phase, *chronological_dates(args.split, phase)) for phase in ('A', 'B')]
    else:
        windows = list(t3_windows())
    values = {}
    for window, _, days in windows[:-1]:
        keys = forecast_keys(days, sites)
        folder = root / window / 'forecasts'
        lock = sha256(folder / 'predictions_frozen.json')
        if score['gate']['screen_locks'][window] != lock:
            raise ValueError('Two-part diagnostic A lock mismatch.')
        frames = {name: read_arm(folder, name, keys, track) for name in ('baseline', *ARMS)}
        values[window] = group_errors(labels(args.data_root, days, sites), frames)
    if score.get('B_status') == 'scored':
        _, _, days = windows[-1]
        keys = forecast_keys(days, sites)
        folder = root / 'B'
        ledger = folder / 'predictions_frozen.json'
        if read_json(folder / 'b_score.json')['prediction_lock_sha256'] != sha256(ledger):
            raise ValueError('Two-part diagnostic B ledger changed.')
        locks = read_json(ledger)
        pairs = {}
        for seed in B_SEEDS:
            path = folder / str(seed)
            if locks[str(seed)] != sha256(path / 'predictions_frozen.json'):
                raise ValueError('Two-part diagnostic B prediction lock changed.')
            pairs[seed] = {arm: read_arm(path, arm, keys, track) for arm in ('baseline', 'candidate')}
        values['B'] = group_errors(labels(args.data_root, days, sites), average_seeds(keys, pairs, track))
    freeze_json(root / 'power_groups.json', values)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out': RUNS / 'power_hurdle_gpu_v1', 'data-root': Path('/workspace/qhl/data'), 'split': SPLIT_PATH,
                'workdir': Path('/workspace/cpy/code/workdir'), 'companion-code': Path('/workspace/cpy/code'),
                'phase-run': RUNS / 'phase_anchor_cpu_v1', 'dynamic-run': RUNS / 't1_dynamic_gradient_gpu_v1',
                'ridge-run': RUNS / 'daily_bias_ridge_gpu_v1', 'fusion-run': RUNS / 'fusion_mechanisms_cpu_v1',
                'short-tree': RUNS / 'three_track_gpu_batch_v1/T3', 'transfer-run': RUNS / 'same_issue_followup_v1/T3',
                't3-confirm': RUNS / 't3_transfer_confirmation_gpu_v1'}
    for name, value in defaults.items():
        parser.add_argument(f'--{name}', type=Path, default=value)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_json(args.out / 'plan.json', {'args': {name: str(value) for name, value in vars(args).items()},
                    'split': sha256(args.split), 'arms': ARMS, 'seeds': list(B_SEEDS),
                    'code': {name: sha256(Path(__file__).with_name(name)) for name in
                             ('run_power_hurdle.py', 'run_companion_warm_t1.py', 'batch_energy_short.py',
                              'tree_model.py', 'run_fusion_mechanisms.py', 'run_intraday_residuals.py')}})
        start, result, pending = monotonic(), {}, {}
        studies = {'T1': (partial(t1_predict, args), tuple(ARMS)), 'T3': t3_study(args)}
        for i, (track, study) in enumerate(studies.items()):
            show_progress(args.out, '两段式功率树', i, len(studies), track)
            try:
                result[track] = {'status': 'complete', **stage(args, track, pending, study=study)}
            except Exception as error:
                traceback.print_exc()
                result[track] = {'status': 'failed', 'error': str(error)}
            write_json(args.out / 'summary.json', result)
        finish_pending(args, result, pending)
        for track, row in result.items():
            if row['status'] == 'complete':
                try:
                    row['power_groups'] = diagnostics(args, track, row)
                except Exception as error:
                    traceback.print_exc()
                    row['diagnostic_error'] = str(error)
        result['seconds'] = round(monotonic() - start, 2)
        write_json(args.out / 'summary.json', result)
        show_progress(args.out, '两段式功率树', len(studies), len(studies), '结果已保存')
        print(f'结果摘要 {args.out / "summary.json"}', flush=True)
        return int(any(result[track]['status'] != 'complete' or 'diagnostic_error' in result[track] for track in studies))


if __name__ == '__main__':
    raise SystemExit(main())
