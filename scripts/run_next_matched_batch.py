#!/usr/bin/env python3
"""Run matched improvements on the current chronological and short-history baselines."""

import argparse
from functools import partial
import fcntl
import hashlib
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic
import traceback

import numpy as np

from advance_batch_b import keys_for, read_frozen_forecast
from batch_energy_short import fit_short_forecasts, load_daily_targets, roster_for, weather_day_targets
from compare_t2_mlp import mix
from compare_warm_t1 import PRIMARY_SCORE, b_statistics
from confirm_working_baselines import average_seeds, keep_sites
from experiment_companion_t1 import assert_matched, identical_seed_groups, read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, chronological_dates, forecast_keys, sha256, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from run_pressure_profile import summarize_b, t3_fit_contract, t3_stage
from run_steamlink_followup import confirmed_t3, native_forecast, old_t1, ramp_timing, t2_scores, wind_alternatives
from run_t1_issue_wind_error import dynamic_baseline
from run_t1_t3_batch import paired_report
from run_wind_attribution import STATION, read_arm
from same_issue_weather import archive_stamp
from score_forecasts import DAY_KEYS, MAX_PREDICTION
from screen_companion_t1_stations import block_comparisons, show_progress
from t1_dynamic_wind import NAMES as DYNAMIC_COLUMNS, append_dynamic
from warm_shared_power import labels, load_roster
from warm_weather_regime import NAMES as WARM_COLUMNS, append_warm_weather


RUNS = Path('/workspace/qhl/runs')
ARMS = ('warm_weather', 'warm_weight')
ENERGY_THRESHOLD = 60.0


def warm_training_weight(days, roster, positions, original):
    positions = np.asarray(positions)
    if (positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer)
            or not len(positions) or positions.min() < 0 or positions.max() >= len(days)*len(roster)*96):
        raise ValueError('Invalid T1 seasonal fit row keys.')
    day, site = np.divmod(positions // 96, len(roster))
    base = np.ones(len(positions)) if original is None else np.asarray(original, dtype=float)
    if base.shape != positions.shape or not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError('Invalid T1 original fit weights.')
    is_warm = np.array([5 <= d.month <= 8 for d in days])[day]
    weights = base * np.where(is_warm, 1., .5)
    pre = np.bincount(site, weights=base, minlength=len(roster))
    post = np.bincount(site, weights=weights, minlength=len(roster))
    weights *= np.divide(pre, post, out=np.ones_like(pre), where=post > 0)[site]
    if not np.allclose(np.bincount(site, weights=weights, minlength=len(roster)), pre):
        raise ValueError('Warm training weights changed station mass.')
    return weights, {'fit_rows': len(weights), 'warm_rows': int(is_warm.sum()),
                     'first_fit': str(days[0]), 'last_fit': str(days[-1]),
                     'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest()}


def t1_predictions(args, phase, seed, keys, wind, arms):
    baseline, digest = dynamic_baseline(args, phase, keys, seed)
    routed, route_lock = old_t1(args, phase, keys, seed)
    wind = keys.assign(prediction=wind.ravel())
    old = args.dynamic_run / 'T1' / phase / str(seed) / 'dynamic_gradient'
    source = old / 'training_receipt.json'
    original = read_json(source)
    original_tree = read_frozen_forecast(old / original['prediction_file'], original['prediction_sha256'], keys, 'T1')
    replay = blend(keep_sites(original_tree, routed, [STATION]), wind, .2, 'T1')
    if not np.allclose(replay.prediction, baseline.prediction, atol=1e-6, rtol=0):
        raise ValueError('Rebuilt T1 v3 baseline differs from its lock.')
    result = {'baseline': baseline}
    receipts = {'v3_forecasts': digest, 'v3_tree': sha256(source), 'station_route': route_lock}
    for arm in arms:
        print(f'[T1 {phase}] {arm} seed={seed}', flush=True)

        def features(root, _cache, X, names, dates, roster, positions):
            X, names = append_dynamic(root, args.dynamic_run / 'native_weather', X, names, dates, roster, positions)
            if arm == 'warm_weather':
                return append_warm_weather(args.workdir, X, names, dates, roster, positions)
            return X, names

        added = WARM_COLUMNS if arm == 'warm_weather' else []
        weighting = warm_training_weight if arm == 'warm_weight' else None
        tree, receipts[arm] = native_forecast(args, phase, seed, keys, component=arm, feature_append=features,
                                           columns=DYNAMIC_COLUMNS + added, training_weight=weighting)
        fitted = read_json(args.out / 'T1' / phase / str(seed) / arm / 'training_receipt.json')
        assert_matched(original, fitted, ('seed', 'data_root', 'feature_workdir', 'chronological_dates',
                       'valid_training_sample_sha256', 'config_sha256', 'companion_cache_metadata_sha256',
                       'early_stopping_days_inside_train', 'early_stopping_valid_rows', 'device', 'threads'))
        if fitted['feature_names'] != original['feature_names'] + added:
            raise ValueError('Unexpected T1 candidate columns.')
        if arm == 'warm_weight' and not fitted['experts'][0].get('training_weight'):
            raise ValueError('Missing T1 seasonal weight receipt.')
        result[arm] = blend(keep_sites(tree, routed, [STATION]), wind, .2, 'T1')
    return result, receipts


def amplitude_and_shift(truth, frames, keys):
    valid = truth.power_limit.eq(1).to_numpy().reshape(-1, 96)
    actual = truth.power.to_numpy(dtype=float).reshape(-1, 96)
    valid = valid & np.isfinite(actual)
    full_count = valid.sum(axis=1)
    common = valid[:, 4:-4]
    take = (full_count >= 72) & (common.sum(axis=1) > 0)
    count = common[take].sum(axis=1)
    day, target, result = common[take], actual[take, 4:-4], {}
    for name, forecast in frames.items():
        ordered = forecast.set_index(list(keys)).reindex(keys.set_index(list(keys)).index)
        curve = ordered.prediction.to_numpy().reshape(-1, 96)[take]
        if not np.isfinite(curve).all() or not take.any():
            raise ValueError('Incomplete T1 forecasts or empty diagnostic cohort.')
        errors = np.stack([np.where(day, curve[:, 4+shift:92+shift] - target, 0.)
                           for shift in range(-4, 5)])
        mse = np.square(errors).sum(axis=2) / count
        bias = errors.sum(axis=2) / count
        shape = np.maximum(0, mse - np.square(bias))
        result[name] = {'common_station_days': int(take.sum()),
                        'rmse_mw': float(np.sqrt(mse[4]).mean()),
                        'oracle_no_bias_rmse_mw': float(np.sqrt(shape[4]).mean()),
                        'oracle_shift_rmse_mw': float(np.sqrt(mse.min(axis=0)).mean()),
                        'oracle_shift_no_bias_rmse_mw': float(np.sqrt(shape.min(axis=0)).mean())}
    return result


def t1_stage(args):
    root = args.out / 'T1'
    roster = load_roster(args.data_root)
    train, days = chronological_dates(args.split, 'A')
    keys = forecast_keys(days, roster)
    wind, _, thresholds, weather = wind_alternatives(args, train, days, roster, selected=ARMS[0])
    arms, sources = t1_predictions(args, 'A', A_SEED, keys, wind, ARMS)
    arms, lock = freeze_window(root / 'A' / 'forecasts', 'T1', keys, arms, {'fits': sources, 'weather': weather})
    actual = labels(args.data_root, days, roster)
    scores = block_comparisons(actual, arms, roster, ARMS)
    primary = {name: row[PRIMARY_SCORE] for name, row in scores['scores'].items()}
    winner = max(ARMS, key=primary.get)
    delta = primary[winner] - primary['baseline']
    gate = {'candidate': winner, 'delta': delta, 'promoted': delta > 1e-8, 'prediction_lock': lock}
    diagnostics = {'shift_oracles': amplitude_and_shift(actual, arms, keys),
                   'ramp_timing': ramp_timing(actual, arms, thresholds[1])}
    freeze_json(root / 'A' / 'score.json', {'scores': scores, 'gate': gate, **diagnostics})
    result = {'A': primary, 'gate': gate, **diagnostics}
    print(f'[T1 A] {primary}; {gate}', flush=True)
    if gate['promoted']:
        train_b, days_b = chronological_dates(args.split, 'B')
        if train_b != train + days:
            raise ValueError('T1 B fit days must include all of A.')
        keys = forecast_keys(days_b, roster)
        wind, _, _, weather = wind_alternatives(args, train_b, days_b, roster, selected=winner)
        pairs, locks = {}, {}
        for seed in B_SEEDS:
            arms, source = t1_predictions(args, 'B', seed, keys, wind, (winner,))
            pairs[seed], locks[str(seed)] = freeze_window(root / 'B' / str(seed) / 'pair', 'T1', keys,
                {'baseline': arms['baseline'], 'candidate': arms[winner]},
                {'gate': gate, 'source': source, 'weather': weather})
        sha = freeze_json(root / 'B' / 'predictions_frozen.json', locks)
        result['B'] = summarize_b(paired_report(root / 'B', 'T1', labels(args.data_root, days_b, roster),
                                                roster, keys, pairs, sha))
    return result


def t2_conditional_factor(conditional, actual):
    rows = conditional.merge(actual, on=DAY_KEYS, suffixes=('_pred', '_actual'), validate='one_to_one')
    low = rows.energy_pred.lt(ENERGY_THRESHOLD)
    ratio = (rows.loc[low, 'energy_pred'] / rows.loc[low, 'energy_actual']).to_numpy()
    return float(np.clip((ratio.sum() + 50) / (np.square(ratio).sum() + 50), .80, 1.20)), int(low.sum())


def t2_pair(args, phase, seed, roster, train, days):
    src = args.t2_prior / 'T2'
    gate_a = read_json(src / 'a_score.json')['gate']
    if gate_a['selected'] != 'station_calibrated' or not gate_a['promoted']:
        raise ValueError('Expected the frozen station-calibrated T2 recipe.')
    if phase == 'A':
        source = src / 'A' / 'forecasts'
        if gate_a['forecast_lock'] != sha256(source / 'predictions_frozen.json'):
            raise ValueError('T2 A was not locked by its source gate.')
        calibrated_name = 'station_calibrated'
    else:
        source = src / 'B' / str(seed) / 'forecasts'
        if read_json(src / 'B' / 'predictions_frozen.json')[str(seed)] != sha256(source / 'predictions_frozen.json'):
            raise ValueError('T2 B seed forecasts have a different lock.')
        calibrated_name = 'candidate'
    marker = source / 'predictions_frozen.json'
    manifest = read_json(marker)['sources']
    if phase == 'B' and manifest['gate'] != gate_a:
        raise ValueError('T2 B base recipe differs from A selection.')
    keys = keys_for(days, roster, 'T2')
    original = read_arm(source, 'baseline', keys, 'T2')
    calibrated = read_arm(source, calibrated_name, keys, 'T2')
    fold = src / phase / str(seed) / 'calibration'
    info = manifest['calibration']
    length = len(train) // 5
    fit_days, cal_days = train[:-length], train[-length:]
    plan = read_json(fold / 'plan.json')
    expected = {'train': list(map(str, fit_days)), 'forecast': list(map(str, cal_days)),
                'seed': seed, 'calibration': True}
    assert_matched(plan, expected, expected)
    mlp_receipt = read_json(fold / 'mlp' / 'receipt.json')
    tree_receipt = read_json(fold / 'relative_mse' / 'fit.json')
    if mlp_receipt != info['history']['MLP'] or tree_receipt != info['history']['relative']:
        raise ValueError('T2 calibration input has a different frozen receipt.')
    cal_keys = keys_for(cal_days, roster, 'T2')
    mlp = read_frozen_forecast(fold / 'mlp' / 'T2_development.csv', mlp_receipt['forecast_sha256'], cal_keys, 'T2')
    tree = read_frozen_forecast(fold / 'relative_mse' / 'forecast.csv', tree_receipt['prediction_sha256'], cal_keys, 'T2')
    cal = mix(mlp, tree)

    def apply_station(frame):
        return frame.assign(energy=(frame.energy * frame.station_id.map(info['factors']).fillna(1.))
                            .clip(0, MAX_PREDICTION['T2']))

    if not np.allclose(apply_station(original).energy, calibrated.energy, atol=1e-8, rtol=0):
        raise ValueError('T2 calibrated baseline could not be reconstructed.')
    _, actual = load_daily_targets(weather_day_targets(args.data_root, cal_days), roster)
    factor, count = t2_conditional_factor(apply_station(cal), actual)
    result = calibrated.assign(energy=(calibrated.energy * np.where(calibrated.energy < ENERGY_THRESHOLD, factor, 1.))
                               .clip(0, MAX_PREDICTION['T2']))
    metadata = {'baseline_lock': sha256(marker), 'fold_sha256': sha256(fold / 'plan.json'),
                'threshold_mwh': ENERGY_THRESHOLD, 'factor': factor, 'calibration_population': count}
    return keys, {'baseline': calibrated, 'candidate': result, 'raw_75_25': original}, metadata


def t2_stage(args):
    root = args.out / 'T2'
    roster = roster_for(args.data_root, 'T2')
    train, days = chronological_dates(args.split, 'A')
    keys, arms, state = t2_pair(args, 'A', A_SEED, roster, train, days)
    arms, lock = freeze_window(root / 'A' / 'forecasts', 'T2', keys, arms, state)
    scores, diagnosis = t2_scores(args, days, roster, arms)
    values = {name: row[PRIMARY_SCORE] for name, row in scores.items()}
    delta = values['candidate'] - values['baseline']
    gate = {'promoted': delta > 1e-8, 'delta': delta, 'prediction_lock': lock}
    result = {'A': values, 'gate': gate, 'groups_A': diagnosis, 'fit_A': state}
    freeze_json(root / 'A' / 'score.json', {**result, 'detailed_scores': scores})
    print(f'[T2 A] {values}; {gate}; low energy multiplier={state["factor"]:.4f}', flush=True)
    if gate['promoted']:
        train_b, days_b = chronological_dates(args.split, 'B')
        if train_b != train + days:
            raise ValueError('T2 B fit days must include all of A.')
        keys = keys_for(days_b, roster, 'T2')
        pairs, locks = {}, {}
        for seed in B_SEEDS:
            _, arms, state = t2_pair(args, 'B', seed, roster, train_b, days_b)
            pairs[seed], locks[str(seed)] = freeze_window(root / 'B' / str(seed) / 'pair', 'T2', keys,
                {arm: arms[arm] for arm in ('baseline', 'candidate')}, {'gate': gate, **state})
        lock_b = freeze_json(root / 'B' / 'predictions_frozen.json', locks)
        seed_scores = {str(seed): t2_scores(args, days_b, roster, pair)[0] for seed, pair in pairs.items()}
        ensemble, subgroups = t2_scores(args, days_b, roster, average_seeds(keys, pairs, 'T2'))

        def difference(score):
            return score['candidate'][PRIMARY_SCORE] - score['baseline'][PRIMARY_SCORE]

        deltas = {seed: difference(scores) for seed, scores in seed_scores.items()}
        decision = b_statistics(list(deltas.values()), difference(ensemble))
        identical = {arm: identical_seed_groups({seed: pair[arm].energy.to_numpy() for seed, pair in pairs.items()})
                     for arm in ('baseline', 'candidate')}
        result['B'] = {'scores': {name: row[PRIMARY_SCORE] for name, row in ensemble.items()},
                       'acceptance': decision, 'seed_deltas': deltas, 'groups_B': subgroups,
                       'identical_seed_groups': identical, 'prediction_lock': lock_b}
        freeze_json(root / 'B' / 'score.json', {**result['B'], 'pairs': seed_scores})
        print(f'[T2 B] {result["B"]["scores"]}; {decision}', flush=True)
    return result


def t3_pair(args, window, train, days, short, all_sites, model_source, seed):
    keys, tree, ridge, base, lock = confirmed_t3(args, window, train, days, short, seed)
    old, contract = t3_fit_contract(args, window, seed)
    config = args.config or args.companion_code / 'configs/default.json'
    settings = read_json(args.short_tree / 'results.json')['settings']
    expected = {'data_root': str(args.data_root.resolve()), 'split_sha256': sha256(args.split),
                'device': args.device, 'threads': args.threads, 'companion_config_sha256': sha256(config)}
    assert_matched(settings, expected, expected)
    *schema, params = model_source
    old_leaves = int(params.get('num_leaves', 31))
    leaves = max(2, old_leaves // 2)
    if leaves >= old_leaves:
        raise ValueError('T3 effective leaf limit is already minimal.')
    target = args.out / 'T3' / window / str(seed)
    target.mkdir(parents=True, exist_ok=True)
    fitted = target / 'tree'
    marker = fitted / 'fit.json'
    sig = {'seed': seed, 'leaf_limit': leaves, 'prior_leaf_limit': old_leaves, 'source_lock': lock,
           'fit_contract': contract, 'cache': archive_stamp(args.workdir, train + days)}
    if not marker.is_file():
        folder = Path(mkdtemp(dir=target, prefix='.fit-attempt-'))
        _, _, fit = fit_short_forecasts(args, folder, train, days, all_sites, short,
            (*schema, dict(params, random_state=seed, num_leaves=leaves)), modes=('short_only',))
        assert_matched(old, fit, ('train_days', 'score_days', 'inner_stop_start', 'cache', 'sample_keys'))
        assert_matched(old['fit']['short_only'], fit['fit']['short_only'],
                       ('fit_points', 'target_points', 'early_stop_points'))
        freeze_json(folder / 'fit.json', {'fit': fit, 'signature': sig})
        folder.rename(fitted)
    note = read_json(marker)
    if sig != note['signature']:
        raise ValueError('T3 candidate cache has a different training signature.')
    improved = read_frozen_forecast(fitted / 'short_only.csv', note['fit']['predictions_sha256']['short_only'], keys, 'T3')
    return (keys, *freeze_window(target / 'forecasts', 'T3', keys,
        {'baseline': base, 'candidate': blend(improved, ridge, .25, 'T3'), 'baseline_tree': tree,
         'limited_tree': improved}, {'baseline_lock': lock, 'fit_receipt': sha256(marker)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out': RUNS / 'next_matched_gpu_v1', 'data-root': Path('/workspace/qhl/data'),
        'workdir': Path('/workspace/cpy/code/workdir'), 'companion-code': Path('/workspace/cpy/code'),
        'dynamic-run': RUNS / 't1_dynamic_gradient_gpu_v1', 'power-run': RUNS / 't1_t3_fast_gpu_v1',
        't2-prior': RUNS / 'steamlink_followup_gpu_v1', 'short-tree': RUNS / 'three_track_gpu_batch_v1/T3',
        'transfer-run': RUNS / 'same_issue_followup_v1/T3', 't3-confirm': RUNS / 't3_transfer_confirmation_gpu_v1',
        'split': SPLIT_PATH}
    for field, default in defaults.items():
        parser.add_argument(f'--{field}', type=Path, default=default)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--tracks', nargs='+', choices=('T1', 'T2', 'T3'), default=['T1', 'T2', 'T3'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = {'T2': t2_stage, 'T3': partial(t3_stage, predict_pair=t3_pair, title='T3 低容量树'), 'T1': t1_stage}
    stages = {track: operation for track, operation in stages.items() if track in args.tracks}
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start, result = monotonic(), {}
        freeze_json(args.out / 'plan.json', {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                            'script': sha256(Path(__file__)),
                                            'warm_features': sha256(Path(__file__).with_name('warm_weather_regime.py')),
                                            'split_sha256': sha256(args.split),
                                            'companion_config_sha256': sha256(args.config or args.companion_code / 'configs/default.json') if any(k != 'T2' for k in stages) else None})
        for step, (track, operation) in enumerate(stages.items()):
            show_progress(args.out, '三赛道配对', step, len(stages), track)
            (args.out / track).mkdir(exist_ok=True)
            try:
                result[track] = {'status': 'complete', **operation(args)}
            except Exception as exc:
                traceback.print_exc()
                result[track] = {'status': 'failed', 'error': str(exc)}
            write_json(args.out / 'summary.json', result)
        result['seconds'] = round(monotonic()-start, 2)
        write_json(args.out / 'summary.json', result)
        show_progress(args.out, '三赛道配对', len(stages), len(stages), '已保存')
        print(f'结果摘要 {args.out / "summary.json"}', flush=True)
        return int(any(result[track]['status'] != 'complete' for track in stages))


if __name__ == '__main__':
    raise SystemExit(main())
