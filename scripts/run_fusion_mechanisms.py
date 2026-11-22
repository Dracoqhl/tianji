#!/usr/bin/env python3
"""Compare three low-cost fusion changes against the accepted track baselines."""
import argparse
import fcntl
from functools import partial
from pathlib import Path
from time import monotonic
import traceback

import numpy as np

from advance_batch_b import keys_for, read_frozen_forecast
from batch_energy_short import roster_for, t3_windows, weather_day_targets
from compare_t2_mlp import mix
from compare_warm_t1 import PRIMARY_SCORE, b_statistics
from confirm_working_baselines import average_seeds
from experiment_companion_t1 import identical_seed_groups, read_json
from independent_t2 import load_daily_targets
from run_capacity_confirmation import cached_direct, original_t2
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, chronological_dates, forecast_keys, sha256, write_json
from run_daily_bias_ridge import RIDGE_NEW
from run_forecast_blends import blend, freeze_json, freeze_window
from run_pressure_profile import summarize_b
from run_steamlink_followup import confirmed_t3, corrected_sources, energy_for, finite_mean, mapped, t2_scores
from run_t1_issue_wind_error import dynamic_baseline, pair_for
from run_t1_source_power import simplex_weights
from run_t1_t3_batch import paired_report
from run_wind_attribution import STATION, observed_curve, read_arm, wind_labels
from same_issue_weather import archived_issue_weather
from score_forecasts import DAY_KEYS, MAX_PREDICTION, MIN_VALID_POINTS, POINT_KEYS, score_track
from screen_companion_t1_stations import block_comparisons, show_progress, station_scores
from warm_shared_power import labels, load_roster


RUNS = Path('/workspace/qhl/runs')
LEAD = 24
FOLD = 30
EQUAL = np.full(3, 1 / 3)
T1_ARMS = ('lead_weights', 'lead_disagreement')
T2_ARMS = ('t1_v3_integral_10', 'disagreement_calibration')
T3_SHAPE = {'shape_12_5': .125, 'shape_37_5': .375}
T2_PRIOR_DAYS = 60


def disagreement(sources):
    finite = np.isfinite(sources)
    high = np.max(np.where(finite, sources, -np.inf), axis=0)
    low = np.min(np.where(finite, sources, np.inf), axis=0)
    return np.where(finite.sum(axis=0) >= 2, high - low, np.nan)


def weighted_wind(sources, weights):
    known = np.isfinite(sources)
    available = np.sum(weights * known, axis=0)
    return np.divide(np.sum(weights * np.where(known, sources, 0), axis=0), available,
                     out=finite_mean(sources), where=available > 0)


def fit_source_weights(oof, observed, scoreable, future):
    fit_range, future_range = disagreement(oof), disagreement(future)
    weighted = {name: np.empty(future.shape[1:]) for name in T1_ARMS}
    receipt = {}
    for start in range(0, 96, LEAD):
        lead = slice(start, start + LEAD)
        x = np.moveaxis(oof[..., lead], 0, -1)
        y = observed[..., lead]
        aligned = scoreable[..., lead] & np.isfinite(y) & (y >= 0) & np.isfinite(x).all(axis=-1)

        def solve(mask, prior):
            return simplex_weights(x[mask], y[mask], np.full(mask.sum(), 1 / mask.sum())) if mask.sum() >= 300 else prior

        pooled = solve(aligned, EQUAL)
        threshold = float(np.median(fit_range[..., lead][aligned])) if aligned.any() else None
        high = np.zeros_like(aligned)
        if threshold is not None:
            high = fit_range[..., lead] > threshold
        low_fit = aligned & ~high
        high_fit = aligned & high
        group_weights = [solve(mask, pooled) for mask in (low_fit, high_fit)]
        future_lead = future[..., lead]
        group_forecasts = [weighted_wind(future_lead, weight[:, None, None, None]) for weight in group_weights]
        overall = weighted_wind(future_lead, pooled[:, None, None, None])
        weighted[T1_ARMS[0]][..., lead] = overall
        widths = future_range[..., lead]
        if threshold is None:
            weighted[T1_ARMS[1]][..., lead] = overall
        else:
            weighted[T1_ARMS[1]][..., lead] = np.where(
                np.isfinite(widths), np.where(widths > threshold, group_forecasts[1], group_forecasts[0]), overall)
        receipt[str(start + 1)] = {'lead_points': [start + 1, start + LEAD], 'weights': pooled.tolist(),
                                   'split': threshold, 'group_weights': [w.tolist() for w in group_weights],
                                   'rows': [int(low_fit.sum()), int(high_fit.sum())]}
    return weighted, receipt


def t1_weather(args, train, days, sites):
    count = len(train)
    weather, columns, stamp = archived_issue_weather(args.workdir, args.data_root, train + days, sites, print)
    weather = weather.reshape(count + len(days), len(sites), 96, len(columns))
    truth = wind_labels(args.data_root, train, sites)
    shape = (count, len(sites), 96)
    wind = truth.wind.to_numpy(dtype=float).reshape(shape)
    power = truth.power.to_numpy(dtype=float).reshape(shape)
    valid = truth.power_limit.eq(1).to_numpy().reshape(shape) & np.isfinite(power)
    curves = [observed_curve(wind[:, i], power[:, i], valid[:, i]) for i in range(len(sites))]
    first = count - 3 * FOLD
    if first <= 0 or max(train) >= min(days):
        raise ValueError('T1 source calibration requires a strictly earlier training prefix.')
    folds = []
    for i in range(first, count, FOLD):
        print(f'[T1 OOF] {train[i]}–{train[i+FOLD-1]}', flush=True)
        folds.append(corrected_sources(weather, columns, wind, slice(0, i), slice(i, i+FOLD)))
    future = corrected_sources(weather, columns, wind, slice(0, count), slice(count, None))
    weights, receipt = fit_source_weights(np.concatenate(folds, axis=1), wind[first:], valid[first:], future)
    fixed = mapped(finite_mean(future), curves)
    alternatives = {}
    for name, adjusted in weights.items():
        candidate = mapped(adjusted, curves)
        # Retain the accepted exception at this station.
        if STATION in sites:
            candidate[:, sites.index(STATION)] = fixed[:, sites.index(STATION)]
        alternatives[name] = {'baseline_branch': fixed, 'candidate_branch': candidate}
    return alternatives, {'weather': stamp, 'weights': receipt,
                          'folds': [[str(train[i]), str(train[i+FOLD-1])] for i in range(first, count, FOLD)]}


def t1_predict(args, phase, train, days, sites, seed, names, weather=None):
    keys = forecast_keys(days, sites)
    alternatives, fit = weather or t1_weather(args, train, days, sites)
    arms, receipts = {}, {}
    for name in names:
        pair, receipts[name] = pair_for(args, phase, seed, keys, alternatives[name])
        arms['baseline'], arms[name] = pair['baseline'], pair['candidate']
    return keys, arms, {'calibration': fit, 'references': receipts}


def mlp_component(args, phase, seed, keys, expected, calibration, train, days):
    root = args.t2_prior / 'T2' / phase / str(seed) / ('calibration' if calibration else 'full')
    receipt = read_json(root / 'mlp/receipt.json')
    scope = read_json(root / 'plan.json')
    if (receipt != expected or scope['train'] != list(map(str, train)) or scope['forecast'] != list(map(str, days))
            or scope['seed'] != seed or scope['calibration'] != calibration
            or scope['recipe'] != 'warm_shared_power:16:64'
            or sha256(root / 'mlp/training_receipt.json') != receipt['training_sha256']):
        raise ValueError('The frozen T2 MLP does not match its calibration contract.')
    return read_frozen_forecast(root / 'mlp/T2_development.csv', receipt['forecast_sha256'], keys, 'T2')


def correction_fit(mlp, direct, actual, factors):
    keys = mlp[DAY_KEYS]
    pooled = mix(mlp, direct)
    calibrated = pooled.energy * pooled.station_id.map(factors).fillna(1)
    frame = keys.assign(pred=calibrated.clip(0, MAX_PREDICTION['T2']), mlp_higher=mlp.energy >= direct.energy)
    frame = frame.merge(actual[DAY_KEYS + ['energy']], on=DAY_KEYS, validate='one_to_one')
    ratio = frame.pred / frame.energy
    receipt = {}
    for group in (False, True):
        # Equal station contribution within each group, with a fixed shrinkage toward identity.
        sample = frame.loc[frame.mlp_higher.eq(group)].assign(ratio=ratio, square=ratio**2)
        stats = sample.groupby('station_id')[['ratio', 'square']].mean()
        mass = len(sample)
        numerator, denominator = stats.mean().fillna(0).tolist()
        factor = (mass * numerator + T2_PRIOR_DAYS) / (mass * denominator + T2_PRIOR_DAYS)
        receipt[str(group)] = {'factor': float(np.clip(factor, .9, 1.1)), 'days': mass}
    return receipt


def t2_predict(args, phase, train, days, sites, seed, names):
    keys = keys_for(days, sites, 'T2')
    if phase == 'B' and not read_json(args.t2_prior / 'T2/B/b_score.json')['acceptance']['accepted']:
        raise ValueError('T2 calibrated source did not pass its paired acceptance gate.')
    base, lock, source = original_t2(args, phase, seed, days, sites)
    arms, meta = {'baseline': base}, {'base_lock': lock}
    if T2_ARMS[0] in names:
        curve, meta['t1_lock'] = dynamic_baseline(args, phase, forecast_keys(days, sites), seed)
        arms[T2_ARMS[0]] = blend(base, energy_for(curve, keys), .10, 'T2')
    if T2_ARMS[1] in names:
        n = len(train) // 5
        prefix, tail = train[:-n], train[-n:]
        cal_keys = keys_for(tail, sites, 'T2')
        cal = source['calibration']
        history = mlp_component(args, phase, seed, cal_keys, cal['history']['MLP'], True, prefix, tail)
        hist_tree, meta['cal_tree'] = cached_direct(args, phase, seed, cal_keys, True, source)
        _, actual = load_daily_targets(weather_day_targets(args.data_root, tail), sites)
        correction = correction_fit(history, hist_tree, actual, cal['factors'])
        current = mlp_component(args, phase, seed, keys, source['MLP'], False, train, days)
        current_tree, meta['full_tree'] = cached_direct(args, phase, seed, keys, False, source)
        reconstructed = (mix(current, current_tree).energy * keys.station_id.map(cal['factors']).fillna(1)).clip(
            0, MAX_PREDICTION['T2'])
        if not np.allclose(reconstructed, base.energy, rtol=0, atol=1e-6):
            raise ValueError('Current T2 components do not reproduce the frozen baseline.')
        factor = (current.energy >= current_tree.energy).astype(str).map(
            {group: row['factor'] for group, row in correction.items()})
        arms[T2_ARMS[1]] = base.assign(energy=(base.energy * factor).clip(0, MAX_PREDICTION['T2']))
        meta['correction'] = correction
    return keys, arms, meta


def t3_components(args, window, train, days, sites, seed):
    keys, tree, _, _, original_lock = confirmed_t3(args, window, train, days, sites, seed)
    root = args.ridge_run / 'T3' / window
    folder = root / str(seed)
    lock = sha256(folder / 'forecasts/predictions_frozen.json')
    if window == 'fixed_50_10':
        pair = folder / 'pair/predictions_frozen.json'
        if read_json(root / 'predictions_frozen.json')[str(seed)] != sha256(pair):
            raise ValueError('T3 ridge reference does not match its fixed-window lock.')
        reference = read_json(pair)['sources']['forecast_lock']
        if not read_json(root / 'b_score.json')['acceptance']['accepted']:
            raise ValueError('T3 ridge source was not accepted.')
    else:
        reference = read_json(root / 'score.json')['prediction_lock']
    inputs = read_json(folder / 'forecasts/predictions_frozen.json')['sources']
    base = read_arm(folder / 'forecasts', 'candidate', keys, 'T3')
    receipt = folder / 'heads/worker_receipt.json'
    if lock != reference or inputs['reference_lock'] != original_lock or inputs['fit_receipt'] != sha256(receipt):
        raise ValueError('T3 ridge forecast is not linked to its paired reference.')
    ridge_name = f'ridge_{RIDGE_NEW:.2f}'
    new = read_frozen_forecast(folder / 'heads' / f'{ridge_name}.csv',
                               read_json(receipt)['hashes'][ridge_name], keys, 'T3')
    if not np.allclose(blend(tree, new, .25, 'T3').prediction, base.prediction, atol=1e-6, rtol=0):
        raise ValueError('T3 components do not reproduce the accepted fusion.')
    return keys, tree, new, base, {'reference': lock, 'source': original_lock}


def t3_shape_blend(tree, ridge, fraction, *, base=None):
    if base is None:
        base = blend(tree, ridge, .25, 'T3')
    delta = (ridge.prediction - tree.prediction).to_numpy().reshape(-1, 96)
    shape = (delta - delta.mean(axis=1, keepdims=True)).ravel()
    return base.assign(prediction=(base.prediction + (fraction - .25) * shape).clip(0, MAX_PREDICTION['T3']))


def t3_predict(args, window, train, days, sites, seed, names):
    keys, tree, ridge, base, source = t3_components(args, window, train, days, sites, seed)
    return keys, {'baseline': base, **{name: t3_shape_blend(tree, ridge, T3_SHAPE[name], base=base)
                                       for name in names}}, source


def lead_rmse(truth, arms):
    usable = truth.loc[truth.power_limit.eq(1) & np.isfinite(truth.power), POINT_KEYS + ['power']]
    count = usable.groupby(DAY_KEYS).power.transform('size')
    usable = usable.loc[count.ge(MIN_VALID_POINTS)]
    report = {}
    for name, frame in arms.items():
        rows = usable.merge(frame, on=POINT_KEYS, validate='one_to_one')
        square = (rows.power - rows.prediction)**2
        values = square.groupby((rows.point_index - 1) // LEAD).agg(['mean', 'size'])
        report[name] = {str(int(index) * LEAD + 1): {'rmse_mw': float(np.sqrt(row['mean'])),
                        'points': int(row['size'])} for index, row in values.iterrows()}
    return report


def score_window(args, track, days, sites, arms):
    if track == 'T2':
        scores, groups = t2_scores(args, days, sites, arms)
        return {'scores': scores, 'groups': groups}
    truth = labels(args.data_root, days, sites)
    if track == 'T1':
        report = block_comparisons(truth, arms, sites, tuple(name for name in arms if name != 'baseline'))
        return {**report, 'lead_rmse': lead_rmse(truth, arms)}
    return {'scores': {name: dict(score_track('T3', truth, frame, sites, allow_unscored_stations=True),
                                 by_station=station_scores(truth, frame)) for name, frame in arms.items()}}


def stage(args, track, pending=None):
    root = args.out / track
    root.mkdir(parents=True, exist_ok=True)
    sites = roster_for(args.data_root, track) if track != 'T1' else load_roster(args.data_root)
    if track == 'T3':
        *screen, (final_window, fit_b, days_b) = t3_windows()
        predict, names = partial(t3_predict, args), tuple(T3_SHAPE)
    else:
        train, days = chronological_dates(args.split, 'A')
        fit_b, days_b = chronological_dates(args.split, 'B')
        if fit_b != train + days:
            raise ValueError('B fit must include all A dates.')
        screen, final_window = [('A', train, days)], 'B'
        predict, names = (partial(t1_predict, args), T1_ARMS) if track == 'T1' else (
            partial(t2_predict, args), T2_ARMS)

    initial, screen_locks = {}, {}
    for i, (window, train, days) in enumerate(screen):
        show_progress(root, f'{track} 筛选', i, len(screen), window)
        keys, arms, source = predict(window, train, days, sites, A_SEED, names)
        frames, lock = freeze_window(root / window / 'forecasts', track, keys, arms, source)
        report = score_window(args, track, days, sites, frames)
        report['score_values'] = {name: row[PRIMARY_SCORE] for name, row in report['scores'].items()}
        freeze_json(root / window / 'score.json', {**report, 'lock': lock})
        initial[window] = report['score_values']
        screen_locks[window] = lock
    gains = {name: float(np.mean([row[name] - row['baseline'] for row in initial.values()])) for name in names}
    selected = max(gains, key=gains.get)
    promoted = gains[selected] > 0
    gate = {'selected': selected, 'deltas': gains, 'promoted': promoted, 'screen_locks': screen_locks}
    freeze_json(root / 'gate.json', gate)
    show_progress(root, f'{track} 筛选', len(screen), len(screen), str(gate))
    result = {'A': initial, 'gate': gate}
    if not promoted:
        return result

    extra = {'weather': t1_weather(args, fit_b, days_b, sites)} if track == 'T1' else {}
    pairs, locks = {}, {}
    for i, seed in enumerate(B_SEEDS):
        show_progress(root, f'{track} 三种子复核', i, len(B_SEEDS), str(seed))
        keys, arms, source = predict(final_window, fit_b, days_b, sites, seed, (selected,), **extra)
        pair = {'baseline': arms['baseline'], 'candidate': arms[selected]}
        pairs[seed], locks[str(seed)] = freeze_window(root / 'B' / str(seed), track, keys, pair,
                                                      {'source': source, 'gate': gate})
    lock = freeze_json(root / 'B/predictions_frozen.json', locks)
    finish = partial(score_final, args, track, root, days_b, sites, keys, pairs, lock)
    if pending is None:
        result['B'] = finish()
    else:
        pending[track] = finish
        result['B_status'] = 'predictions_frozen'
    return result


def score_final(args, track, root, days_b, sites, keys, pairs, lock):
    if track != 'T2':
        truth = labels(args.data_root, days_b, sites)
        report = paired_report(root / 'B', track, truth, sites, keys, pairs, lock)
        result = summarize_b(report)
        if track == 'T1':
            freeze_json(root / 'B/lead_rmse.json', {'lock': lock,
                        'arms': lead_rmse(truth, average_seeds(keys, pairs, track))})
    else:
        def score(arms):
            report, groups = t2_scores(args, days_b, sites, arms)
            values = {name: row[PRIMARY_SCORE] for name, row in report.items()}
            return values['candidate'] - values['baseline'], values, groups
        deltas = {str(seed): score(pair)[0] for seed, pair in pairs.items()}
        delta, values, groups = score(average_seeds(keys, pairs, track))
        result = {'scores': values, 'seed_deltas': deltas, 'groups': groups,
                  'acceptance': b_statistics(list(deltas.values()), delta),
                  'identical_seed_groups': {arm: identical_seed_groups(
                      {seed: pair[arm].energy.to_numpy() for seed, pair in pairs.items()})
                      for arm in ('baseline', 'candidate')}}
        freeze_json(root / 'B/score.json', {**result, 'prediction_lock': lock})
    show_progress(root, f'{track} 三种子复核', len(B_SEEDS), len(B_SEEDS), str(result['acceptance']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out': RUNS / 'fusion_mechanisms_cpu_v1', 'data-root': Path('/workspace/qhl/data'),
                'workdir': Path('/workspace/cpy/code/workdir'), 'split': SPLIT_PATH,
                'dynamic-run': RUNS / 't1_dynamic_gradient_gpu_v1', 'wind-run': RUNS / 'wind_attribution_gpu_v1',
                'power-run': RUNS / 't1_t3_fast_gpu_v1', 't2-prior': RUNS / 'steamlink_followup_gpu_v1',
                'transfer-run': RUNS / 'same_issue_followup_v1/T3', 'short-tree': RUNS / 'three_track_gpu_batch_v1/T3',
                't3-confirm': RUNS / 't3_transfer_confirmation_gpu_v1', 'ridge-run': RUNS / 'daily_bias_ridge_gpu_v1'}
    for name, value in defaults.items():
        parser.add_argument(f'--{name}', type=Path, default=value)
    parser.add_argument('--device', choices=('cuda',), default='cuda', help='Expected device of frozen source models.')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_json(args.out / 'plan.json', {
            'args': {name: str(value) for name, value in vars(args).items()}, 'split': sha256(args.split),
            'code': sha256(Path(__file__)), 'seeds': list(B_SEEDS)})
        start, result, pending = monotonic(), {}, {}
        for i, track in enumerate(('T1', 'T2', 'T3')):
            show_progress(args.out, '融合机制', i, 3, track)
            try:
                result[track] = {'status': 'complete', **stage(args, track, pending)}
            except Exception as error:
                traceback.print_exc()
                result[track] = {'status': 'failed', 'error': str(error)}
            write_json(args.out / 'summary.json', result)
        shared_fit_failed = any(result[track]['status'] == 'failed' for track in ('T1', 'T2'))
        for track, finish in pending.items():
            if track in ('T1', 'T2') and shared_fit_failed:
                result[track].update(status='failed', B_status='waiting_for_shared_track')
                continue
            try:
                result[track].update(B=finish(), B_status='scored')
            except Exception as error:
                traceback.print_exc()
                result[track].update(status='failed', B_status='scoring_failed', error=str(error))
            write_json(args.out / 'summary.json', result)
        result['seconds'] = round(monotonic() - start, 2)
        write_json(args.out / 'summary.json', result)
        show_progress(args.out, '融合机制', 3, 3, '结果已保存')
        print(f'结果摘要 {args.out / "summary.json"}', flush=True)
        return int(any(result[track]['status'] != 'complete' for track in ('T1', 'T2', 'T3')))


if __name__ == '__main__':
    raise SystemExit(main())
