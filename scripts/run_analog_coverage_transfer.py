#!/usr/bin/env python3
"""Compare weather analogs, energy adjustments and three-direction transfer."""
import argparse
import fcntl

import hashlib
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic
import traceback

import numpy as np
import pandas as pd

from advance_batch_b import keys_for, read_frozen_forecast
from batch_energy_short import energy_features, fit_energy, load_daily_targets, t3_windows, weather_day_targets
from compare_t2_mlp import mix
from experiment_companion_t1 import read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, forecast_keys, sha256, write_json
from run_forecast_blends import blend, freeze_json
from run_fusion_mechanisms import finish_pending, mlp_component, stage, t3_shape_blend
from run_intraday_residuals import (BASIS, MULTIPLIER, POINT_LIMIT, context_matrix, fast_shape, project_history,
                                    standardized, t1_v4_reference, weather_context)
from run_phase_anchor import historical_reference, t2_v3_reference
from run_steamlink_followup import energy_for, station_calibration
from run_t1_issue_wind_error import dynamic_baseline
from run_t3_nearby_donors import v3_reference
from run_wind_attribution import read_arm
from same_issue_weather import feature_spec
from score_forecasts import DAY_KEYS, POINT_KEYS, MAX_PREDICTION, MIN_VALID_POINTS
from screen_companion_t1_stations import show_progress
from warm_shared_power import labels


RUNS = Path('/workspace/qhl/runs')
T1_ARMS = ('wind_tercile', 'same_site_analogs')
T2_ARMS = ('coverage_weight_050', 'anchor_residual')
T3_ARMS = ('target_only', 'target_with_pseudo_prior')
COVERAGE_MULTIPLIER = .50
ANCHOR_FRACTION = .10
TRANSFER_LIMIT = .15


class AnalogFit:
    def __init__(self, history, truth, weather):
        sites, kept, coefficients = project_history(history, truth)
        x = context_matrix(history, weather)[kept]
        use = np.isfinite(x[:, 2])
        x = x[use]
        if not len(x):
            raise ValueError('No eligible historical weather samples for analogs.')
        chosen = sites[kept][use]
        size = np.isfinite(x).sum(axis=0).clip(1)
        self.center = np.where(np.isfinite(x), x, 0).sum(axis=0) / size
        self.scale = np.sqrt(np.where(np.isfinite(x), (x-self.center)**2, 0).sum(axis=0) / size).clip(.001)
        scaled = standardized(x, self.center, self.scale)
        self.groups = {site: (scaled[chosen == site], x[chosen == site, 2], coefficients[use][chosen == site])
                       for site in sorted(set(chosen))}

    def apply(self, base, weather, names):
        x = context_matrix(base, weather)
        scaled = standardized(x, self.center, self.scale)
        sites = base.station_id.iloc[::96].to_numpy()
        delta = {name: np.zeros((len(sites), 96)) for name in names}
        for site, (fit_x, winds, fit_y) in self.groups.items():
            if len(winds) < 10:
                continue
            boundaries = np.quantile(winds, [1/3, 2/3])
            fit_bins = np.searchsorted(boundaries, winds)
            for row in np.flatnonzero((sites == site) & np.isfinite(x[:, 2])):
                choices = {}
                if T1_ARMS[0] in names:
                    choices[T1_ARMS[0]] = fit_bins == np.searchsorted(boundaries, x[row, 2])
                if T1_ARMS[1] in names:
                    distance = np.square(fit_x-scaled[row]).mean(axis=1)
                    choices[T1_ARMS[1]] = np.argsort(distance, kind='stable')[:20]
                for name, mask in choices.items():
                    outcomes = fit_y[mask]
                    if len(outcomes) >= 10:
                        delta[name][row] = MULTIPLIER * outcomes.mean(axis=0) @ BASIS.T
        for name in names:
            value = delta[name]
            value /= np.maximum(1., np.abs(value).max(axis=1) / POINT_LIMIT)[:, None]
            result = base.prediction + value.ravel()
            delta[name] = base.assign(prediction=result.clip(0, MAX_PREDICTION['T1']))
        return delta


def t1_study(args):
    model = receipt = None
    covariates = {}

    def predict(phase, train, days, sites, seed, names):
        nonlocal model, receipt
        if model is None:
            history_days, history, source = historical_reference(args, train, sites)
            weather, origin = weather_context(args, history_days, sites)
            truth = labels(args.data_root, history_days, sites)
            model = AnalogFit(history, truth, weather)
            receipt = freeze_json(args.out / 'T1/analog_fit.json', {'warm_replay': source, 'weather': origin,
                'center': model.center.tolist(), 'scale': model.scale.tolist(),
                'sites': {site: {'days': len(row[1]), 'coefficient_sha256': hashlib.sha256(row[2].tobytes()).hexdigest()}
                          for site, row in model.groups.items()}})
        if phase not in covariates:
            covariates[phase] = weather_context(args, days, sites)
        weather, lock = covariates[phase]
        keys = forecast_keys(days, sites)
        base, origin = t1_v4_reference(args, phase, keys, seed)
        return keys, {'baseline': base, **model.apply(base, weather, names)}, {
            'reference': origin, 'analog_fit': receipt, 'weather': lock}

    return predict, T1_ARMS


def anchor_design(calibrated, anchor):
    gap = anchor - calibrated
    level = np.clip(np.log(np.maximum(calibrated, 24.1026) / 100.), -1.5, 1.5)
    return np.column_stack((gap, gap * level))


def fit_anchor(calibrated, anchor, actual):
    matched = calibrated.rename(columns={'energy':'calibrated'}).merge(
        anchor.rename(columns={'energy':'anchor'}), on=DAY_KEYS, validate='one_to_one')
    matched = matched.merge(actual, on=DAY_KEYS, validate='one_to_one')
    if matched.empty:
        raise ValueError('No eligible historical T2 anchor pairs.')
    x = anchor_design(matched.calibrated, matched.anchor)
    denominator = matched.energy.to_numpy()
    baseline = (1-ANCHOR_FRACTION)*matched.calibrated + ANCHOR_FRACTION*matched.anchor
    y = (denominator-baseline.to_numpy()) / denominator
    x = x / denominator[:, None]
    weight = 1 / matched.groupby('station_id').energy.transform('size').to_numpy()
    weight /= weight.sum()
    beta = np.linalg.solve(x.T @ (x * weight[:, None]) + .10 * np.eye(x.shape[1]), x.T @ (y * weight))
    return beta, len(matched)


def historical_anchor(args, train, sites):
    days, historical, t1_lock = historical_reference(args, train, sites)
    root = args.warm_run / 'T2'
    folder = root / 'forecasts'
    lock = sha256(folder / 'predictions_frozen.json')
    if read_json(root / 'score.json')['prediction_lock'] != lock:
        raise ValueError('T2 warm replay source does not match its scoring receipt.')
    keys = keys_for(days, sites, 'T2')
    calibrated = read_arm(folder, 'station_calibrated', keys, 'T2')
    anchor = energy_for(historical, keys)
    _, actual = load_daily_targets(weather_day_targets(args.data_root, days), sites)
    beta, count = fit_anchor(calibrated, anchor, actual)
    stamp = freeze_json(args.out / 'T2/anchor_fit.json', {'warm_t1': t1_lock, 'warm_t2': lock,
                                                        'coefficients': beta.tolist(), 'eligible_days': count})
    return beta, stamp


def coverage_weight(rows, initial):
    weight = initial * np.where(rows.valid_points.lt(96), COVERAGE_MULTIPLIER, 1.)
    return weight * (initial.sum() / weight.sum())


def coverage_inputs(args, train, days, sites):
    matrix, info = energy_features(args, sites, train + days, feature_spec(args.workdir))
    points, eligible = load_daily_targets(weather_day_targets(args.data_root, train), sites)
    valid = points.loc[points.power_limit.eq(1) & np.isfinite(points.power)]
    counts = valid.groupby(DAY_KEYS).size().rename('valid_points').reset_index()
    actual = eligible.merge(counts, on=DAY_KEYS, validate='one_to_one')
    table = matrix.merge(actual, on=DAY_KEYS, validate='one_to_one')
    if len(table) != len(eligible):
        raise ValueError('Coverage tree fit lost eligible T2 station-days.')
    return matrix, info, table, eligible


def coverage_pair(args, phase, train, days, sites, seed, source, inputs, keys, anchor):
    from run_capacity_confirmation import original_t2

    _, _, baseline_source = original_t2(args, phase, seed, days, sites)
    matrix, info, table, actual = inputs
    n = len(train) // 5
    prefix, tail = train[:-n], train[-n:]
    tail_keys = keys_for(tail, sites, 'T2')
    root = args.out / 'T2' / phase / str(seed)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / 'coverage_tree'
    marker = directory / 'fit.json'
    fingerprint = {'fit': list(map(str, train)), 'forecast': list(map(str, days)), 'seed': seed,
                   'feature_cache': info['sha256'], 'algorithm': COVERAGE_MULTIPLIER,
                   'label_sha256': hashlib.sha256(table[['energy', 'valid_points']].to_numpy().tobytes()).hexdigest(),
                   'baseline': source}
    if not marker.is_file():
        temp = Path(mkdtemp(dir=root, prefix='.coverage-attempt-'))
        for name, fit, pred in [('full', train, days), ('calibration', prefix, tail)]:
            training = table.loc[table.date.isin(list(map(str, fit)))]
            future = matrix.loc[matrix.date.isin(list(map(str, pred)))]
            values, _, receipt = fit_energy(args, info['columns'], training, fit, future, 'relative_mse',
                temp / name, seed=seed, training_weight=coverage_weight)
            out = temp / name / 'forecast.csv'
            future[DAY_KEYS].assign(energy=values).to_csv(out, index=False)
            freeze_json(temp / name / 'receipt.json', {'fit': receipt, 'hash': sha256(out),
                       'partial_rows': int(training.valid_points.lt(96).sum()), 'rows': len(training)})
        freeze_json(temp / 'fit.json', {'fingerprint': fingerprint, 'branches': {
            name: sha256(temp/name/'receipt.json') for name in ('full', 'calibration')}})
        temp.rename(directory)
    record = read_json(marker)
    if record['fingerprint'] != fingerprint:
        raise ValueError('Cached coverage tree has different matched inputs.')

    def prediction(name, grid):
        folder = directory / name
        receipt = folder / 'receipt.json'
        if sha256(receipt) != record['branches'][name]:
            raise ValueError('T2 coverage tree receipt changed.')
        return read_frozen_forecast(folder / 'forecast.csv', read_json(receipt)['hash'], grid, 'T2')

    cal = baseline_source['calibration']
    history = mlp_component(args, phase, seed, tail_keys, cal['history']['MLP'], True, prefix, tail)
    mlp = mlp_component(args, phase, seed, keys, baseline_source['MLP'], False, train, days)
    factors, counts = station_calibration(mix(history, prediction('calibration', tail_keys)),
                                      actual.loc[actual.date.isin(list(map(str, tail)))])
    raw = mix(mlp, prediction('full', keys))
    candidate = raw.assign(energy=(raw.energy * raw.station_id.map(factors).fillna(1.)).clip(0, MAX_PREDICTION['T2']))
    return blend(candidate, anchor, ANCHOR_FRACTION, 'T2'), {
        'tree': sha256(marker), 'calibration_factors': factors, 'eligible_by_station': counts}


def t2_study(args):
    fitted = None
    matrices = {}

    def predict(phase, train, days, sites, seed, names):
        nonlocal fitted
        keys, calibrated, base, source = t2_v3_reference(args, phase, days, sites, seed)
        power, t1_lock = dynamic_baseline(args, phase, forecast_keys(days, sites), seed)
        anchor = energy_for(power, keys)
        if t1_lock != source['anchor'] or not np.allclose(blend(calibrated, anchor, ANCHOR_FRACTION, 'T2').energy,
                                                         base.energy, rtol=0, atol=1e-6):
            raise ValueError('Unable to reproduce the frozen T2 v3 energy anchor.')
        arms = {'baseline': base}
        if T2_ARMS[0] in names:
            if phase not in matrices:
                matrices[phase] = coverage_inputs(args, train, days, sites)
            arms[T2_ARMS[0]], source['coverage_fit'] = coverage_pair(
                args, phase, train, days, sites, seed, source.copy(), matrices[phase], keys, anchor)
        if T2_ARMS[1] in names:
            if fitted is None:
                fitted = historical_anchor(args, train, sites)
            beta, stamp = fitted
            offset = np.clip(anchor_design(calibrated.energy, anchor.energy) @ beta, -30, 30)
            arms[T2_ARMS[1]] = base.assign(energy=(base.energy + offset).clip(0, MAX_PREDICTION['T2']))
            source['anchor_fit'] = stamp
        return keys, arms, source

    return predict, T2_ARMS


def transfer_directions(tree, ridge):
    diff = (ridge.prediction - tree.prediction).to_numpy().reshape(-1, 96)
    mean = np.broadcast_to(diff.mean(axis=1, keepdims=True), diff.shape)
    fast = fast_shape(diff)
    return np.stack((mean, diff - mean - fast, fast), axis=-1)


def transfer_statistics(tree, ridge, base, truth):
    data = base.merge(truth[POINT_KEYS + ['power', 'power_limit']], on=POINT_KEYS, how='left', sort=False,
                      validate='one_to_one', indicator=True)
    if not data['_merge'].eq('both').all():
        raise ValueError('T3 transfer fit lost forecast-label keys.')
    x = transfer_directions(tree, ridge)
    target = data.power.to_numpy().reshape(-1, 96)
    valid = np.isfinite(target) & data.power_limit.eq(1).to_numpy().reshape(target.shape)
    counts = valid.sum(axis=1)
    kept = counts >= MIN_VALID_POINTS
    x = x[kept]
    weights = valid[kept] / counts[kept, None]
    residual = target - data.prediction.to_numpy().reshape(target.shape)
    rhs = np.einsum('dti,dt,dt->di', x, np.where(valid, residual, 0)[kept], weights)
    gram = np.einsum('dti,dtj,dt->dij', x, x, weights)
    sites = data.station_id.iloc[::96].to_numpy()[kept]
    return {site: (gram[sites == site].sum(axis=0), rhs[sites == site].sum(axis=0), int((sites == site).sum()))
            for site in sorted(set(sites))}


def transfer_coefficients(parts, sites, prior):
    result = {}
    for site in sites:
        rows = [part[site] for part in parts if site in part]
        gram = sum((row[0] for row in rows), np.zeros((3, 3)))
        rhs = sum((row[1] for row in rows), np.zeros(3))
        fit = np.linalg.solve(gram + 20 * np.eye(3), rhs + 20 * prior)
        result[site] = {'offset': np.clip(fit, -TRANSFER_LIMIT, TRANSFER_LIMIT).tolist(),
                        'days': sum(row[2] for row in rows)}
    return result


def pseudo_prior(args, before):
    folder = args.wind_run / 'T3_pseudo'
    plan_file = folder / 'plan.json'
    plan = read_json(plan_file)
    days = [pd.Timestamp(day).date() for day in plan['score_dates']]
    held = set(plan['sites'])
    donors = set(plan['donors'])
    historical = plan['target_train_dates'] + plan['donor_dates']
    if (not days or max(days) >= before or held & donors or len(held) != 2 or plan['seed'] != A_SEED
            or any(date >= str(min(days)) for date in historical)):
        raise ValueError('Pseudo transfer prior requires independent sites and earlier score dates.')
    keys = forecast_keys(days, plan['sites'])
    tree = folder / 'tree/fit.json'
    neural = folder / 'neural/predictions_frozen.json'
    lock = sha256(plan_file)
    if read_json(tree)['plan_sha256'] != lock or read_json(neural)['sources']['plan_sha256'] != lock:
        raise ValueError('Pseudo transfers do not match their split and source exclusion plan.')
    direct = read_frozen_forecast(tree.parent / 'short_only.csv',
        read_json(tree)['fit']['predictions_sha256']['short_only'], keys, 'T3')
    ridge = read_arm(neural.parent, 'station_ridge', keys, 'T3')
    stats = transfer_statistics(direct, ridge, t3_shape_blend(direct, ridge, .375),
                                labels(args.data_root, days, plan['sites']))
    pooled = {'source': (sum((r[0] for r in stats.values()), np.zeros((3, 3))),
                         sum((r[1] for r in stats.values()), np.zeros(3)), sum(r[2] for r in stats.values()))}
    prior = transfer_coefficients([pooled], ['source'], np.zeros(3))['source']
    stamp = freeze_json(args.out / 'T3/pseudo_prior.json', {'plan': lock, 'tree': sha256(tree),
                                                          'neural': sha256(neural), **prior})
    return np.asarray(prior['offset']), stamp


def t3_study(args):
    prior = None
    historical = {}

    def predict(window, train, days, sites, seed, names):
        nonlocal prior
        if prior is None:
            prior = pseudo_prior(args, min(train))
        for earlier, fit, evaluate in list(t3_windows())[:2]:
            if set(evaluate).issubset(train) and earlier not in historical:
                _, tree, ridge, base, origin = v3_reference(args, earlier, fit, evaluate, sites, A_SEED)
                historical[earlier] = (transfer_statistics(tree, ridge, base,
                                                           labels(args.data_root, evaluate, sites)), origin)
        parts = [record[0] for record in historical.values()]
        keys, tree, ridge, base, origin = v3_reference(args, window, train, days, sites, seed)
        directions = transfer_directions(tree, ridge)
        models, arms = {}, {'baseline': base}
        for name in names:
            center = prior[0] if name == T3_ARMS[1] else np.zeros(3)
            model = transfer_coefficients(parts, sites, center)
            offset = np.asarray([model[site]['offset'] for site in keys.station_id.iloc[::96]])
            correction = np.einsum('dti,di->dt', directions, offset).ravel()
            arms[name] = base.assign(prediction=(base.prediction + correction).clip(0, MAX_PREDICTION['T3']))
            models[name] = model
        return keys, arms, {'reference': origin, 'pseudo_prior': prior[1], 'site_models': models,
                            'target_folds': {name: row[1] for name, row in historical.items()}}

    return predict, T3_ARMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out': RUNS / 'analog_coverage_transfer_gpu_v1', 'data-root': Path('/workspace/qhl/data'),
                'split': SPLIT_PATH, 'workdir': Path('/workspace/cpy/code/workdir'),
                'phase-run': RUNS / 'phase_anchor_cpu_v1', 'dynamic-run': RUNS / 't1_dynamic_gradient_gpu_v1',
                'warm-run': RUNS / 'strict_warm_replay_gpu_v1', 't2-prior': RUNS / 'steamlink_followup_gpu_v1',
                'fusion-run': RUNS / 'fusion_mechanisms_cpu_v1', 'ridge-run': RUNS / 'daily_bias_ridge_gpu_v1',
                'short-tree': RUNS / 'three_track_gpu_batch_v1/T3', 'transfer-run': RUNS / 'same_issue_followup_v1/T3',
                't3-confirm': RUNS / 't3_transfer_confirmation_gpu_v1', 'wind-run': RUNS / 'wind_attribution_gpu_v1'}
    for key, value in defaults.items():
        parser.add_argument(f'--{key}', type=Path, default=value)
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_json(args.out / 'plan.json', {
            'args': {name: str(value) for name, value in vars(args).items() if name != 'split'},
            'split': sha256(args.split), 'seeds': list(B_SEEDS),
            'code': {name: sha256(Path(__file__).with_name(name)) for name in ('run_analog_coverage_transfer.py',
                'run_intraday_residuals.py', 'run_phase_anchor.py', 'run_fusion_mechanisms.py', 'batch_energy_short.py')}})
        inventory = RUNS / 't1_source_power_gpu_v1' / 'summary.json'
        result = {'source_power_inventory': {'path': str(inventory),
                  'status': 'available' if inventory.is_file() else 'report_missing'}}
        if inventory.is_file():
            result['source_power_inventory']['sha256'] = sha256(inventory)
        start, pending = monotonic(), {}
        studies = {'T1': t1_study(args), 'T2': t2_study(args), 'T3': t3_study(args)}
        for i, (track, study) in enumerate(studies.items()):
            show_progress(args.out, '天气检索与迁移', i, len(studies), track)
            try:
                result[track] = {'status': 'complete', **stage(args, track, pending, study=study)}
            except Exception as error:
                traceback.print_exc()
                result[track] = {'status': 'failed', 'error': str(error)}
            write_json(args.out / 'summary.json', result)
        finish_pending(args, result, pending)
        result['seconds'] = round(monotonic() - start, 2)
        write_json(args.out / 'summary.json', result)
        show_progress(args.out, '天气检索与迁移', len(studies), len(studies), '结果已保存')
        print(f'结果摘要 {args.out / "summary.json"}', flush=True)
        return int(any(result[track]['status'] != 'complete' for track in studies))


if __name__ == '__main__':
    raise SystemExit(main())
