#!/usr/bin/env python3
"""Run the Steamlink-inspired three-track chronological replay."""
import argparse
import fcntl
from functools import partial
from pathlib import Path
import traceback
from tempfile import mkdtemp
from types import SimpleNamespace

import numpy as np
import pandas as pd

from advance_batch_b import keys_for, read_frozen_forecast
from batch_energy_short import (energy_arm_report, energy_features, fit_energy, fit_short_forecasts,
                                roster_for, short_source, t3_windows, weather_day_targets)
from compare_t2_mlp import mix, select_mlp_runtime
from compare_warm_t1 import PRIMARY_SCORE, b_statistics
from confirm_working_baselines import average_seeds, keep_sites
from experiment_companion_t1 import assert_matched, identical_seed_groups, read_json
from independent_t2 import load_daily_targets
from run_companion_warm_t1 import (A_SEED, B_SEEDS, EARLY_STOP_DAYS, SPLIT_PATH, chronological_dates,
                                   forecast_keys, run as run_tree, sha256, write_json)
from run_forecast_blends import blend, freeze_json, freeze_window
from run_t1_t3_batch import call, paired_report, point_diagnostics
from run_t1_wind_aux import prior_baseline
from run_wind_attribution import (RUNS, SOURCES, STATION, baseline_a, baseline_b, map_wind, observed_curve,
                                  pseudo_dates, read_arm, wind_features, wind_labels)
from same_issue_weather import archive_stamp, archived_issue_weather, feature_spec
from score_forecasts import DAY_KEYS, MAX_PREDICTION
from screen_companion_t1_stations import block_comparisons, show_progress, station_scores
from steamlink_native_wind import NAMES, append_native
from warm_shared_power import fit_forecasts, labels, load_roster, power_weather

ENTRY = Path(__file__).resolve()
POWER_MEAN = 'mean_source_power'
RESIDUAL = 'wind_residual_integral'
NATIVE = 'native_shape'
CALIBRATED = 'station_calibrated'
INTEGRAL = 't1_v2_integral'


def finite_mean(values):
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    return np.divide(np.where(finite, values, 0).sum(axis=0), count,
                     out=np.full(count.shape, np.nan), where=count > 0)


def corrected_sources(weather, columns, wind, fit_idx, eval_idx):
    forecasts = []
    for source in SOURCES:
        features = wind_features(weather, columns, source, True)
        forecasts.append(np.stack([
            map_wind(features[fit_idx, site], wind[fit_idx, site], features[eval_idx, site], True)[0]
            for site in range(wind.shape[1])
        ], axis=1))
    return np.stack(forecasts)


def mapped(forecast, curves, *, fill=True):
    output = np.empty(forecast.shape, dtype=float)
    for site, (support, values, fallback) in enumerate(curves):
        source = forecast[:, site]
        power = np.interp(np.where(np.isfinite(source), source, fallback), support, values)
        output[:, site] = power if fill else np.where(np.isfinite(source), power, np.nan)
    return np.clip(output, 0, MAX_PREDICTION['T1'])


def residual_integral(tail_prediction, tail_truth, forecast, curves):
    result = mapped(forecast, curves)
    quantiles = (np.arange(9) + .5) / 9
    receipt = {}
    for site, (support, values, _) in enumerate(curves):
        prediction = tail_prediction[:, site]
        observed = tail_truth[:, site]
        valid = np.isfinite(prediction) & np.isfinite(observed) & (observed >= 0)
        if not valid.any():
            continue
        residual = observed - prediction
        thresholds = np.quantile(prediction[valid], [1/3, 2/3])
        bins, future = np.digitize(prediction, thresholds), np.digitize(forecast[:, site], thresholds)
        population = np.quantile(residual[valid], quantiles)
        receipt[str(site)] = {'speed_thresholds': thresholds.tolist(), 'bins': {}}
        for lead in range(0, 96, 24):
            columns = slice(lead, lead + 24)
            for group in range(3):
                sample = residual[:, columns][valid[:, columns] & (bins[:, columns] == group)]
                xf = forecast[:, site, columns]
                use = (future[:, columns] == group) & np.isfinite(xf)
                w = len(sample) / (len(sample) + 500)
                global_power = np.interp(np.maximum(xf[use][:, None] + population, 0), support, values).mean(axis=-1)
                local = np.quantile(sample, quantiles) if len(sample) else population
                local_power = np.interp(np.maximum(xf[use][:, None] + local, 0), support, values).mean(axis=-1)
                result[:, site, columns][use] = (1-w) * global_power + w * local_power
                receipt[str(site)]['bins'][f'{lead}:{group}'] = int(len(sample))
    return np.clip(result, 0, MAX_PREDICTION['T1']), receipt


def wind_alternatives(args, train, days, roster, selected=None):
    count = len(train)
    weather, columns, stamp = archived_issue_weather(args.workdir, args.data_root, [*train, *days], roster, print)
    weather = weather.reshape(count + len(days), len(roster), 96, len(columns))
    truth = wind_labels(args.data_root, train, roster)
    shape = (count, len(roster), 96)
    wind = truth.wind.to_numpy(dtype=float).reshape(shape)
    power = truth.power.to_numpy(dtype=float).reshape(shape)
    valid = truth.power_limit.eq(1).to_numpy().reshape(shape)
    curves = [observed_curve(wind[:, i], power[:, i], valid[:, i]) for i in range(len(roster))]
    active = valid & np.isfinite(power)
    adjacent = active[..., 1:] & active[..., :-1]
    hour = active[..., 4:] & active[..., :-4]
    threshold = (float(np.quantile(np.abs(np.diff(power, axis=-1))[adjacent], .9)),
                 float(np.quantile(np.abs(power[..., 4:] - power[..., :-4])[hour], .9)))
    sources = corrected_sources(weather, columns, wind, slice(0, count), slice(count, None))
    mean_wind = finite_mean(sources)
    reference = mapped(mean_wind, curves)
    result, details = {}, {'weather': stamp}
    if selected in (None, POWER_MEAN):
        values = finite_mean(np.stack([mapped(values, curves, fill=False) for values in sources]))
        result[POWER_MEAN] = np.where(np.isfinite(values), values, reference)
    if selected in (None, RESIDUAL):
        calibration = []
        for first in range(count - 90, count, 30):
            estimates = corrected_sources(weather, columns, wind, slice(0, first), slice(first, first + 30))
            calibration.append(finite_mean(estimates))
        result[RESIDUAL], details['residual'] = residual_integral(np.concatenate(calibration), wind[-90:], mean_wind, curves)
        details['calibration_folds'] = [[str(train[i]), str(train[i + 29])] for i in range(count - 90, count, 30)]
    return reference, result, threshold, details


def native_forecast(args, phase, seed, keys):
    directory = args.out / 'T1' / phase / str(seed) / NATIVE
    directory.parent.mkdir(parents=True, exist_ok=True)
    old = args.power_run / 'T1' / phase / str(seed) / 'tree' / 'training_receipt.json'
    baseline = read_json(old)
    marker = directory / 'training_receipt.json'
    if not marker.is_file():
        pending = Path(mkdtemp(dir=directory.parent, prefix='.native-attempt-'))
        options = SimpleNamespace(**vars(args))
        options.out, options.seed, options.phase = pending, seed, phase
        options.chronological, options.train_block, options.smoke, options.check_only = True, None, False, False
        options.backend, options.components, options.point_partition = 'xgboost', [], 'full'
        options.early_stop_days, options.diagnostic = EARLY_STOP_DAYS, None
        run_tree(options, extra_features=partial(append_native, args.data_root, args.out / 'native_weather'))
        pending.rename(directory)
    info = read_json(marker)
    matched = ('seed', 'chronological_dates', 'early_stopping_days_inside_train', 'backend', 'device',
               'threads', 'config_sha256', 'companion_cache_metadata_sha256', 'valid_training_sample_sha256',
               'feature_spec_sha256', 'companion_source_sha256')
    assert_matched(baseline, info, matched)
    if info['feature_names'] != baseline['feature_names'] + NAMES:
        raise ValueError('Wrong native feature schema.')
    return read_frozen_forecast(directory / info['prediction_file'], info['prediction_sha256'], keys, 'T1'), sha256(marker)


def old_t1(args, phase, keys, seed):
    source = SimpleNamespace(prior=args.power_run)
    return baseline_a(source, keys) if phase == 'A' else baseline_b(source, keys, seed)


def t1_pred(args, phase, seed, keys, reference, alternatives, names):
    baseline, lock = prior_baseline(args, phase, keys, seed)
    prior, stamp = old_t1(args, phase, keys, seed)
    power = keys.assign(prediction=reference.ravel())
    replay = blend(prior, power, .2, 'T1')
    if not np.allclose(replay.prediction, baseline.prediction, atol=1e-6, rtol=0):
        raise ValueError('Rebuilt wind branch differs from the accepted T1 v2.')
    forecasts = {'baseline': baseline}
    receipts = {'v2': lock, 'prior': stamp}
    for name in names:
        if name == NATIVE:
            updated, receipts[NATIVE] = native_forecast(args, phase, seed, keys)
            updated = keep_sites(updated, prior, [STATION])
            forecasts[name] = blend(updated, power, .2, 'T1')
        else:
            forecasts[name] = blend(prior, keys.assign(prediction=alternatives[name].ravel()), .2, 'T1')
    return forecasts, receipts


def ramp_timing(truth, forecasts, threshold):
    keys = truth[['station_id', 'date', 'point_index']]
    actual = truth.power.to_numpy(dtype=float).reshape(-1, 96)
    usable = truth.power_limit.eq(1).to_numpy().reshape(actual.shape) & np.isfinite(actual)
    comparable = (usable[:, 4:] & usable[:, :-4]) & (usable.sum(axis=1, keepdims=True) >= 72)
    change = actual[:, 4:] - actual[:, :-4]
    report = {}
    for name, forecast in forecasts.items():
        power = keys.merge(forecast, on=list(keys), validate='one_to_one').prediction.to_numpy().reshape(actual.shape)
        predicted = power[:, 4:] - power[:, :-4]
        events = {}
        for label, sign in (('up', 1), ('down', -1)):
            observed = np.where(comparable, sign * change, -np.inf)
            present = observed.max(axis=1) >= threshold
            target = np.argmax(observed[present], axis=1)
            scores = np.where(comparable[present], sign * predicted[present], -np.inf)
            located = scores.max(axis=1) > 0
            location = np.argmax(scores[located], axis=1)
            deviation = location - target[located]
            events[label] = {'actual_events': int(present.sum()), 'forecast_events': int(located.sum()),
                             'timing_mae_minutes': float(np.abs(deviation).mean()*15) if len(deviation) else None,
                             'timing_bias_minutes': float(deviation.mean()*15) if len(deviation) else None}
        report[name] = events
    return report


def t1_stage(args):
    directory = args.out / 'T1'
    roster = load_roster(args.data_root)
    train, days = chronological_dates(args.split, 'A')
    keys = forecast_keys(days, roster)
    names = (POWER_MEAN, RESIDUAL, NATIVE)
    print('[T1 A] 重建已采用的三源订正，比较功率映射顺序与原生天气特征', flush=True)
    ref, alternatives, threshold, details = wind_alternatives(args, train, days, roster)
    frames, source = t1_pred(args, 'A', A_SEED, keys, ref, alternatives, names)
    frames, lock = freeze_window(directory / 'A' / 'forecasts', 'T1', keys, frames, {'fits': details, 'source': source})
    truth = labels(args.data_root, days, roster)
    comparison = block_comparisons(truth, frames, roster, names)
    scores = {k: v[PRIMARY_SCORE] for k, v in comparison['scores'].items()}
    chosen = max(names, key=scores.get)
    gain = scores[chosen] - scores['baseline']
    gate = {'selected': chosen, 'delta': gain, 'promoted': gain > 0, 'forecast_lock': lock}
    freeze_json(directory / 'a_score.json', {'scores': comparison, 'gate': gate,
                                               'ramp': point_diagnostics(truth, frames, threshold[0]),
                                               'hour_ramp_timing': ramp_timing(truth, frames, threshold[1])})
    print(f'[T1 A] {scores}，准入 {gate}', flush=True)
    return {'A': scores, 'gate': gate}


def t1_b_predict(args, gate):
    directory = args.out / 'T1'
    roster = load_roster(args.data_root)
    chosen = gate['selected']
    train, days = chronological_dates(args.split, 'B')
    keys = forecast_keys(days, roster)
    ref, alternatives, _, details = wind_alternatives(args, train, days, roster, selected=chosen)
    pairs, locks = {}, {}
    for seed in B_SEEDS:
        print(f'[T1 B] {seed}', flush=True)
        frames, source = t1_pred(args, 'B', seed, keys, ref, alternatives, (chosen,))
        pairs[seed], locks[str(seed)] = freeze_window(directory / 'B' / str(seed), 'T1', keys,
            {'baseline': frames['baseline'], 'candidate': frames[chosen]}, {'gate': gate, 'fits': details, 'source': source})
    digest = freeze_json(directory / 'B' / 'predictions_frozen.json', locks)
    def score():
        report = paired_report(directory / 'B', 'T1', labels(args.data_root, days, roster), roster, keys, pairs, digest)
        return {'acceptance': report['acceptance'], 'seed_deltas': {k: v['delta'] for k, v in report['pairs'].items()}}

    return score


def worker_args(args):
    return ('--data-root', args.data_root, '--workdir', args.workdir, '--device', args.device,
            '--split', args.split, '--mlp-cache', args.mlp_cache, '--wind-run', args.wind_run)


def t2_mlp_worker(args):
    train, days = chronological_dates(args.split, args.phase)
    tail = max(1, len(train) // 5)
    if args.calibration:
        train, days = train[:-tail], train[-tail:]
    if args.worker == 'prepare_weather':
        power_weather(args.data_root, train, days, args.mlp_cache, 2, print)
        return
    fit_forecasts(args.data_root, args.out, train, days, 2, 16, 64, args.device,
                  cache_root=args.mlp_cache, seed=args.seed)


def mlp_curve(args, phase, seed, directory, keys, calibration, runtime):
    train, days = chronological_dates(args.split, phase)
    if calibration:
        cut = len(train) // 5
        train, days = train[:-cut], train[-cut:]
    plan = {'train': list(map(str, train)), 'forecast': list(map(str, days)), 'seed': seed,
            'device': args.device, 'recipe': 'warm_shared_power:16:64', 'calibration': calibration}
    directory.mkdir(parents=True, exist_ok=True)
    freeze_json(directory / 'plan.json', plan)
    folder = directory / 'mlp'
    if not folder.exists():
        temp = Path(mkdtemp(dir=directory, prefix='.mlp-attempt-'))
        interpreter, weather_python = runtime
        common = (*worker_args(args), '--phase', phase, '--seed', seed, '--out', temp)
        if calibration:
            common = (*common, '--calibration')
        if interpreter != weather_python:
            call(weather_python, ENTRY, *common, '--worker', 'prepare_weather')
        call(interpreter, ENTRY, *common, '--worker', 'mlp')
        file = temp / 'T2_development.csv'
        read_frozen_forecast(file, sha256(file), keys, 'T2')
        freeze_json(temp / 'receipt.json', {'forecast_sha256': sha256(file),
                                           'training_sha256': sha256(temp / 'training_receipt.json')})
        temp.rename(folder)
    record = read_json(folder / 'receipt.json')
    info = read_json(folder / 'training_receipt.json')
    if (record['training_sha256'] != sha256(folder / 'training_receipt.json')
            or info['train_days'] != len(train) or info['development_days'] != len(days)
            or info['seed'] != seed or info['epochs'] != 16 or info['device_requested'] != args.device):
        raise ValueError('T2 MLP receipt changed or does not match dates/seed.')
    return read_frozen_forecast(folder / 'T2_development.csv', record['forecast_sha256'], keys, 'T2'), record


def t2_components(args, phase, seed, directory, roster, train, days, runtime, *, calibration=False):
    keys = keys_for(days, roster, 'T2')
    spec = feature_spec(args.workdir)
    mlp, mlp_info = mlp_curve(args, phase, seed, directory, keys, calibration, runtime)
    fit, fit_info = energy_features(args, roster, train, spec)
    forecast, forecast_info = energy_features(args, roster, days, spec)
    if fit_info['columns'] != forecast_info['columns']:
        raise ValueError('T2 feature order changed.')
    _, actual = load_daily_targets(weather_day_targets(args.data_root, train), roster)
    training = fit.merge(actual, on=DAY_KEYS, validate='one_to_one')
    if len(training) != len(actual):
        raise ValueError('T2 daily matrix lost eligible rows.')
    folder = directory / 'relative_mse'
    folder.mkdir(exist_ok=True)
    stamp = {'training': fit_info['sha256'], 'forecast': forecast_info['sha256'], 'seed': seed}
    path = folder / 'fit.json'
    file = folder / 'forecast.csv'
    if not path.is_file():
        values, _, info = fit_energy(args, fit_info['columns'], training, train, forecast, 'relative_mse', folder,
                                     seed=seed)
        from batch_energy_short import save_csv
        freeze_json(path, {'fit': info, 'stamp': stamp, 'prediction_sha256': save_csv(
            file, forecast[DAY_KEYS].assign(energy=values))})
    fit = read_json(path)
    if fit['stamp'] != stamp:
        raise ValueError('Resumed T2 tree uses a different feature matrix.')
    tree = read_frozen_forecast(file, fit['prediction_sha256'], keys, 'T2')
    return keys, {'mlp': mlp, 'relative_mse': tree, 'baseline': mix(mlp, tree)}, {'MLP': mlp_info, 'relative': fit}


def station_calibration(pred, actual):
    aligned = pred.merge(actual, on=DAY_KEYS, suffixes=('_pred', '_actual'), validate='one_to_one')
    ratio = aligned.energy_pred / aligned.energy_actual
    table = aligned.assign(ratio=ratio, square=ratio**2).groupby('station_id').agg(
        days=('ratio', 'size'), num=('ratio', 'sum'), den=('square', 'sum'))
    factor = ((table.num + 20) / (table.den + 20)).clip(.6, 1.2)
    return factor.to_dict(), table.days.to_dict()


def energy_for(curve, keys):
    values = curve.groupby(DAY_KEYS, as_index=False).prediction.sum()
    return keys.merge(values.rename(columns={'prediction': 'energy'}).assign(
        energy=lambda rows: (rows.energy / 4).clip(0, MAX_PREDICTION['T2'])), on=DAY_KEYS, validate='one_to_one')


def t2_pred(args, phase, seed, roster, runtime, requested):
    train, days = chronological_dates(args.split, phase)
    target = args.out / 'T2' / phase / str(seed)
    keys, frames, info = t2_components(args, phase, seed, target / 'full', roster, train, days, runtime)
    if CALIBRATED in requested:
        size = len(train) // 5
        fit_days, cal_days = train[:-size], train[-size:]
        _, other, history = t2_components(args, phase, seed, target / 'calibration', roster, fit_days, cal_days,
                                          runtime, calibration=True)
        _, actual = load_daily_targets(weather_day_targets(args.data_root, cal_days), roster)
        factors, counts = station_calibration(other['baseline'], actual)
        frames[CALIBRATED] = frames['baseline'].assign(energy=lambda df: (
            df.energy * df.station_id.map(factors).fillna(1)).clip(0, MAX_PREDICTION['T2']))
        info['calibration'] = {'history': history, 'factors': factors, 'scored_days': counts}
    if INTEGRAL in requested:
        curve, lock = prior_baseline(args, phase, forecast_keys(days, roster), seed)
        frames[INTEGRAL] = blend(frames['baseline'], energy_for(curve, keys), .10, 'T2')
        info['T1_v2'] = lock
        if phase == 'A':
            path = args.power_run / 'T1' / phase / str(seed) / 'tree'
            receipt = read_json(path / 'training_receipt.json')
            tree = read_frozen_forecast(path / receipt['prediction_file'], receipt['prediction_sha256'],
                                        forecast_keys(days, roster), 'T1')
            frames['tree_integral'] = energy_for(tree, keys)
            frames['tree_integral_10pct'] = blend(frames['baseline'], frames['tree_integral'], .10, 'T2')
            info['T1_tree_receipt'] = sha256(path / 'training_receipt.json')
    return keys, frames, info


def t2_scores(args, days, roster, forecasts):
    truth, eligible = load_daily_targets(weather_day_targets(args.data_root, days), roster)
    report = energy_arm_report(truth, eligible, forecasts, roster)
    valid = truth.power_limit.eq(1) & np.isfinite(truth.power)
    coverage = truth.loc[valid].groupby(DAY_KEYS).size().rename('valid_points').reset_index()
    cohort = eligible.merge(coverage, on=DAY_KEYS)
    diagnostics = {}
    for arm, forecast in forecasts.items():
        rows = cohort.merge(forecast, on=DAY_KEYS, suffixes=('_true', '_pred'))
        ratio = rows.energy_pred / rows.energy_true
        groups = {'72_95': rows.valid_points.lt(96), '96': rows.valid_points.eq(96),
                  'low_24_60_mwh': rows.energy_true.lt(60), 'high_ge_180_mwh': rows.energy_true.ge(180)}
        diagnostics[arm] = {name: {'station_days': int(mask.sum()),
                                   'relative_rmse': float(np.sqrt(((ratio[mask]-1)**2).mean())) if mask.any() else None}
                            for name, mask in groups.items()}
    return report, diagnostics


def t2_stage(args):
    roster = roster_for(args.data_root, 'T2')
    if roster != load_roster(args.data_root):
        raise ValueError('T1 and T2 station rosters differ.')
    runtime = select_mlp_runtime(args)
    directory = args.out / 'T2'
    _, days = chronological_dates(args.split, 'A')
    names = (CALIBRATED, INTEGRAL)
    keys, frames, info = t2_pred(args, 'A', A_SEED, roster, runtime, names)
    frames, lock = freeze_window(directory / 'A' / 'forecasts', 'T2', keys, frames, info)
    report, diagnostics = t2_scores(args, days, roster, frames)
    scores = {k: v[PRIMARY_SCORE] for k, v in report.items()}
    chosen = max(names, key=scores.get)
    gain = scores[chosen] - scores['baseline']
    gate = {'selected': chosen, 'delta': gain, 'promoted': gain > 0, 'forecast_lock': lock}
    freeze_json(directory / 'a_score.json', {'scores': report, 'groups': diagnostics, 'gate': gate})
    print(f'[T2 A] {scores}，准入 {gate}', flush=True)
    return {'A': scores, 'gate': gate}


def t2_b_predict(args, gate):
    directory = args.out / 'T2'
    roster = roster_for(args.data_root, 'T2')
    chosen = gate['selected']
    runtime = select_mlp_runtime(args)
    _, days = chronological_dates(args.split, 'B')
    pairs, locks = {}, {}
    for seed in B_SEEDS:
        print(f'[T2 B] {seed}', flush=True)
        keys, frames, info = t2_pred(args, 'B', seed, roster, runtime, (chosen,))
        pairs[seed], locks[str(seed)] = freeze_window(directory / 'B' / str(seed) / 'forecasts', 'T2', keys,
            {'baseline': frames['baseline'], 'candidate': frames[chosen]}, {'gate': gate, **info})
    digest = freeze_json(directory / 'B' / 'predictions_frozen.json', locks)
    def score():
        truth, eligible = load_daily_targets(weather_day_targets(args.data_root, days), roster)
        report = {str(seed): energy_arm_report(truth, eligible, pairs[seed], roster) for seed in B_SEEDS}
        ensemble = energy_arm_report(truth, eligible, average_seeds(keys, pairs, 'T2'), roster)
        def gain(arms):
            return arms['candidate'][PRIMARY_SCORE] - arms['baseline'][PRIMARY_SCORE]
        deltas = {seed: gain(scores) for seed, scores in report.items()}
        acceptance = b_statistics(list(deltas.values()), gain(ensemble))
        groups = {name: identical_seed_groups({seed: arms[name].energy.to_numpy()
                                              for seed, arms in pairs.items()})
                  for name in ('baseline', 'candidate')}
        freeze_json(directory / 'B' / 'b_score.json', {'pairs': report, 'ensemble': ensemble, 'prediction_lock': digest,
                                                       'acceptance': acceptance, 'seed_deltas': deltas, 'identical_seed_groups': groups})
        print(f'[T2 B] 三种子配对 {acceptance}', flush=True)
        return {'seed_deltas': deltas, 'acceptance': acceptance}

    return score


def pseudo_heads(args):
    import torch
    from screen_same_issue_context import make_context_network, CONTEXTS
    from screen_short_transfer import adjust_short, targets

    old = args.wind_run / 'T3_pseudo'
    plan = read_json(old / 'plan.json')
    sites = plan['sites']
    donor, train, days = pseudo_dates(args)
    if (plan['score_dates'] != list(map(str, days)) or plan['target_train_dates'] != list(map(str, train))
            or plan['donor_dates'] != list(map(str, donor)) or set(sites) & set(plan['donors'])
            or plan['device'] != args.device or plan['archive'] != archive_stamp(args.workdir, [*donor, *days])):
        raise ValueError('Pseudo-target source checkpoint belongs to a different fold.')
    checkpoint = old / 'neural' / 'frozen_encoder.pt'
    locked = read_json(old / 'neural' / 'predictions_frozen.json')
    if locked['sources']['plan_sha256'] != sha256(old / 'plan.json'):
        raise ValueError('Pseudo-target checkpoint has a different plan.')
    stamp = {'checkpoint': sha256(checkpoint), 'neural': sha256(old / 'neural' / 'predictions_frozen.json')}
    model = None
    frames = {}
    keys = forecast_keys(days, sites)
    for size in (20, 35, 50):
        print(f'[T3 模拟] 迁移头 {size} 天', flush=True)
        if size == 50:
            frames[str(size)] = read_arm(old / 'neural', 'station_ridge', keys, 'T3')
            continue
        prefix = train[-size:]
        weather = archived_issue_weather(args.workdir, args.data_root, [*prefix, *days], sites, print)[0]
        norm = locked['sources']['fit']['source_normalization']
        mean, scale = np.asarray(norm['mean']), np.asarray(norm['scale'])
        present = np.isfinite(weather)
        weather = np.concatenate([np.where(present, (weather-mean)/scale, 0), ~present], axis=-1).astype(np.float32)
        cut = len(prefix) * len(sites)
        if model is None:
            model = make_context_network(weather.shape[-1], CONTEXTS['full_context']).to(args.device)
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=args.device))
            model.eval()

        def predict():
            with torch.inference_mode():
                chunks = [torch.sigmoid(model(torch.as_tensor(weather[i:i+64], device=args.device))).reshape(-1, 96)
                          .cpu().numpy() for i in range(0, len(weather), 64)]
                return np.concatenate(chunks).astype(float) * MAX_PREDICTION['T3']

        actual, valid = targets(args.data_root, prefix, sites)
        values, _ = adjust_short(model, predict, weather[:cut], weather[cut:], actual, valid, sites, args.device)
        frames[str(size)] = keys.assign(prediction=values.ravel())
    freeze_window(args.out, 'T3', keys, frames, stamp)


def pseudo_stage(args):
    old = args.wind_run / 'T3_pseudo'
    directory = args.out / 'T3_pseudo'
    directory.mkdir(exist_ok=True)
    plan = read_json(old / 'plan.json')
    sites = plan['sites']
    _, train, days = pseudo_dates(args)
    keys = forecast_keys(days, sites)
    heads = directory / 'heads'
    heads.mkdir(exist_ok=True)
    if not (heads / 'predictions_frozen.json').is_file():
        interpreter = select_mlp_runtime(args, require_weather=False)[0]
        call(interpreter, ENTRY, *worker_args(args), '--out', heads, '--worker', 'pseudo')
    lock = read_json(heads / 'predictions_frozen.json')
    expected = {'checkpoint': sha256(old / 'neural' / 'frozen_encoder.pt'),
                'neural': sha256(old / 'neural' / 'predictions_frozen.json')}
    if lock['sources'] != expected or plan['target_train_dates'] != list(map(str, train)):
        raise ValueError('Pseudo replay checkpoint or dates changed.')
    whole, (indices, spec, names, build, params) = short_source(
        args, load_roster(args.data_root) + roster_for(args.data_root, 'T3'))
    selected_indices = sorted(indices[whole.index(site)] for site in sites)
    ordered = [whole[index] for index in selected_indices]
    config = args.config or args.companion_code / 'configs/default.json'
    digest = sha256(config)
    scores = {}
    for size in (20, 35, 50):
        print(f'[T3 模拟] 短历史树 {size} 天', flush=True)
        cut = train[-size:]
        tree = old / 'tree' if size == 50 else directory / str(size)
        path = tree / 'fit.json'
        stamp = {'dates': list(map(str, cut)), 'archive': archive_stamp(args.workdir, [*cut, *days]), 'config': digest}
        if size == 50 and (read_json(path)['plan_sha256'] != sha256(old / 'plan.json') or plan['tree_config'] != digest):
            raise ValueError('The existing 50-day tree uses a different plan.')
        if not path.is_file():
            _, _, fit = fit_short_forecasts(args, tree, cut, days, ordered, ordered,
                                            (selected_indices, spec, names, build, params), modes=('short_only',))
            freeze_json(path, {'fit': fit, 'stamp': stamp})
        info = read_json(path)
        if size != 50 and info['stamp'] != stamp:
            raise ValueError('Pseudo tree cache changed.')
        baseline = read_frozen_forecast(tree / 'short_only.csv', info['fit']['predictions_sha256']['short_only'], keys, 'T3')
        transfer = read_arm(heads, str(size), keys, 'T3')
        frames, locked = freeze_window(directory / f'score_{size}', 'T3', keys,
            {'short_only': baseline, 'tree_transfer_75_25': blend(baseline, transfer, .25, 'T3')},
            {'tree': sha256(path), 'neural': sha256(heads / 'predictions_frozen.json')})
        truth = labels(args.data_root, days, sites)
        hist = labels(args.data_root, cut, sites)
        valid = hist.power_limit.eq(1) & np.isfinite(hist.power)
        coverage = valid.groupby([hist.station_id, hist.date]).sum().ge(72).groupby(level=0).sum()
        scores[str(size)] = {'scores': {name: station_scores(truth, frame) for name, frame in frames.items()},
                             'coverage': {k: int(v) for k, v in coverage.items()}, 'lock': locked}
    return scores


def confirmed_t3(args, window, train, days, roster, seed):
    keys = forecast_keys(days, roster)
    if window != 'fixed_50_10':
        transfer = args.transfer_run / window
        marker = transfer / 'predictions_frozen.json'
        record = read_json(marker)
        plan = record['plan']
        if (plan['fit_dates'] != list(map(str, train)) or plan['score_dates'] != list(map(str, days))
                or plan['target_sites'] != roster or plan['device'] != args.device or plan['seed'] != seed):
            raise ValueError('T3 early forecast uses different registered dates.')
        tree = read_frozen_forecast(args.short_tree / window / 'short_only.csv', plan['reference_sha256'], keys, 'T3')
        other = read_frozen_forecast(transfer / 'frozen_context_ridge.csv', record['forecasts']['frozen_context_ridge'], keys, 'T3')
        return keys, tree, other, blend(tree, other, .25, 'T3'), sha256(marker)
    folder = args.t3_confirm / 'T3'
    root_lock = folder / 'predictions_frozen.json'
    stored = read_json(root_lock)
    score = read_json(folder / 'score.json')
    seed_folder = folder / str(seed)
    marker = seed_folder / 'predictions_frozen.json'
    plan = read_json(folder / 'plan.json')
    if (score['active_recipe'] != 'tree_transfer_75_25' or score['prediction_lock_sha256'] != sha256(root_lock)
            or stored['plan'] != sha256(folder / 'plan.json') or stored['seed_locks'][str(seed)] != sha256(marker)
            or plan['fraction'] != .25):
        raise ValueError('T3 confirmed reference changed.')
    tree = read_arm(seed_folder, 'baseline', keys, 'T3')
    baseline = read_arm(seed_folder, 'candidate', keys, 'T3')
    # The accepted convex blend is unclipped when both inputs lie inside the same power bounds.
    transfer = tree.assign(prediction=(4 * baseline.prediction - 3 * tree.prediction).clip(0, MAX_PREDICTION['T3']))
    return keys, tree, transfer, baseline, sha256(marker)


def t3_weights(history, roster):
    frame = pd.concat(history, ignore_index=True)
    usable = frame.power_limit.eq(1) & np.isfinite(frame.power)
    frame = frame.loc[usable].copy()
    grouped = frame.groupby(DAY_KEYS)
    frame = frame.loc[grouped.power.transform('size').ge(72)].copy()
    difference = frame.transfer - frame.tree
    centered = frame.power - frame.tree - .25 * difference
    frame['numerator'] = difference * centered
    frame['denominator'] = difference**2
    per_day = frame.groupby(DAY_KEYS)[['numerator', 'denominator']].mean()
    prior = per_day.denominator.mean() if len(per_day) else 1.
    station = per_day.groupby(level='station_id').sum()
    scale = max(1e-6, 20 * prior)
    return {site: float(.25 + np.clip(station.loc[site, 'numerator'] /
                                     (scale + station.loc[site, 'denominator']), -.25, .25))
            if site in station.index else .25 for site in roster}


def t3_stage(args):
    roster = roster_for(args.data_root, 'T3')
    history, report = [], {}
    output = args.out / 'T3'
    output.mkdir(exist_ok=True)
    for window, train, days in t3_windows():
        if window == 'fixed_50_10':
            break
        keys, tree, transfer, baseline, stamp = confirmed_t3(args, window, train, days, roster, A_SEED)
        weight = t3_weights([rows for max_day, rows in history if max_day <= max(train)], roster) if history else None
        arms = {'baseline': baseline}
        if weight:
            arms['candidate'] = blend(tree, transfer, tree.station_id.map(weight), 'T3')
        arms, digest = freeze_window(output / window, 'T3', keys, arms, {'source': stamp, 'weights': weight})
        truth = labels(args.data_root, days, roster)
        station = {k: station_scores(truth, frame) for k, frame in arms.items()}
        comparable = sorted(station['baseline'])
        delta = (float(np.mean([station['candidate'][site]['score'] - station['baseline'][site]['score']
                                for site in comparable])) if weight and comparable else None)
        report[window] = {'delta': delta, 'weights': weight, 'stations': station, 'lock': digest}
        history.append((max(days), truth.merge(keys.assign(tree=tree.prediction, transfer=transfer.prediction),
                                               on=['station_id', 'date', 'point_index'], validate='one_to_one')))
    gate = {'delta': report['inner_mar']['delta'], 'promoted': (report['inner_mar']['delta'] or 0) > 0}
    freeze_json(output / 'a_score.json', {'windows': report, 'gate': gate})
    result = {'A': report, 'gate': gate}
    print(f'[T3] 权重候选准入 {gate}', flush=True)
    if not gate['promoted']:
        return result
    # Fit final weights only on early predictions from dates in the fixed fold's training prefix.
    weight = t3_weights([rows for max_day, rows in history if max_day <= max(train)], roster)
    pairs, locks = {}, {}
    for seed in B_SEEDS:
        keys, tree, transfer, baseline, stamp = confirmed_t3(args, window, train, days, roster, seed)
        pairs[seed], locks[str(seed)] = freeze_window(output / window / str(seed), 'T3', keys,
            {'baseline': baseline, 'candidate': blend(tree, transfer, tree.station_id.map(weight), 'T3')},
            {'source': stamp, 'weights': weight, 'gate': gate})
    digest = freeze_json(output / window / 'predictions_frozen.json', locks)
    result['B'] = paired_report(output / window, 'T3', labels(args.data_root, days, roster), roster, keys, pairs, digest)
    return result


def execute(args):
    code = (ENTRY, ENTRY.with_name('steamlink_native_wind.py'), ENTRY.with_name('run_companion_warm_t1.py'))
    paths = ('data_root', 'workdir', 'companion_code', 'wind_run', 'power_run', 'short_tree', 't3_confirm', 'transfer_run')
    freeze_json(args.out / 'plan.json', {'paths': {key: str(getattr(args, key).resolve()) for key in paths},
                                        'code': {path.name: sha256(path) for path in code},
                                        'split': sha256(args.split), 'threads': args.threads, 'device': args.device})
    stages = [('T1', t1_stage), ('T3_pseudo', pseudo_stage), ('T3_weights', t3_stage), ('T2', t2_stage)]
    result = {}
    for index, (name, function) in enumerate(stages):
        show_progress(args.out, 'Steamlink 三赛道', index, len(stages), name)
        try:
            result[name] = {'status': 'complete', 'result': function(args)}
        except Exception as error:
            traceback.print_exc()
            result[name] = {'status': 'failed', 'error': str(error)}
        write_json(args.out / 'summary.json', result)
    score_b = {}
    block_b = False
    for name, predict in [('T1', t1_b_predict), ('T2', t2_b_predict)]:
        current = result[name]
        if current['status'] != 'complete' or not current['result']['gate']['promoted']:
            continue
        try:
            score_b[name] = predict(args, current['result']['gate'])
        except Exception as error:
            traceback.print_exc()
            current.update(status='failed', B_error=str(error))
            block_b = True
    for name, score in score_b.items():
        current = result[name]
        if block_b:
            current.update(status='failed', B_error='A promoted B arm failed to freeze. Re-run to score both.')
            continue
        try:
            current['result']['B'] = score()
        except Exception as error:
            traceback.print_exc()
            current.update(status='failed', B_error=str(error))
    write_json(args.out / 'summary.json', result)
    show_progress(args.out, 'Steamlink 三赛道', len(stages), len(stages), '结果已写入 summary.json')
    print(f'摘要 {args.out / "summary.json"}', flush=True)
    return int(any(stage['status'] != 'complete' for stage in result.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        'out': RUNS / 'steamlink_followup_gpu_v1', 'data-root': Path('/workspace/qhl/data'),
        'workdir': Path('/workspace/cpy/code/workdir'), 'companion-code': Path('/workspace/cpy/code'),
        'split': SPLIT_PATH, 'wind-run': RUNS / 'wind_attribution_gpu_v1', 'power-run': RUNS / 't1_t3_fast_gpu_v1',
        'mlp-cache': RUNS / 'warm_power_feature_cache', 'mlp-run': RUNS / 'warm_shared_power_v1',
        'short-tree': RUNS / 'three_track_gpu_batch_v1/T3', 'transfer-run': RUNS / 'same_issue_followup_v1/T3',
        't3-confirm': RUNS / 't3_transfer_confirmation_gpu_v1',
    }
    for key, default in defaults.items():
        parser.add_argument(f'--{key}', type=Path, default=default)
    for key in ('config', 'mlp-python', 'mlp-weather-python'):
        parser.add_argument(f'--{key}', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--phase', choices=('A', 'B'), default='A')
    parser.add_argument('--seed', type=int, choices=B_SEEDS, default=A_SEED)
    parser.add_argument('--worker', choices=('mlp', 'prepare_weather', 'pseudo'))
    parser.add_argument('--calibration', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker == 'pseudo':
        pseudo_heads(args)
    elif args.worker:
        t2_mlp_worker(args)
    else:
        with (args.out / '.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(execute(args))


if __name__ == '__main__':
    main()
