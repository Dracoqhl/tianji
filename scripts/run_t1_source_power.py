#!/usr/bin/env python3
"""Test weather-source power experts against the frozen chronological T1 v2."""
import argparse
import fcntl
import hashlib
from inspect import getsourcefile
from itertools import combinations
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import pandas as pd

from advance_batch_b import read_frozen_forecast
from batch_energy_short import save_csv
from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import assert_matched, read_json
from run_companion_warm_t1 import (
    A_SEED, B_SEEDS, EARLY_STOP_DAYS, SPLIT_PATH, cache_paths, cache_version, check_training_sample,
    chronological_dates, forecast_keys, row_positions, sha256, training_targets, weather_rows, write_json,
)
from run_forecast_blends import blend, freeze_json, freeze_window
from run_t1_t3_batch import paired_report
from run_t1_wind_aux import prior_baseline
from score_forecasts import CAPACITY_MW, MAX_PREDICTION, MIN_VALID_POINTS, POINT_KEYS
from screen_companion_t1_stations import block_comparisons, show_progress
from tree_model import best_iteration, check_backend, fitted_device, predict_tree, save_tree_model, train_tree
from warm_shared_power import labels, load_roster

RUNS = Path('/workspace/qhl/runs')
SOURCES = ('tianji', 'ec', 'aifs')
COMPACT = 'compact_all_sources'
EQUAL = 'equal_source_blend'
SHRUNK = 'shrunk_lead_blend'
CONTROL = 'compact_all_sources_blend'
FRACTION = .20
RIDGE_MW2 = 1.5
SCORE_TOLERANCE = 1e-8
PHYSICAL_ORDER = ('u100', 'v100', 'ws100', 't2m', 'wpd', 'sp', 'u10', 'v10', 'ws10')


def feature_views(names, nwp_names):
    """Keep common inputs intact and use the smallest source's NWP column budget."""
    if not set(nwp_names).issubset(names) or len(set(names)) != len(names):
        raise ValueError('Invalid companion feature schema.')
    grouped = {s: [key for key in nwp_names if key.startswith(f'{s}_')] for s in SOURCES}
    minimum = min(map(len, grouped.values()))
    if minimum < len(SOURCES) or sum(map(len, grouped.values())) != len(nwp_names):
        raise ValueError('The cached NWP schema must contain exactly the three named sources.')
    rank = {suffix: index for index, suffix in enumerate(PHYSICAL_ORDER)}
    ranked = []
    for source in SOURCES:
        ranked.append(sorted(grouped[source], key=lambda key: (rank.get(key.split('_', 1)[1], len(rank)), key)))
    # Round-robin selection provides all three sources at each family of fields until the budget is filled.
    budget = [fields[i] for i in range(minimum) for fields in ranked][:minimum]
    common = set(names).difference(nwp_names)
    selected = {source: fields[:minimum] for source, fields in zip(SOURCES, ranked)}
    selected[COMPACT] = budget
    views = {arm: [key for key in names if key in common or key in selected[arm]] for arm in selected}
    if not all('station_idx' in view for view in views.values()):
        raise ValueError('The weather views lost the station category.')
    return views


def simplex_weights(x, y, weights):
    """Solve the three-expert ridge problem over all simplex faces."""
    hessian = x.T @ (x * weights[:, None]) + RIDGE_MW2 * np.eye(len(SOURCES))
    rhs = x.T @ (weights * y) + RIDGE_MW2 / len(SOURCES)
    best_loss, best = np.inf, None
    for size in range(1, len(SOURCES) + 1):
        for active in combinations(range(len(SOURCES)), size):
            ids = list(active)
            ones = np.ones(size)
            sub = hessian[np.ix_(ids, ids)]
            solve = np.linalg.solve(sub, np.column_stack((rhs[ids], ones)))
            base, normal = solve.T
            candidate = base - normal * ((base.sum() - 1) / normal.sum())
            if candidate.min() < -1e-12:
                continue
            trial = np.zeros(len(SOURCES))
            trial[ids] = np.maximum(candidate, 0)
            trial /= trial.sum()
            loss = trial @ hessian @ trial - 2 * rhs @ trial
            if loss < best_loss:
                best_loss, best = loss, trial
    if best is None:
        raise ValueError('Could not solve source blend weights.')
    return best


def source_weights(truth, keys, arms):
    target = keys.merge(truth[[*POINT_KEYS, 'power', 'power_limit']], on=POINT_KEYS, validate='one_to_one')
    use = target.power_limit.eq(1) & np.isfinite(target.power)
    station_day = [target.station_id, target.date]
    use &= use.groupby(station_day).transform('sum').ge(MIN_VALID_POINTS)
    x = np.column_stack([arms[s].prediction.to_numpy() for s in SOURCES])
    result = {}
    for lead in range(0, 96, 24):
        mask = use & target.point_index.between(lead + 1, lead + 24)
        rows = target.loc[mask]
        if not len(rows):
            result[str(lead)] = {'weights': [1/3] * 3, 'station_days': 0, 'points': 0}
            continue
        day_count = rows.groupby('station_id').date.transform('nunique').to_numpy()
        point_count = rows.groupby(['station_id', 'date']).power.transform('size').to_numpy()
        weight = 1 / (day_count * point_count)
        trained = simplex_weights(x[mask], rows.power.to_numpy(), weight / weight.sum())
        result[str(lead)] = {'weights': trained.tolist(), 'station_days': int(rows.groupby(['station_id', 'date']).ngroups),
                             'points': len(rows)}
    return result


def average_sources(keys, arms, weights=None):
    block = np.column_stack([arms[s].prediction.to_numpy() for s in SOURCES])
    if weights is None:
        values = block.mean(axis=1)
    else:
        groups = (keys.point_index.to_numpy() - 1) // 24
        simplex = np.asarray([weights[str(lead)]['weights'] for lead in range(0, 96, 24)])
        values = np.sum(block * simplex[groups], axis=1)
    return keys.assign(prediction=values)


def residual_correlations(truth, keys, forecasts):
    actual = keys.merge(truth, on=POINT_KEYS, validate='one_to_one')
    valid = actual.power_limit.eq(1) & np.isfinite(actual.power)
    valid &= valid.groupby([actual.station_id, actual.date]).transform('sum').ge(MIN_VALID_POINTS)
    errors = pd.DataFrame({name: forecasts[name].prediction - actual.power for name in (*SOURCES, COMPACT, 'baseline')})
    bands = {'all': valid, **{str(i): valid & actual.point_index.between(i, i + 23) for i in (1, 25, 49, 73)}}
    report = {}
    for band, mask in bands.items():
        corr = errors.loc[mask].corr()
        report[band] = {'points': int(mask.sum()), 'mae_mw': errors.loc[mask].abs().mean().dropna().to_dict(),
                        'error_correlation': corr.astype(object).where(corr.notna(), None).to_dict()}
    return report


def companion(args):
    sys.path.insert(0, str(args.companion_code / 'src'))
    from tianji.config import load_config
    from tianji.constants import ALL_STATIONS, REGULAR_IDX
    from tianji.data.dataset import build_gbdt_rows, gbdt_feature_names, load_feature_spec

    spec = load_feature_spec(args.workdir, 'train')
    config = args.config or args.companion_code / 'configs/default.json'
    params = dict(load_config(str(config))['gbdt'])
    if params.get('objective', 'huber') != 'huber':
        raise ValueError('Expected the companion Huber recipe.')
    stations = list(REGULAR_IDX)
    roster = [ALL_STATIONS[index] for index in stations]
    if set(roster) != set(load_roster(args.data_root)):
        raise ValueError('T1 feature roster differs from submission roster.')
    names = list(gbdt_feature_names(spec))
    source_files = {'config': config, 'feature_spec': args.workdir / 'stats/feature_spec_train.json'}
    source_files.update({fn.__name__: Path(getsourcefile(fn))
                         for fn in (load_config, build_gbdt_rows, gbdt_feature_names)})
    return roster, stations, spec, names, build_gbdt_rows, params, {k: sha256(v) for k, v in source_files.items()}


def fit_one(args, seed, inputs, columns, target, inner, forecast, params, directory):
    params = dict(params, random_state=seed, n_jobs=args.threads)
    options = {'categorical_features': ['station_idx'], 'backend': 'xgboost', 'device': args.device,
               'early_stopping_rounds': int(params.get('early_stopping_rounds', 100))}
    start = monotonic()
    model = train_tree(inputs[~inner], target[~inner], columns, params, X_valid=inputs[inner],
                       y_valid=target[inner], **options)
    best = best_iteration(model, backend='xgboost', fallback=params['n_estimators'])
    del model
    print(f'重训 {best} 棵树，seed={seed}', flush=True)
    final = train_tree(inputs, target, columns, dict(params, n_estimators=best), **options)
    save_tree_model(final, directory / 'model.joblib')
    prediction = predict_tree(final, forecast, backend='xgboost').astype(float) * CAPACITY_MW
    return np.clip(prediction, 0, MAX_PREDICTION['T1']), {
        'best_iteration': best, 'effective_device': fitted_device(final.booster), 'seconds': monotonic()-start,
    }


def frozen_arms(directory, keys, signature, schema):
    forecasts = {}
    receipts = {}
    for name in schema:
        folder = directory / name
        marker = folder / 'fit.json'
        if not marker.is_file():
            continue
        info = read_json(marker)
        if info['signature'] != signature or info['columns'] != schema[name]:
            raise ValueError(f'{marker}: registered fit or feature schema changed.')
        forecasts[name] = read_frozen_forecast(folder / 'forecast.csv', info['forecast_sha256'], keys, 'T1')
        receipts[name] = sha256(marker)
    return forecasts, receipts


def arm_forecasts(args, inputs, train, days, seed, directory, arms, *, reference_phase=None):
    roster, stations, spec, names, build, params, sources = inputs
    keys = forecast_keys(days, roster)
    paths, wanted = cache_paths(args.workdir, train), cache_paths(args.workdir, days)
    versions = {'training': cache_version(paths), 'prediction': cache_version(wanted)}
    schema = feature_views(names, spec['nwp_names'])
    schema = {arm: schema[arm] for arm in arms}
    signature = {'seed': seed, 'threads': args.threads, 'device': args.device, 'cache': versions, 'sources': sources,
                 'training': list(map(str, train)), 'forecast': list(map(str, days))}
    frames, receipts = frozen_arms(directory, keys, signature, schema)
    if len(frames) == len(arms):
        return keys, frames, receipts

    context, _, dtype = check_training_sample(paths, train, stations, roster, spec, names, build, args.data_root)
    history, _, metadata = build(paths, stations, CAPACITY_MW, include_invalid=True, verbose=True)
    positions, row_days, _ = row_positions(metadata, stations, len(train), phase='train', feature_rows=len(history))
    del metadata
    target, valid = training_targets(args.data_root, train, roster, positions, 'train')
    masked_positions = positions[valid]
    digest = hashlib.sha256(masked_positions.tobytes() + target.tobytes()).hexdigest()
    if reference_phase:
        prior = args.power_run / 'T1' / reference_phase / str(seed) / 'tree' / 'training_receipt.json'
        reference = read_json(prior)
        assert_matched(reference, {'seed': seed, 'chronological_dates': {
            'training': signature['training'], 'forecast': signature['forecast']},
            'companion_cache_metadata_sha256': versions, 'valid_training_sample_sha256': digest,
            'feature_names': names, 'feature_spec_sha256': sources['feature_spec'],
            'config_sha256': sources['config'], 'device': args.device, 'threads': args.threads,
            'early_stopping_days_inside_train': EARLY_STOP_DAYS, 'backend': 'xgboost',
            'companion_source_sha256': {key: sources[key] for key in
                                       ('load_config', 'build_gbdt_rows', 'gbdt_feature_names')}},
            ('seed', 'chronological_dates', 'companion_cache_metadata_sha256', 'valid_training_sample_sha256',
             'feature_names', 'feature_spec_sha256', 'config_sha256', 'device', 'threads',
             'early_stopping_days_inside_train', 'backend', 'companion_source_sha256'))
    inner = row_days[valid] >= len(train) - EARLY_STOP_DAYS
    if not inner.any() or inner.all():
        raise ValueError('Missing fit or early-stop samples.')
    forecast = weather_rows(wanted, stations, roster, spec, names, context, dtype)
    for arm in arms:
        if arm in frames:
            continue
        folder = directory / arm
        folder.mkdir(parents=True, exist_ok=True)
        columns = schema[arm]
        index = [names.index(name) for name in columns]
        print(f'[{directory.name}] {arm}，{len(columns)} 列，seed={seed}', flush=True)
        matrix = history[np.ix_(valid, index)].astype(np.float32, copy=False)
        values, info = fit_one(args, seed, matrix, columns, target, inner, forecast[:, index], params, folder)
        del matrix
        if versions != {'training': cache_version(paths), 'prediction': cache_version(wanted)}:
            raise ValueError('Companion weather cache changed during this experiment.')
        frame = keys.assign(prediction=values)
        hash_ = save_csv(folder / 'forecast.csv', frame)
        info.update(signature=signature, columns=columns, valid_sample_sha256=digest, forecast_sha256=hash_)
        receipts[arm] = freeze_json(folder / 'fit.json', info)
        frames[arm] = read_frozen_forecast(folder / 'forecast.csv', hash_, keys, 'T1')
    return keys, frames, receipts


def a_comparison(args, inputs):
    directory = args.out / 'A'
    roster = inputs[0]
    train, days = chronological_dates(args.split, 'A')
    progress = 'T1 分源 A'
    show_progress(args.out, progress, 0, 3, '三源和同列数对照')
    keys, arms, receipts = arm_forecasts(args, inputs, train, days, A_SEED, directory / 'fits',
                                          (*SOURCES, COMPACT), reference_phase='A')
    baseline, locked = prior_baseline(args, 'A', keys, A_SEED)
    show_progress(args.out, progress, 1, 3, '从训练内部预测估计时效权重')
    prefix, calibration = train[:-90], train[-90:]
    ck, cf, source = arm_forecasts(args, inputs, prefix, calibration, A_SEED, directory / 'inner', SOURCES)
    cf, cal_lock = freeze_window(directory / 'inner_forecasts', 'T1', ck, cf, {'receipts': source})
    weights = source_weights(labels(args.data_root, calibration, roster), ck, cf)
    weights_digest = freeze_json(directory / 'weights.json',
                                 {'weights': weights, 'forecast_lock': cal_lock,
                                  'calibration': list(map(str, calibration)), 'ridge_mw2': RIDGE_MW2})
    predicted = {'baseline': baseline, **arms,
                 EQUAL: blend(baseline, average_sources(keys, arms), FRACTION, 'T1'),
                 SHRUNK: blend(baseline, average_sources(keys, arms, weights), FRACTION, 'T1'),
                 CONTROL: blend(baseline, arms[COMPACT], FRACTION, 'T1')}
    predicted, a_lock = freeze_window(directory / 'forecasts', 'T1', keys, predicted,
                                    {'base': locked, 'receipts': receipts, 'weights': weights_digest})
    truth = labels(args.data_root, days, roster)
    compare = block_comparisons(truth, predicted, roster, tuple(arm for arm in predicted if arm != 'baseline'))
    scores = {arm: row[PRIMARY_SCORE] for arm, row in compare['scores'].items()}
    chosen = SHRUNK if scores[SHRUNK] - scores[EQUAL] > SCORE_TOLERANCE else EQUAL
    advantage = scores[chosen] - scores['baseline']
    compact_advantage = scores[chosen] - scores[CONTROL]
    gate = {'chosen': chosen, 'delta': advantage, 'delta_vs_compact': compact_advantage,
            'promoted': advantage > SCORE_TOLERANCE and compact_advantage > SCORE_TOLERANCE,
            'prediction_lock': a_lock}
    months = {}
    for month in sorted(set(keys.date.str[:7])):
        pred = {name: frame.loc[frame.date.str[:7].eq(month)].reset_index(drop=True)
                for name, frame in predicted.items()}
        months[month] = block_comparisons(truth.loc[truth.date.str[:7].eq(month)], pred, roster,
                                          (EQUAL, SHRUNK, CONTROL))
    freeze_json(directory / 'score.json', {'comparison': compare, 'monthly': months, 'gate': gate,
                                          'residual_correlation': residual_correlations(truth, keys, predicted)})
    show_progress(args.out, progress, 3, 3, f'入围 {chosen}，增量 {advantage:+.6f}')
    print(f'[T1 A] {scores}，准入 {gate}', flush=True)
    return {'scores': scores, 'gate': gate, 'weights': weights}


def b_comparison(args, inputs, gate):
    directory = args.out / 'B'
    roster = inputs[0]
    train, days = chronological_dates(args.split, 'B')
    a_train, a_days = chronological_dates(args.split, 'A')
    if train != a_train + a_days:
        raise ValueError('B training must include all of chronological A.')
    weights, weight_sha = None, None
    if gate['chosen'] == SHRUNK:
        ck = forecast_keys(a_days, roster)
        stored = args.out / 'A' / 'forecasts' / 'predictions_frozen.json'
        if sha256(stored) != gate['prediction_lock']:
            raise ValueError('A source predictions changed before B calibration.')
        locks = read_json(stored)['hashes']
        frames = {source: read_frozen_forecast(stored.parent / f'{source}.csv', locks[source], ck, 'T1')
                  for source in SOURCES}
        weights = source_weights(labels(args.data_root, a_days, roster), ck, frames)
        directory.mkdir(parents=True, exist_ok=True)
        weight_sha = freeze_json(directory / 'weights.json', {'weights': weights, 'A_lock': sha256(stored),
                                                             'ridge_mw2': RIDGE_MW2})
    pairs, locks = {}, {}
    for i, seed in enumerate(B_SEEDS):
        show_progress(args.out, 'T1 分源 B', i, len(B_SEEDS), f'seed={seed} 分源预测')
        folder = directory / str(seed)
        keys, arms, receipts = arm_forecasts(args, inputs, train, days, seed, folder / 'fits', SOURCES,
                                             reference_phase='B')
        baseline, digest = prior_baseline(args, 'B', keys, seed)
        predictions = {'baseline': baseline,
                       'candidate': blend(baseline, average_sources(keys, arms, weights), FRACTION, 'T1')}
        pairs[seed], locks[str(seed)] = freeze_window(folder / 'pair', 'T1', keys, predictions,
            {'base': digest, 'gate': gate, 'weights': weight_sha, 'receipts': receipts})
    digest = freeze_json(directory / 'predictions_frozen.json', locks)
    report = paired_report(directory, 'T1', labels(args.data_root, days, roster), roster, keys, pairs, digest)
    show_progress(args.out, 'T1 分源 B', len(B_SEEDS), len(B_SEEDS), '配对结果已保存')
    return {'acceptance': report['acceptance'], 'seed_deltas': {key: value['delta'] for key, value in report['pairs'].items()},
            'scores': {arm: value[PRIMARY_SCORE] for arm, value in report['ensemble']['scores'].items()},
            'identical_seed_groups': report['identical_seed_groups'], 'weights': weights}


def execute(args):
    check_backend('xgboost', args.device)
    signature = {'script': sha256(Path(__file__)), 'split': sha256(args.split), 'threads': args.threads,
                 'device': args.device, 'seeds': list(B_SEEDS), 'fraction': FRACTION, 'ridge_mw2': RIDGE_MW2,
                 'paths': {name: str(getattr(args, name).resolve()) for name in
                           ('data_root', 'wind_run', 'power_run', 'workdir', 'companion_code')}}
    freeze_json(args.out / 'plan.json', signature)
    inputs = companion(args)
    try:
        a = a_comparison(args, inputs)
    except Exception as error:
        write_json(args.out / 'summary.json', {'status': 'A_failed', 'error': str(error)})
        raise
    summary = {'status': 'A_complete', 'A': a}
    write_json(args.out / 'summary.json', summary)
    if a['gate']['promoted']:
        try:
            summary['B'] = b_comparison(args, inputs, a['gate'])
        except Exception as error:
            write_json(args.out / 'summary.json', {**summary, 'status': 'B_failed', 'error': str(error)})
            raise
    summary['status'] = 'complete'
    write_json(args.out / 'summary.json', summary)
    print(f'摘要 {args.out / "summary.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RUNS / 't1_source_power_gpu_v1')
    parser.add_argument('--wind-run', type=Path, default=RUNS / 'wind_attribution_gpu_v1')
    parser.add_argument('--power-run', type=Path, default=RUNS / 't1_t3_fast_gpu_v1')
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)


if __name__ == '__main__':
    main()
