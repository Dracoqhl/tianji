#!/usr/bin/env python3
"""Evaluate nearby long-history donor rows in the T3 short-history tree."""
import argparse
from datetime import timedelta
import fcntl
import hashlib
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic
import traceback

import numpy as np
import pandas as pd

from advance_batch_b import read_frozen_forecast
from batch_energy_short import roster_for, t3_windows
from compare_warm_t1 import PRIMARY_SCORE
from confirm_working_baselines import average_seeds
from experiment_companion_t1 import assert_matched, read_json
from run_companion_warm_t1 import (A_SEED, B_SEEDS, CAPACITY_MW, SPLIT_PATH, block_early_stop_days, cache_paths,
                                   cache_version, check_training_sample, forecast_keys, sha256, weather_rows, write_json)
from run_forecast_blends import freeze_json, freeze_window
from run_fusion_mechanisms import T3_SHAPE, t3_components, t3_shape_blend
from run_pressure_profile import t3_fit_contract, t3_stage
from score_forecasts import MAX_PREDICTION, score_track
from screen_companion_t1_stations import station_scores
from tree_model import best_iteration, fitted_device, predict_tree, save_tree_model, train_tree
from warm_shared_power import labels, load_roster


RUNS = Path('/workspace/qhl/runs')
ALPHA = .10
DONOR_DAYS = 180
NEIGHBORS = 3
SHAPE = 'shape_37_5'


def donor_plan(coords, short, regular):
    if len(regular) < NEIGHBORS or set(short) & set(regular):
        raise ValueError('Donor and target rosters must be disjoint and have enough candidates.')
    lat, lon = np.radians(coords.loc[regular, ['latitude', 'longitude']].to_numpy(dtype=float)).T
    target = np.radians(coords.loc[short, ['latitude', 'longitude']].to_numpy(dtype=float))
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or not np.isfinite(target).all():
        raise ValueError('Invalid donor or target coordinates.')
    prior, links = {}, {}
    for site, (target_lat, target_lon) in zip(short, target):
        arc = np.sin((lat - target_lat) / 2)**2 + np.cos(lat) * np.cos(target_lat) * np.sin((lon - target_lon) / 2)**2
        distance = 2 * 6371 * np.arcsin(np.sqrt(np.clip(arc, 0, 1)))
        chosen = np.argsort(distance, kind='stable')[:NEIGHBORS]
        mass = np.maximum(distance[chosen], 1.0)**-2
        links[site] = []
        for i, weight in zip(chosen, mass / mass.sum()):
            donor = regular[i]
            prior[donor] = prior.get(donor, 0.) + float(weight) / len(short)
            links[site].append({'station': donor, 'distance_km': float(distance[i]), 'prior': float(weight)})
    return prior, links


def donor_dates(cutoff, available):
    first = cutoff - timedelta(days=DONOR_DAYS - 1)
    return [day for day in available if first <= day <= cutoff]


def row_weights(n_targets, donor_sites, prior):
    if n_targets <= 0 or not len(donor_sites):
        raise ValueError('Target and donor fits both require eligible rows.')
    sites, reverse, counts = np.unique(donor_sites, return_inverse=True, return_counts=True)
    raw = np.asarray([prior[site] for site in sites]) / counts
    ratio = raw[reverse]
    donor_weights = ratio / ratio.sum() * (n_targets * ALPHA / (1 + ALPHA))
    target_weights = np.full(n_targets, 1 / (1 + ALPHA))
    weights = np.concatenate((target_weights, donor_weights))
    if not np.isfinite(weights).all() or (weights <= 0).any() or not np.isclose(weights.sum(), n_targets):
        raise ValueError('Donor weights violate the registered mass contract.')
    return weights, {'target_rows': n_targets, 'donor_rows': len(donor_sites),
                     'target_weight': float(target_weights.sum()), 'donor_weight': float(donor_weights.sum()),
                     'source_fraction': float(donor_weights.sum() / weights.sum()),
                     'by_donor': {str(site): {'rows': int(count), 'mass': float(count * weight)}
                                  for site, count, weight in zip(sites, counts, raw / ratio.sum() * donor_weights.sum())},
                     'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest()}


def v3_reference(args, window, train, days, short, seed):
    keys, tree, ridge, v2, ridge_source = t3_components(args, window, train, days, short, seed)
    root = args.fusion_run / 'T3'
    if window == 'fixed_50_10':
        folder = root / 'B' / str(seed)
        root_lock = root / 'B/predictions_frozen.json'
        report = read_json(root / 'B/b_score.json')
        if (not report['acceptance']['accepted'] or report['prediction_lock_sha256'] != sha256(root_lock)
                or read_json(folder / 'predictions_frozen.json')['sources'] != {
                    'source': ridge_source, 'gate': read_json(root / 'gate.json')}):
            raise ValueError('Frozen T3 v3 fixed-window source changed.')
        expected = read_json(root_lock)[str(seed)]
        name = 'candidate'
    else:
        folder = root / window / 'forecasts'
        expected = read_json(root / window / 'score.json')['lock']
        if read_json(folder / 'predictions_frozen.json')['sources'] != ridge_source:
            raise ValueError('Frozen T3 v3 early-window inputs changed.')
        name = SHAPE
    gate = read_json(root / 'gate.json')
    lock = sha256(folder / 'predictions_frozen.json')
    if gate['selected'] != SHAPE or not gate['promoted'] or lock != expected:
        raise ValueError('Wrong locked T3 v3 baseline.')
    v3 = read_frozen_forecast(folder / f'{name}.csv', read_json(folder / 'predictions_frozen.json')['hashes'][name],
                              keys, 'T3')
    rebuilt = t3_shape_blend(tree, ridge, T3_SHAPE[SHAPE], base=v2)
    if not np.allclose(v3.prediction, rebuilt.prediction, atol=1e-6, rtol=0):
        raise ValueError('Rebuilding the v3 baseline changed its predictions.')
    return keys, tree, ridge, v3, lock


def valid_labels(root, dates, sites):
    frame = labels(root, dates, sites)
    return frame.loc[frame.power_limit.eq(1) & np.isfinite(frame.power)].copy()


def arrays_for_window(args, train, days, short, roster, source, stages, prior, old):
    stations, spec, names, build_rows, _ = source
    train_paths = cache_paths(args.workdir, train)
    test_paths = cache_paths(args.workdir, days)
    context, sample, dtype = check_training_sample(train_paths, train, stations, roster, spec, names, build_rows,
                                                    args.data_root)
    assert_matched(old, {'sample_keys': sample,
                         'cache': {'train': cache_version(train_paths), 'forecast': cache_version(test_paths)}},
                   ('sample_keys', 'cache'))
    target_slots = [i for i, site in enumerate(roster) if site in short]
    donor_slots = [i for i, site in enumerate(roster) if site in prior]
    donor_roster = [roster[i] for i in donor_slots]
    target_roster = [roster[i] for i in target_slots]

    def features(paths, slots, sites):
        return weather_rows(paths, [stations[i] for i in slots], sites, spec, names,
                            {**context, 'station_idx': context['station_idx'][slots]}, dtype)

    target = valid_labels(args.data_root, train, target_roster)
    target_x = features(train_paths, target_slots, target_roster)[target.index.to_numpy()]
    donor_days = sorted(set().union(*(value['days'] for value in stages.values())))
    donor = valid_labels(args.data_root, donor_days, donor_roster)
    donor_x = features(cache_paths(args.workdir, donor_days), donor_slots, donor_roster)[donor.index.to_numpy()]
    forecast = features(test_paths, target_slots, target_roster).reshape(len(days), len(short), 96, len(names))
    forecast = forecast[:, [target_roster.index(site) for site in short]].reshape(-1, len(names))
    stopping = target.date.ge(stages['refit']['stop_start']).to_numpy()
    target_y = target.power.to_numpy(dtype=float) / CAPACITY_MW
    assert_matched(old['fit']['short_only'], {'fit_points': len(target), 'target_points': len(target),
                                            'early_stop_points': int(stopping.sum())},
                   ('fit_points', 'target_points', 'early_stop_points'))
    return target_x, target_y, stopping, donor_x, donor, forecast


def fit_candidate(args, staging, arrays, prior, stages, names, params):
    target_x, target_y, stopping, donor_x, donor, forecast = arrays
    donor_y = donor.power.to_numpy(dtype=float) / CAPACITY_MW
    audit = {}

    def fit(stage, target_mask, config, **validation):
        donor_mask = donor.date.isin([str(day) for day in stages[stage]['days']]).to_numpy()
        weights, audit[stage] = row_weights(int(target_mask.sum()), donor.loc[donor_mask, 'station_id'].to_numpy(), prior)
        matrix = np.concatenate((target_x[target_mask], donor_x[donor_mask]))
        target = np.concatenate((target_y[target_mask], donor_y[donor_mask]))
        print(f'[T3 近邻 {stage}] 目标 {audit[stage]["target_rows"]} 点，来源 {audit[stage]["donor_rows"]} 点', flush=True)
        return train_tree(matrix, target, names, config, categorical_features=['station_idx'], sample_weight=weights,
                          early_stopping_rounds=int(params.get('early_stopping_rounds', 100)), backend='xgboost',
                          device=args.device, **validation)

    start = monotonic()
    selected = fit('selection', ~stopping, params, X_valid=target_x[stopping], y_valid=target_y[stopping])
    best = best_iteration(selected, backend='xgboost', fallback=params['n_estimators'])
    del selected
    final = fit('refit', np.ones(len(target_y), dtype=bool), dict(params, n_estimators=best))
    prediction = predict_tree(final, forecast, backend='xgboost').astype(float) * CAPACITY_MW
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite T3 donor model predictions.')
    save_tree_model(final, staging / 'nearby.joblib')
    audit.update(best_iteration=best, effective_device=fitted_device(final.booster), fit_seconds=monotonic() - start,
                 target_sha256=hashlib.sha256(target_y.tobytes()).hexdigest(),
                 model_sha256=sha256(staging / 'nearby.joblib'))
    return np.clip(prediction, 0, MAX_PREDICTION['T3']), audit


class NearbyPair:
    def __init__(self, args):
        self.args = args
        self.window = None
        self.arrays = None
        self.config = sha256(args.config or args.companion_code / 'configs/default.json')
        self.static = args.data_root / 'static/power_stations.csv'
        self.coords = pd.read_csv(self.static, dtype={'station_id': str}).set_index('station_id', verify_integrity=True)
        self.regular = load_roster(args.data_root)
        self.available = sorted(pd.to_datetime(pd.read_csv(args.split, usecols=['date']).date).dt.date.unique())
        expected = read_json(args.short_tree / 'results.json')['settings']
        assert_matched(expected, {'data_root': str(args.data_root.resolve()), 'split_sha256': sha256(args.split),
                                  'device': args.device, 'threads': args.threads, 'companion_config_sha256': self.config},
                       ('data_root', 'split_sha256', 'device', 'threads', 'companion_config_sha256'))

    def __call__(self, args, window, train, days, short, roster, source, seed):
        if max(train) >= min(days):
            raise ValueError('T3 donor fitting must be earlier than the scored dates.')
        old, contract = t3_fit_contract(args, window, seed)
        keys, tree, ridge, baseline, reference = v3_reference(args, window, train, days, short, seed)
        prior, links = donor_plan(self.coords, short, [site for site in roster if site in self.regular])
        stop_index = len(train) - block_early_stop_days(train)
        if stop_index <= 0 or old['inner_stop_start'] != str(train[stop_index]):
            raise ValueError('T3 early stopping is inconsistent with the short-tree baseline.')
        stages = {}
        for stage, cutoff in [('selection', train[stop_index - 1]), ('refit', train[-1])]:
            dates = donor_dates(cutoff, self.available)
            kept = [day for day in dates if (args.workdir / 'features/train' / f'{day:%Y%m%d}_0000.npz').is_file()]
            if not kept:
                raise ValueError(f'No donor cache days available for {stage}.')
            stages[stage] = {'cutoff': str(cutoff), 'days': kept, 'stop_start': str(train[stop_index]),
                             'missing_cache_days': [str(day) for day in dates if day not in kept]}
        stage_plan = {name: {**recipe, 'days': list(map(str, recipe['days'])),
                             'weather': cache_version(cache_paths(args.workdir, recipe['days']))}
                      for name, recipe in stages.items()}
        # Include all fit power files in the receipt, while leaving evaluation power files closed.
        label_dates = sorted(set(train) | set(stages['selection']['days']) | set(stages['refit']['days']))
        labels_digest = hashlib.sha256(''.join(sha256(args.data_root / 'train' / f'{d:%Y%m%d}_0000/power_target.csv')
                                               for d in label_dates).encode()).hexdigest()
        signature = {'reference': reference, 'short_tree_fit': contract, 'seed': seed, 'config': self.config,
                     'weather': old['cache'], 'label_sha256': labels_digest, 'stages': stage_plan,
                     'prior': prior, 'neighbors': links, 'alpha': ALPHA, 'features': source[2]}
        target = args.out / 'T3' / window / str(seed)
        target.mkdir(parents=True, exist_ok=True)
        folder = target / 'tree'
        marker = folder / 'fit.json'
        if not marker.is_file():
            if self.window != window:
                self.arrays = None
                self.arrays = arrays_for_window(args, train, days, short, roster, source, stages, prior, old)
                self.window = window
            else:
                stop = self.arrays[2]
                assert_matched(old['fit']['short_only'], {'fit_points': len(stop), 'target_points': len(stop),
                                                        'early_stop_points': int(stop.sum())},
                               ('fit_points', 'target_points', 'early_stop_points'))
            staging = Path(mkdtemp(dir=target, prefix='.nearby-attempt-'))
            values, audit = fit_candidate(args, staging, self.arrays, prior, stages, source[2],
                                          dict(source[-1], random_state=seed))
            file = staging / 'nearby.csv'
            keys.assign(prediction=values).to_csv(file, index=False)
            freeze_json(staging / 'fit.json', {'signature': signature, 'audit': audit, 'forecast': sha256(file)})
            staging.rename(folder)
        record = read_json(marker)
        if record['signature'] != signature:
            raise ValueError('Existing T3 donor model was fitted against different inputs.')
        candidate = read_frozen_forecast(folder / 'nearby.csv', record['forecast'], keys, 'T3')
        frames, lock = freeze_window(target / 'forecasts', 'T3', keys, {
            'baseline': baseline, 'candidate': t3_shape_blend(candidate, ridge, T3_SHAPE[SHAPE]),
            'short_tree': tree, 'nearby_tree': candidate}, {'v3_reference': reference, 'tree_fit': sha256(marker)})
        return keys, frames, lock


def tree_diagnostics(args):
    window, _, days = list(t3_windows())[-1]
    sites = roster_for(args.data_root, 'T3')
    keys = forecast_keys(days, sites)
    root = args.out / 'T3' / window
    locks = read_json(root / 'predictions_frozen.json')
    forecasts = {}
    for seed in B_SEEDS:
        folder = root / str(seed)
        pair = folder / 'pair/predictions_frozen.json'
        manifest = folder / 'forecasts/predictions_frozen.json'
        if sha256(pair) != locks[str(seed)] or sha256(manifest) != read_json(pair)['sources']['forecast_lock']:
            raise ValueError('T3 tree diagnostic inputs do not match the paired B lock.')
        hashes = read_json(manifest)['hashes']
        forecasts[seed] = {slot: read_frozen_forecast(folder / 'forecasts' / f'{arm}.csv', hashes[arm], keys, 'T3')
                           for slot, arm in [('baseline', 'short_tree'), ('candidate', 'nearby_tree')]}
    truth = labels(args.data_root, days, sites)
    ensemble = average_seeds(keys, forecasts, 'T3')
    result = {arm: {'score': score_track('T3', truth, ensemble[slot], sites, allow_unscored_stations=True)[PRIMARY_SCORE],
                    'by_station': station_scores(truth, ensemble[slot])}
              for slot, arm in [('baseline', 'short_tree'), ('candidate', 'nearby_tree')]}
    freeze_json(root / 'tree_diagnostic.json', {'lock': sha256(root / 'predictions_frozen.json'), 'scores': result})
    return {arm: row['score'] for arm, row in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out': RUNS / 't3_nearby_donors_gpu_v1', 'data-root': Path('/workspace/qhl/data'),
                'workdir': Path('/workspace/cpy/code/workdir'), 'companion-code': Path('/workspace/cpy/code'),
                'short-tree': RUNS / 'three_track_gpu_batch_v1/T3', 'transfer-run': RUNS / 'same_issue_followup_v1/T3',
                't3-confirm': RUNS / 't3_transfer_confirmation_gpu_v1', 'ridge-run': RUNS / 'daily_bias_ridge_gpu_v1',
                'fusion-run': RUNS / 'fusion_mechanisms_cpu_v1', 'split': SPLIT_PATH}
    for name, value in defaults.items():
        parser.add_argument(f'--{name}', type=Path, default=value)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_json(args.out / 'plan.json', {'args': {k: str(v) for k, v in vars(args).items()}, 'split': sha256(args.split),
            'alpha': ALPHA, 'donor_days': DONOR_DAYS, 'neighbors': NEIGHBORS, 'seeds': list(B_SEEDS),
            'code': {name: sha256(Path(__file__).with_name(name)) for name in
                     ('run_t3_nearby_donors.py', 'run_fusion_mechanisms.py', 'run_pressure_profile.py', 'tree_model.py')}})
        start = monotonic()
        (args.out / 'T3').mkdir(exist_ok=True)
        try:
            predictor = NearbyPair(args)
            report = t3_stage(args, predict_pair=predictor, title='T3 近邻来源')
            if report['gate']['promoted']:
                report['B']['tree_scores'] = tree_diagnostics(args)
            result = {'status': 'complete', **report}
        except Exception as error:
            traceback.print_exc()
            result = {'status': 'failed', 'error': str(error)}
        write_json(args.out / 'summary.json', {'T3': result, 'seconds': round(monotonic() - start, 2)})
        print(f'结果摘要 {args.out / "summary.json"}', flush=True)
        return int(result['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
