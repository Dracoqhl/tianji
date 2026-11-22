#!/usr/bin/env python3
"""Run chronological T1 loss and frozen T3 head comparisons without warm-season gating."""

import argparse
import fcntl
from functools import partial
from pathlib import Path
import subprocess
import sys
from tempfile import mkdtemp
from time import monotonic

import numpy as np
import pandas as pd

from advance_batch_b import read_frozen_forecast
from batch_energy_short import roster_for, t3_windows
from compare_t2_mlp import select_mlp_runtime
from compare_warm_t1 import MINIMUM_GAIN, PRIMARY_SCORE, b_statistics
from confirm_working_baselines import average_seeds, score_t3
from experiment_companion_t1 import event, identical_seed_groups, read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, cache_paths, cache_version
from run_companion_warm_t1 import check_training_sample, chronological_dates, date_roles, forecast_keys
from run_companion_warm_t1 import sha256, weather_rows, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from same_issue_weather import archive_stamp, archived_issue_weather
from score_forecasts import CAPACITY_MW, DAY_KEYS, MAX_PREDICTION, score_track
from screen_companion_t1_stations import block_comparisons, combine_forecasts, show_progress, station_routes, station_scores
from screen_same_issue_context import CONTEXTS, make_context_network
from screen_short_transfer import pretrain_dates, ridge_residual, targets
from tree_model import load_tree_model, predict_tree
from warm_shared_power import fit_power, labels, load_roster, normalize


SCRIPT = Path(__file__).resolve()
RUNS = Path('/workspace/qhl/runs')
LOSSES = ('daily_rmse', 'daily_mse', 'station_equal_mse')
HEADS = ('shared_ridge', 'half_shared_ridge')
T3_FIXED = 'fixed_50_10'
TRANSFER = 'frozen_context_ridge'


def timed(output, name, operation):
    start = monotonic()
    value = operation()
    event(output, name, seconds=monotonic() - start)
    return value


def station_day_weights(mask, sites):
    present = mask.any(axis=1)
    if not present.any():
        raise ValueError('No valid training station-days for weighting.')
    counts = present.reshape(-1, sites).sum(axis=0)
    counts_per_row = np.tile(counts, len(mask) // sites)
    weights = np.divide(present, counts_per_row, out=np.zeros(len(mask)), where=counts_per_row > 0)
    return weights * present.sum() / weights.sum()


def timing_setup(args, train, days, roster):
    return {'training_dates': list(map(str, train)), 'forecast_dates': list(map(str, days)),
            'weather': archive_stamp(args.workdir, [*train, *days]),
            'coordinates': sha256(args.data_root / 'static/power_stations.csv'),
            'stations': roster, 'device': args.device}


def fit_t1_networks(args):
    import torch

    train, days = chronological_dates(args.split, args.phase)
    roster = load_roster(args.data_root)
    keys = forecast_keys(days, roster)
    settings = {**timing_setup(args, train, days, roster), 'seed': args.seed, 'epochs': 16,
                'network': list(CONTEXTS['full_context']), 'batch': 64}
    freeze_json(args.out / 'settings.json', settings)
    pending = [loss for loss in args.losses if not (args.out / loss / 'predictions_frozen.json').is_file()]
    if not pending:
        return
    weather = timed(args.out, 'neural_weather_read',
                    lambda: archived_issue_weather(args.workdir, args.data_root, [*train, *days], roster, print)[0])
    cut = len(train) * len(roster)
    x_train, x_test, _, _ = timed(args.out, 'neural_normalization', lambda: normalize(weather[:cut], weather[cut:]))
    del weather
    actual, valid = timed(args.out, 'neural_training_labels', lambda: targets(args.data_root, train, roster))
    weights = station_day_weights(valid, len(roster))
    network = partial(make_context_network, dilations=CONTEXTS['full_context'])
    for name in pending:
        print(f'[T1 {args.phase}] seed={args.seed} {name}', flush=True)
        model, predict, trajectory, device, _ = timed(
            args.out, name + '_fit',
            lambda: fit_power(x_train, actual, valid, x_test, 16, 64, args.device, print, seed=args.seed,
                              network_factory=network, objective='daily_rmse' if name == 'daily_rmse' else 'daily_mse',
                              day_weight=weights if name == 'station_equal_mse' else None))
        values = timed(args.out, name + '_predict', predict)
        directory = args.out / name
        directory.mkdir(exist_ok=True)
        torch.save(model.state_dict(), directory / 'network.pt')
        freeze_window(directory, 'T1', keys, {'network': keys.assign(prediction=values.ravel())},
                      {'settings_sha256': sha256(args.out / 'settings.json'), 'loss': name,
                       'device': device, 'train_daily_rmse_mw': trajectory})
        del model, predict


def tree_source(args, phase, seed, keys):
    train, days = chronological_dates(args.split, phase)
    directory = args.out / 'T1' / phase / str(seed) / 'tree'
    receipt = directory / 'training_receipt.json'
    config = args.config or args.companion_code / 'configs/default.json'
    settings = {**timing_setup(args, train, days, load_roster(args.data_root)),
                'config': sha256(config), 'seed': seed, 'threads': args.threads}
    directory.parent.mkdir(parents=True, exist_ok=True)
    plan = directory.parent / 'tree_plan.json'
    freeze_json(plan, settings)
    if not receipt.is_file():
        attempt = Path(mkdtemp(dir=directory.parent, prefix='.tree-attempt-'))
        call(sys.executable, SCRIPT.with_name('run_companion_warm_t1.py'),
             '--chronological', '--phase', phase, '--seed', seed, '--out', attempt, '--data-root', args.data_root,
             '--workdir', args.workdir, '--companion-code', args.companion_code, '--split', args.split,
             '--config', config, '--backend', 'xgboost', '--threads', args.threads, '--device', args.device)
        attempt.rename(directory)
    info = read_json(receipt)
    if (info['chronological_dates'] != {'training': settings['training_dates'], 'forecast': settings['forecast_dates']}
            or info['seed'] != seed or info['backend'] != 'xgboost' or info['device'] != args.device
            or info['config_sha256'] != settings['config']
            or info['feature_spec_sha256'] != settings['weather']['feature_spec']
            or info['companion_cache_metadata_sha256'] != {
                'training': cache_version(cache_paths(args.workdir, train)),
                'prediction': cache_version(cache_paths(args.workdir, days))}):
        raise ValueError('T1 tree receipt conflicts with the new split or inputs.')
    return read_frozen_forecast(directory / info['prediction_file'], info['prediction_sha256'], keys, 'T1'), directory


def call(interpreter, entry, *options):
    subprocess.run([str(interpreter), '-u', str(entry), *map(str, options)], check=True)


def worker_options(args):
    return ('--data-root', args.data_root, '--workdir', args.workdir, '--device', args.device, '--split', args.split)


def t1_arms(args, phase, seed, tree, keys, losses, interpreter):
    output = args.out / 'T1' / phase / str(seed) / 'networks'
    output.mkdir(parents=True, exist_ok=True)
    call(interpreter, SCRIPT, '--worker', 'network', '--phase', phase, '--seed', seed,
         '--losses', *losses, '--out', output, *worker_options(args))
    arms = {'baseline': tree}
    hashes = {}
    for loss in losses:
        path = output / loss / 'predictions_frozen.json'
        manifest = read_json(path)
        if manifest['sources']['settings_sha256'] != sha256(output / 'settings.json'):
            raise ValueError('The network forecast settings changed.')
        hashes[loss] = sha256(path)
        raw = read_frozen_forecast(path.parent / 'network.csv', manifest['hashes']['network'], keys, 'T1')
        arms[loss] = raw
        arms[loss + '_blend'] = blend(tree, raw, 0.20, 'T1')
    return arms, hashes


def point_diagnostics(truth, arms, train_threshold):
    tables = {}
    keys = truth[['station_id', 'date', 'point_index']]
    valid = truth.power_limit.eq(1) & np.isfinite(truth.power)
    scored = valid.groupby([truth.station_id, truth.date]).transform('sum').ge(72)
    adjacent = valid & valid.shift(fill_value=False) & truth.point_index.gt(1)
    actual = truth.power.to_numpy(dtype=float)
    for name, frame in arms.items():
        predicted = keys.merge(frame, validate='one_to_one').prediction
        residual = predicted - actual
        table = {}
        for first in (1, 25, 49, 73):
            mask = valid & scored & truth.point_index.between(first, first + 23)
            if mask.any():
                table[f'{first}-{first+23}'] = float(np.sqrt(np.square(residual[mask]).mean()))
        large = adjacent & scored & truth.power.diff().abs().ge(train_threshold)
        if large.any():
            table['large_ramp_points'] = int(large.sum())
            table['large_ramp_change_mae'] = float((predicted.diff() - truth.power.diff()).abs()[large].mean())
        tables[name] = table
    return tables


def t1_stage(args, interpreter):
    dates, days = chronological_dates(args.split, 'A')
    roster = load_roster(args.data_root)
    keys = forecast_keys(days, roster)
    tree, tree_dir = tree_source(args, 'A', A_SEED, keys)
    waves, hashes = t1_arms(args, 'A', A_SEED, tree, keys, LOSSES, interpreter)
    root = args.out / 'T1'
    waves, lock = freeze_window(root / 'A_forecasts', 'T1', keys, waves,
                                {'tree': sha256(tree_dir / 'training_receipt.json'), 'networks': hashes})
    truth = labels(args.data_root, days, roster)
    training = labels(args.data_root, dates, roster)
    ok = training.power_limit.eq(1) & np.isfinite(training.power)
    shifts = training.power.diff().abs()[ok & ok.shift(fill_value=False) & training.point_index.gt(1)]
    threshold = float(shifts.quantile(.90))
    arms = tuple(name for name in waves if name != 'baseline')
    score = timed(root, 'A_score', lambda: block_comparisons(truth, waves, roster, arms))
    date_blocks = np.array_split(days, 3)
    blocks = []
    for window in date_blocks:
        included = {str(day) for day in window}
        block = block_comparisons(truth[truth.date.isin(included)].reset_index(drop=True),
                                  {name: frame[frame.date.isin(included)] for name, frame in waves.items()}, roster, arms)
        blocks.append(block)
    routes, station_evidence = station_routes(roster, blocks, candidates=arms)
    baseline = score['scores']['baseline'][PRIMARY_SCORE]
    rankings = {name: score['scores'][name][PRIMARY_SCORE] - baseline for name in arms}
    overall = max(rankings, key=rankings.get)
    gate = {'candidate': overall, 'delta': rankings[overall], 'routes': {},
            'promoted': rankings[overall] > 0}
    if routes:
        routed = combine_forecasts(waves['baseline'], waves, routes)
        routed_gain = score_track('T1', truth, routed, roster, allow_unscored_stations=True)[PRIMARY_SCORE] - baseline
        if not gate['promoted'] or routed_gain > gate['delta']:
            gate = {'candidate': 'station_routes', 'delta': routed_gain, 'routes': routes, 'promoted': True}
    loss_deltas = {}
    for loss in LOSSES[1:]:
        loss_deltas[loss] = rankings[loss] - rankings['daily_rmse']
        loss_deltas[loss + '_blend'] = rankings[loss + '_blend'] - rankings['daily_rmse_blend']
    freeze_json(root / 'a_score.json', {'forecasts_sha256': lock, 'score': score, 'blocks': blocks,
                                      'site_evidence': station_evidence, 'rankings': rankings, 'gate': gate,
                                      'loss_deltas_vs_rmse': loss_deltas,
                                      'diagnostics': point_diagnostics(truth, waves, threshold),
                                      'large_change_threshold_mw': threshold})
    print(f'[T1 A] 基线 {baseline:.6f}，分差 {rankings}，B 准入 {gate}', flush=True)
    return gate, (dates, days, roster, tree, tree_dir)


def t1_b(args, gate, interpreter):
    if not gate['promoted']:
        return
    names = set(gate['routes'].values()) if gate['routes'] else {gate['candidate']}
    losses = [name for name in LOSSES if name in {label.removesuffix('_blend') for label in names}]
    train, days = chronological_dates(args.split, 'B')
    roster = load_roster(args.data_root)
    keys = forecast_keys(days, roster)
    folder = args.out / 'T1' / 'B'
    folder.mkdir(exist_ok=True)
    forecasts, receipts = {}, {}
    freeze_json(folder / 'gate.json', {**gate, 'a_score_sha256': sha256(args.out / 'T1' / 'a_score.json')})
    for i, seed in enumerate(B_SEEDS, 1):
        show_progress(folder, 'T1 B', i - 1, 3, f'{seed} 预测')
        tree, directory = tree_source(args, 'B', seed, keys)
        waves, hashes = t1_arms(args, 'B', seed, tree, keys, losses, interpreter)
        candidate = combine_forecasts(tree, waves, gate['routes']) if gate['routes'] else waves[gate['candidate']]
        forecasts[seed], receipts[str(seed)] = freeze_window(
            folder / str(seed) / 'pair', 'T1', keys, {'baseline': tree, 'candidate': candidate},
            {'tree': sha256(directory / 'training_receipt.json'), 'networks': hashes, 'gate': gate})
    digest = freeze_json(folder / 'predictions_frozen.json', receipts)
    return folder, days, roster, keys, forecasts, digest, gate['routes']


def shared_head(args, checkpoint, fit, keys, roster, train, days):
    import torch

    weather = archived_issue_weather(args.workdir, args.data_root, [*train, *days], roster, print)[0]
    info = fit['source_normalization']
    mean, scale = np.asarray(info['mean']), np.asarray(info['scale'])
    available = np.isfinite(weather)
    scaled = np.concatenate([np.where(available, (weather - mean) / scale, 0), ~available], axis=-1).astype(np.float32)
    model = make_context_network(scaled.shape[-1], CONTEXTS['full_context']).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    model.eval()
    features, forecasts = [], []
    with torch.inference_mode():
        for part in np.array_split(scaled, max(1, (len(scaled) + 63) // 64)):
            values = model.encode(torch.as_tensor(part, device=args.device))
            features.append(values.cpu().numpy())
            forecasts.append(torch.sigmoid(model.output(values).squeeze(-1)).cpu().numpy())
    embedding = np.concatenate(features)
    raw = np.concatenate(forecasts).astype(float) * MAX_PREDICTION['T3']
    cut = len(train) * len(roster)
    actual, valid = targets(args.data_root, train, roster)
    counts = valid.sum(axis=1, keepdims=True)
    weights = np.divide(valid, counts, out=np.zeros(valid.shape), where=counts > 0)
    weights *= station_day_weights(valid, len(roster))[:, None]
    common, beta, center, scale = ridge_residual((actual - raw[:cut])[valid], weights[valid], embedding[:cut][valid], embedding[cut:])
    common = common.reshape(raw[cut:].shape)
    own = np.zeros_like(common)
    for index, site in enumerate(roster):
        head = fit['heads'][site]
        if head['points']:
            positions = slice(index, None, len(roster))
            coef = np.asarray(head['coefficients'])
            origin = np.asarray(head['feature_center'])
            sigma = np.asarray(head['feature_scale'])
            own[positions] = ((embedding[cut:][positions] - origin) / sigma) @ coef[:-1] + coef[-1]
    def forecast(change):
        return keys.assign(prediction=np.clip(raw[cut:] + change, 0, MAX_PREDICTION['T3']).ravel())
    return forecast(own), dict(zip(HEADS, (forecast(common), forecast((common + own) / 2)))), {
        'coef': beta.tolist(), 'center': center.tolist(), 'scale': scale.tolist()}


def t3_arms(args, window, train, days, roster, seed, selected=HEADS):
    keys = forecast_keys(days, roster)
    transfer = args.transfer_run / window
    tree_run = args.short_tree / window
    old = read_json(transfer / 'predictions_frozen.json')
    plan = old['plan']
    regular = load_roster(args.data_root)
    allowed, _ = date_roles(args.split, 'A')
    source = pretrain_dates(allowed, days)
    if (plan['score_dates'] != list(map(str, days)) or plan['fit_dates'] != list(map(str, train))
            or plan['source_dates'] != list(map(str, source)) or plan['target_sites'] != roster
            or plan['device'] != args.device or plan['seed'] != A_SEED
            or plan['cache'] != archive_stamp(args.workdir, sorted(set(source + train + days)))
            or plan['station_coordinates_sha256'] != sha256(args.data_root / 'static/power_stations.csv')):
        raise ValueError(f'{window}: saved transfer run comes from a different fold.')
    checkpoint, fit = transfer / 'frozen_encoder.pt', old['fit']
    tree = read_frozen_forecast(tree_run / 'short_only.csv', plan['reference_sha256'], keys, 'T3')
    original = read_frozen_forecast(transfer / f'{TRANSFER}.csv', old['forecasts'][TRANSFER], keys, 'T3')
    baseline = blend(tree, original, 0.25, 'T3')
    if window == T3_FIXED:
        confirmed = args.t3_confirm / 'T3'
        lockfile = confirmed / 'predictions_frozen.json'
        score = read_json(confirmed / 'score.json')
        lock = read_json(lockfile)
        if (score['prediction_lock_sha256'] != sha256(lockfile) or score['active_recipe'] != 'tree_transfer_75_25'
                or lock['plan'] != sha256(confirmed / 'plan.json')):
            raise ValueError('T3 confirmation does not match the adopted prediction lock.')
        seed_dir = confirmed / str(seed)
        marker = seed_dir / 'predictions_frozen.json'
        if lock['seed_locks'][str(seed)] != sha256(marker):
            raise ValueError('T3 seed receipt changed.')
        saved = read_json(marker)
        tree = read_frozen_forecast(seed_dir / 'baseline.csv', saved['hashes']['baseline'], keys, 'T3')
        baseline = read_frozen_forecast(seed_dir / 'candidate.csv', saved['hashes']['candidate'], keys, 'T3')
        if seed != A_SEED:
            transfer = seed_dir / 'transfer'
            fit = read_json(transfer / 'receipt.json')['fit']
            if fit != saved['sources']['fit']['transfer']['fit']:
                raise ValueError('T3 transfer receipt changed.')
            checkpoint = transfer / 'frozen_encoder.pt'
    output = args.out / window / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    signature = {'checkpoint_sha256': sha256(checkpoint), 'fit': fit, 'seed': seed,
                 'transfer_plan_sha256': sha256(args.transfer_run / window / 'predictions_frozen.json'),
                 'selected': list(selected), 'dates': list(map(str, days))}
    marker = output / 'predictions_frozen.json'
    if marker.is_file():
        record = read_json(marker)
        if record['sources']['inputs'] != signature:
            raise ValueError(f'{output}: recorded T3 fit inputs changed.')
        return {arm: read_frozen_forecast(output / f'{arm}.csv', digest, keys, 'T3')
                for arm, digest in record['hashes'].items()}, sha256(marker)

    own, candidates, head = timed(output, 'T3_embeddings_and_ridge',
                                  lambda: shared_head(args, checkpoint, fit, keys, roster, train, days))
    rebuilt = blend(tree, own, 0.25, 'T3')
    if not np.allclose(rebuilt.prediction, baseline.prediction, rtol=0, atol=5e-5):
        raise ValueError('Could not reproduce the frozen 75/25 baseline from its T3 checkpoint.')
    arms = {'baseline': baseline, **{name: blend(tree, candidates[name], 0.25, 'T3') for name in selected}}
    return freeze_window(output, 'T3', keys, arms, {'inputs': signature, 'shared_head': head})


def t3_worker(args):
    roster = roster_for(args.data_root, 'T3')
    results = {}
    windows = list(t3_windows())
    for index, (name, train, days) in enumerate(windows[:2]):
        show_progress(args.out, 'T3 A', index, 2, name)
        arms, lock = t3_arms(args, name, train, days, roster, A_SEED)
        truth = labels(args.data_root, days, roster)
        results[name] = {'lock': lock, 'pairs': {arm: score_t3(truth, roster, {'baseline': arms['baseline'],
                                                                             'candidate': arms[arm]})
                                                 for arm in HEADS}}
    means = {arm: float(np.mean([row['pairs'][arm]['delta'] for row in results.values()])) for arm in HEADS}
    chosen = max(means, key=means.get)
    gate = {'candidate': chosen, 'promoted': means[chosen] > 0, 'a_deltas': means}
    freeze_json(args.out / 'a_score.json', {'windows': results, 'gate': gate})
    show_progress(args.out, 'T3 A', 2, 2, f'完成，分差 {means}')
    if gate['promoted']:
        name, train, days = windows[-1]
        locks = {}
        for index, seed in enumerate(B_SEEDS, 1):
            _, locks[str(seed)] = t3_arms(args, name, train, days, roster, seed, selected=(chosen,))
            show_progress(args.out, 'T3 50/10', index, 3, f'{seed} 预测核验')
        freeze_json(args.out / 'b_lock.json', {'gate': gate, 'seeds': locks, 'a_sha256': sha256(args.out / 'a_score.json')})


def paired_report(folder, track, truth, roster, keys, pairs, source, routes=None):
    if track == 'T3':
        def score(pair):
            return score_t3(truth, roster, pair)
    else:
        def score(pair):
            scores = block_comparisons(truth, pair, roster, ('candidate',))
            return {**scores, 'delta': scores['scores']['candidate'][PRIMARY_SCORE]
                    - scores['scores']['baseline'][PRIMARY_SCORE]}
    values = {str(seed): score(pair) for seed, pair in pairs.items()}
    ensemble = score(average_seeds(keys, pairs, track))
    acceptance = b_statistics([row['delta'] for row in values.values()], ensemble['delta'])
    repeated = {arm: identical_seed_groups({seed: pair[arm].prediction.to_numpy() for seed, pair in pairs.items()})
                for arm in ('baseline', 'candidate')}
    result = {'prediction_lock_sha256': source, 'pairs': values, 'ensemble': ensemble,
              'acceptance': acceptance, 'identical_seed_groups': repeated}
    if routes:
        station_checks = {}
        for site in routes:
            gain = ensemble['station_pairs']['candidate'][site]['delta']
            seed_deltas = [row['station_pairs']['candidate'][site]['delta'] for row in values.values()]
            if gain is None or None in seed_deltas:
                station_checks[site] = {'accepted': False, 'reason': 'no_scoreable_days'}
            else:
                stats = b_statistics(seed_deltas, gain)
                station_checks[site] = {**stats, 'accepted': stats['accepted'] and gain >= MINIMUM_GAIN}
        result['station_acceptance'] = station_checks
        result['accepted_routes'] = {site: routes[site] for site, row in station_checks.items() if row['accepted']}
    freeze_json(folder / 'b_score.json', result)
    print(f'[{track} B] 集成 {ensemble["scores"]["baseline"][PRIMARY_SCORE]:.6f} → '
          f'{ensemble["scores"]["candidate"][PRIMARY_SCORE]:.6f} ({ensemble["delta"]:+.6f})，配对 {acceptance}', flush=True)
    if routes:
        print(f'[{track} B] 通过验收的逐站方案 {result["accepted_routes"]}', flush=True)
    return result


def obs_stress(args, track, train, days, roster, checkpoint, reference, output):
    from unittest.mock import patch

    sys.path.insert(0, str(args.companion_code / 'src'))
    from tianji.constants import ALL_STATIONS
    from tianji.data import obs
    from tianji.data.dataset import build_gbdt_rows, gbdt_feature_names, load_feature_spec
    from tianji.data.static import load_obs_station_coords, load_static_features

    stations = [list(ALL_STATIONS).index(site) for site in roster]
    spec = load_feature_spec(args.workdir, 'train')
    names = list(gbdt_feature_names(spec))
    cols = [names.index(name) for name in spec['obs_names']]
    graph = load_static_features(args.workdir / 'stats/static_features.npz')
    indices = graph['obs_neighbor_idx'][stations]
    distance = graph['obs_neighbor_dist'][stations]
    sensors = load_obs_station_coords(args.data_root)['station_id'].tolist()
    candidate_sensors = np.unique(indices[:, :5])
    wind = [obs.OBS_VARS.index(name) for name in ('wind_dir_sin', 'wind_dir_cos', 'wind_speed')]
    context, _, dtype = check_training_sample(cache_paths(args.workdir, train), train, stations, roster, spec,
                                               names, build_gbdt_rows, args.data_root)
    model = load_tree_model(checkpoint)
    original_hash = sha256(checkpoint)
    keys = forecast_keys(days, roster)
    baseline, masked, flags = [], [], []
    for day, path in zip(days, cache_paths(args.workdir, days)):
        directory = args.data_root / 'train' / path.stem
        raw, valid = obs.load_obs_input(directory / 'obs/obs_input.csv', sensors)
        real, observed_names = obs.obs_features(directory, sensors, indices, distance)
        matrix = weather_rows([path], stations, roster, spec, names, context, dtype).reshape(len(roster), 96, -1)
        real_cached = matrix[:, 0, cols]
        if observed_names != spec['obs_names'] or not np.allclose(real, real_cached, atol=1e-5, rtol=1e-6, equal_nan=True):
            raise ValueError(f'{path.name}: raw observation aggregation differs from companion cache.')
        possible = np.argwhere(valid[candidate_sensors][:, :, wind].any(axis=2))
        changed = np.zeros(len(roster), dtype=bool)
        base = predict_tree(model, matrix.reshape(-1, len(names)), backend='xgboost').astype(float)
        perturbed = base.copy()
        location = None
        if len(possible):
            rng = np.random.default_rng(A_SEED + day.toordinal())
            row, time = possible[rng.integers(len(possible))]
            sensor = candidate_sensors[row]
            dropped = raw.copy()
            dropped[sensor, time, wind] = np.nan
            with patch.object(obs, 'load_obs_input', return_value=(dropped, np.isfinite(dropped))):
                derived, _ = obs.obs_features(directory, sensors, indices, distance)
            changed = ~(np.equal(derived, real) | (np.isnan(derived) & np.isnan(real))).all(axis=1)
            if changed.any():
                matrix[..., cols] = derived[:, None, :]
                points = np.repeat(changed, 96)
                perturbed[points] = predict_tree(model, matrix[changed].reshape(-1, len(names)), backend='xgboost')
            location = [sensors[sensor], int(obs.OBS_POINTS[time])]
        baseline.append(base)
        masked.append(perturbed)
        natural = (~valid[indices[:, :5]][:, :, :, wind]).any(axis=(1, 2, 3))
        flags.extend({'station_id': site, 'date': str(day), 'changed': bool(changed[i]),
                      'natural_missing': bool(natural[i]), 'masked_location': location}
                     for i, site in enumerate(roster))
    arms = {'baseline': keys.assign(prediction=np.clip(np.concatenate(baseline) * CAPACITY_MW, 0, MAX_PREDICTION[track])),
            'masked': keys.assign(prediction=np.clip(np.concatenate(masked) * CAPACITY_MW, 0, MAX_PREDICTION[track]))}
    if not np.allclose(arms['baseline'].prediction, reference.prediction, atol=5e-5, rtol=0):
        raise ValueError('Observation stress baseline differs from the stored tree prediction.')
    truth = labels(args.data_root, days, roster).merge(pd.DataFrame(flags), on=DAY_KEYS, validate='many_to_one')
    scores = {}
    for cohort, use in {'all': np.ones(len(truth), dtype=bool), 'changes': truth.changed,
                        'natural_missing': truth.natural_missing, 'natural_complete': ~truth.natural_missing}.items():
        subset = truth.assign(power_limit=truth.power_limit.where(use, 0))
        # A subgroup can contain points without a scoreable day.
        if station_scores(subset, arms['baseline']):
            scores[cohort] = {name: score_track(track, subset, frame, roster, allow_unscored_stations=True)
                              for name, frame in arms.items()}
    output.mkdir(parents=True, exist_ok=True)
    _, lock = freeze_window(output, track, keys, arms, {'tree_sha256': original_hash})
    report = {'prediction_lock': lock, 'affected_station_days': sum(row['changed'] for row in flags),
              'total_station_days': len(flags), 'scores': scores, 'details': flags}
    write_json(output / 'score.json', report)
    print(f'[{track} 缺报诊断] 变化站日 {report["affected_station_days"]}/{len(flags)}，{output}', flush=True)


def observation_checks(args, t1_bundle):
    train, days, roster, tree, directory = t1_bundle
    items = [('T1', 'recent_A', train, days, roster, tree, directory / 'companion_T1.joblib')]
    short = roster_for(args.data_root, 'T3')
    for name, train, days in list(t3_windows())[:2]:
        directory = args.short_tree / name
        digest = read_json(args.short_tree / 'results.json')['windows'][name]['predictions_sha256']['short_only']
        reference = read_frozen_forecast(directory / 'short_only.csv', digest, forecast_keys(days, short), 'T3')
        items.append(('T3', name, train, days, short, reference, directory / 'short_only.joblib'))
    for track, name, train, days, sites, reference, checkpoint in items:
        output = args.out / track / 'obs_stress' / name
        output.mkdir(parents=True, exist_ok=True)
        try:
            timed(output, 'observation_stress', lambda: obs_stress(args, track, train, days, sites, checkpoint, reference, output))
        except (OSError, ModuleNotFoundError, KeyError, ValueError) as error:
            event(output, 'observation_stress', status='unavailable', detail=str(error))
            print(f'[{track} 缺报诊断] {name} 暂未完成，{error}', flush=True)


def t3_score_locked(args):
    folder = args.out / 'T3'
    gate = read_json(folder / 'a_score.json')['gate']
    if not gate['promoted']:
        return {'promoted': False}
    marker = folder / 'b_lock.json'
    locks = read_json(marker)
    if locks['gate'] != gate or locks['a_sha256'] != sha256(folder / 'a_score.json') or set(locks['seeds']) != set(map(str, B_SEEDS)):
        raise ValueError('T3 B enrollment or seed locks changed.')
    name, _, days = list(t3_windows())[-1]
    roster = roster_for(args.data_root, 'T3')
    keys = forecast_keys(days, roster)
    pairs = {}
    for seed in B_SEEDS:
        directory = folder / name / str(seed)
        digest = directory / 'predictions_frozen.json'
        if sha256(digest) != locks['seeds'][str(seed)]:
            raise ValueError('T3 locked forecast hash changed.')
        row = read_json(digest)
        pairs[seed] = {role: read_frozen_forecast(directory / f'{arm}.csv', row['hashes'][arm], keys, 'T3')
                       for role, arm in (('baseline', 'baseline'), ('candidate', gate['candidate']))}
    result = paired_report(folder, 'T3', labels(args.data_root, days, roster), roster, keys, pairs, sha256(marker))
    return {'promoted': True, 'candidate': gate['candidate'], 'accepted': result['acceptance']['accepted'],
            'delta': result['ensemble']['delta']}


def execute(args):
    work = args.out
    sources = {name: str(getattr(args, name).resolve()) for name in ('data_root', 'workdir', 'short_tree', 'transfer_run', 't3_confirm')}
    scripts = (SCRIPT.name, 'warm_shared_power.py', 'run_companion_warm_t1.py', 'screen_short_transfer.py',
               'tree_model.py', 'score_forecasts.py', 'same_issue_weather.py')
    plan = {'source': sources, 'split_sha256': sha256(args.split), 'device': args.device, 'threads': args.threads,
            'code_sha256': {name: sha256(SCRIPT.with_name(name)) for name in scripts},
            't1': {phase: [list(map(str, values)) for values in chronological_dates(args.split, phase)] for phase in ('A', 'B')}}
    freeze_json(work / 'plan.json', plan)
    print('[GPU] 检查 PyTorch 环境', flush=True)
    interpreter = select_mlp_runtime(args, require_weather=False)[0]
    status = {}
    failures = []

    def stage(name, run):
        try:
            result = run()
        except (OSError, KeyError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            print(f'[{name}] {error}', flush=True)
            status[name] = {'status': 'failed', 'detail': str(error)}
            failures.append(name)
            result = None
        else:
            status[name] = {'status': 'complete'}
        write_json(work / 'status.json', status)
        event(work, name, **status[name])
        return result

    show_progress(work, 'T1/T3 对照', 0, 4, 'T1 时序 A')
    first = stage('T1_A', lambda: t1_stage(args, interpreter))
    show_progress(work, 'T1/T3 对照', 1, 4, 'T3 早窗与候选锁定')
    (work / 'T3').mkdir(exist_ok=True)
    stage('T3_predictions', lambda: call(interpreter, SCRIPT, '--worker', 't3', '--out', work / 'T3',
                                          '--short-tree', args.short_tree, '--transfer-run', args.transfer_run,
                                          '--t3-confirm', args.t3_confirm, *worker_options(args)))
    show_progress(work, 'T1/T3 对照', 2, 4, 'T1 B 预测与分赛道验收')
    if first is not None and first[0]['promoted']:
        ready = stage('T1_B_predictions', lambda: t1_b(args, first[0], interpreter))
        if ready is not None:
            folder, dates, roster, keys, arms, source, routes = ready
            stage('T1_B_score', lambda: paired_report(folder, 'T1', labels(args.data_root, dates, roster),
                                                      roster, keys, arms, source, routes))
    elif first is not None:
        print('[T1 B] A 没有通过的候选', flush=True)
    if status['T3_predictions']['status'] == 'complete':
        t3 = stage('T3_B_score', lambda: t3_score_locked(args))
        if t3 is not None and not t3['promoted']:
            print('[T3 B] A 没有通过的候选', flush=True)
    show_progress(work, 'T1/T3 对照', 3, 4, '观测缺报推理')
    if first is not None:
        stage('obs_stress', lambda: observation_checks(args, first[1]))
    show_progress(work, 'T1/T3 对照', 4, 4, '报告已保存')
    print(f'报告目录 {work}', flush=True)
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RUNS / 't1_t3_fast_gpu_v1')
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--short-tree', type=Path, default=RUNS / 'three_track_gpu_batch_v1/T3')
    parser.add_argument('--transfer-run', type=Path, default=RUNS / 'same_issue_followup_v1/T3')
    parser.add_argument('--t3-confirm', type=Path, default=RUNS / 't3_transfer_confirmation_gpu_v1')
    parser.add_argument('--mlp-run', type=Path, default=RUNS / 'warm_shared_power_v1')
    parser.add_argument('--mlp-python', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--worker', choices=('network', 't3'))
    parser.add_argument('--phase', choices=('A', 'B'), default='A')
    parser.add_argument('--seed', type=int, choices=B_SEEDS, default=A_SEED)
    parser.add_argument('--losses', nargs='+', choices=LOSSES, default=LOSSES)
    args = parser.parse_args()
    args.mlp_weather_python = None
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker == 'network':
        fit_t1_networks(args)
    elif args.worker == 't3':
        t3_worker(args)
    else:
        with (args.out / '.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(execute(args))


if __name__ == '__main__':
    main()
