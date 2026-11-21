#!/usr/bin/env python3
"""Call the server-side companion trainer on the fixed warm T1 split."""

import argparse
from datetime import date
import hashlib
from inspect import getsourcefile, signature
import json
from pathlib import Path
import sys
from time import monotonic
from unittest.mock import patch

import numpy as np
import pandas as pd

from companion_t1_features import COMPONENTS, LocalWindCache, add_components, station_day_weights
from score_forecasts import CAPACITY_MW, MAX_PREDICTION, score_track
from tree_model import best_iteration, check_backend, fitted_device, lightgbm_progress, predict_tree
from tree_model import save_tree_model, train_tree, xgboost_parameters
from warm_shared_power import FROZEN_SPLIT, labels, load_roster, manifest_days


def cache_paths(workdir, days):
    paths = [workdir / 'features' / 'train' / f'{day:%Y%m%d}_0000.npz' for day in days]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)} selected daily feature archives, e.g. {missing[:3]}.')
    return paths


def row_positions(meta, station_indices, n_days, *, phase, feature_rows):
    station_indices = [int(index) for index in station_indices]
    stations = {index: i for i, index in enumerate(station_indices)}
    axes = {name: np.asarray(meta[name]) for name in ('day_idx', 'station_idx', 'point_idx')}
    for name, values in axes.items():
        if values.shape != (feature_rows,):
            raise ValueError(f'{phase}: {name} shape {values.shape} does not match {feature_rows} feature rows.')
        numeric = np.asarray(pd.to_numeric(values, errors='raise'))
        if not np.isfinite(numeric).all() or not np.equal(numeric, numeric.astype(np.intp)).all():
            raise ValueError(f'{phase}: {name} must contain finite integer row keys.')
        axes[name] = numeric.astype(np.intp)
    days = axes['day_idx']
    points = axes['point_idx']
    slots = np.fromiter((stations.get(int(i), -1) for i in axes['station_idx']), dtype=np.intp, count=len(days))
    if np.any(slots < 0):
        raise ValueError(f'{phase}: unexpected station indices {np.unique(axes["station_idx"][slots < 0])[:5].tolist()}.')
    if np.any((days < 0) | (days >= n_days)) or np.any((points < 1) | (points > 96)):
        raise ValueError(f'{phase}: indexed days={n_days}, observed day_idx range={days.min()}..{days.max()}, '
                         f'point_idx range={points.min()}..{points.max()}.')
    positions = (days * len(station_indices) + slots) * 96 + points - 1
    expected_count = n_days * len(station_indices) * 96
    unique = np.unique(positions)
    stats = {'rows': feature_rows, 'unique_keys': len(unique), 'full_grid_keys': expected_count,
             'duplicate_rows': feature_rows - len(unique), 'missing_grid_keys': expected_count - len(unique)}
    rows_by_day = np.bincount(days, minlength=n_days)
    stats['rows_per_day_min'] = int(rows_by_day.min())
    stats['rows_per_day_max'] = int(rows_by_day.max())
    print(f'{phase} keys: {json.dumps(stats)}', flush=True)
    if stats['duplicate_rows']:
        raise ValueError(f'{phase}: duplicate row keys. Key counts printed above.')
    return positions, days, stats


def context_values(inputs, positions, size, column):
    indices, first = np.unique(positions, return_index=True)
    values = inputs[first, column]
    if len(indices) != size or not np.isfinite(values).all() or not np.array_equal(values[positions], inputs[:, column]):
        raise ValueError('Sampled training rows cannot establish the companion station or point feature convention.')
    return values


FEATURE_GROUPS = {'nwp_point': 'nwp_names', 'static': 'static_names',
                  'sat': 'sat_names', 'obs': 'obs_names'}
CONTEXT = ('station_idx', 'point_index', 'local_hour')
SPLIT_PATH = Path(__file__).resolve().parents[1] / 'configs/warm_date_split_v1.csv'
A_SEED = 20261020
B_SEEDS = (A_SEED, 20261037, 20261117)
DEFAULT_THREADS = 12
GPU_THREADS = 4
EARLY_STOP_DAYS = 90
SMOKE_SEED = 42
SMOKE_TRAIN_DAYS = 20
SMOKE_VALIDATION_DAYS = 10
SMOKE_MAX_ROUNDS = 20


def sha256(path):
    with path.open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, result):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def date_roles(path, phase):
    split = manifest_days(path)
    if phase == 'A':
        return split['train'], split['development']
    frame = pd.read_csv(path, dtype=str)
    sealed = [date.fromisoformat(day) for day in frame.loc[frame['role'].eq('sealed'), 'date']]
    return sorted(split['train'] + split['development']), sealed


def training_block_dates(path, first, last):
    train, _ = date_roles(path, 'A')
    prior = [day for day in train if day < first]
    held_out = [day for day in train if first <= day <= last]
    if not prior or not held_out or len(held_out) != (last - first).days + 1:
        raise ValueError('A training block must be wholly within the training role and have earlier training days.')
    return prior, held_out


def block_early_stop_days(train_days):
    return min(EARLY_STOP_DAYS, max(7, len(train_days) // 5))


def cache_version(paths):
    files = [(path.name, path.stat()) for path in paths]
    return hashlib.sha256(json.dumps([(name, info.st_size, info.st_mtime_ns)
                                     for name, info in files]).encode()).hexdigest()


def weather_rows(paths, stations, roster, spec, names, context, dtype):
    """Read only named weather and key arrays. These NPZs can also contain targets."""
    n_stations, n_points = len(stations), 96
    station_rows = np.asarray(stations, dtype=np.intp)
    expected = [name for group in FEATURE_GROUPS.values() for name in spec[group]] + list(CONTEXT)
    if len(names) != len(set(names)) or sorted(names) != sorted(expected):
        raise ValueError('Feature columns differ from the companion cache schema.')
    columns = {name: index for index, name in enumerate(names)}
    full = np.empty((len(paths), n_stations, n_points, len(names)), dtype=dtype)
    full[..., columns['station_idx']] = context['station_idx'][:, None]
    for name in CONTEXT[1:]:
        full[..., columns[name]] = context[name]
    groups = {array: [columns[name] for name in spec[group]] for array, group in FEATURE_GROUPS.items()}
    for day, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as cache:
            if cache['date_code'].tolist() != [path.stem] or cache['station_ids'][station_rows].tolist() != roster:
                raise ValueError(f'{path.name}: cached date or station order disagrees with the prediction grid.')
            for array, group_columns in groups.items():
                values = cache[array][station_rows]
                expected_shape = (n_stations, n_points, len(group_columns)) if array == 'nwp_point' else (n_stations, len(group_columns))
                if values.shape != expected_shape:
                    raise ValueError(f'{path.name}: {array} has shape {values.shape}, expected {expected_shape}.')
                full[day][..., group_columns] = values if array == 'nwp_point' else values[:, None, :]
    return full.reshape(-1, len(names))


def check_training_sample(paths, days, stations, roster, spec, names, build_rows, data_root):
    for count in (3, 12):
        indices = np.unique(np.rint(np.linspace(0, len(paths) - 1, count)).astype(int))
        sample_paths = [paths[i] for i in indices]
        sample_days = [days[i] for i in indices]
        matrix, _, metadata = build_rows(sample_paths, stations, CAPACITY_MW, include_invalid=True, verbose=False)
        positions, _, stats = row_positions(metadata, stations, len(indices), phase='sampled training', feature_rows=len(matrix))
        station_slots, point_slots = positions // 96 % len(stations), positions % 96
        if len(np.unique(station_slots)) == len(stations) and len(np.unique(point_slots)) == 96:
            break
    training_targets(data_root, sample_days, roster, positions, 'sampled training')
    context = {'station_idx': context_values(matrix, station_slots, len(stations), names.index('station_idx'))}
    context.update({name: context_values(matrix, point_slots, 96, names.index(name)) for name in CONTEXT[1:]})
    full = weather_rows(sample_paths, stations, roster, spec, names, context, matrix.dtype)
    for column, name in enumerate(names):
        if not np.array_equal(full[positions, column], matrix[:, column], equal_nan=True):
            raise ValueError(f'Weather-only assembly differs from the companion row builder in {name}.')
    print('Weather-only assembly matches the companion training rows on sampled dates.', flush=True)
    return context, stats, matrix.dtype


def training_targets(root, dates, roster, positions, phase):
    truth = labels(root, dates, roster)
    power = truth['power'].to_numpy(dtype=float)
    required = np.isfinite(power) & truth['power_limit'].eq(1).to_numpy()
    valid = required[positions]
    missing = int(required.sum() - valid.sum())
    print(f'{phase}: qhl valid labels={int(required.sum())}, available in features={int(valid.sum())}, missing={missing}', flush=True)
    if missing:
        raise ValueError(f'{phase}: feature rows omit {missing} valid qhl labels. Check the cache and build_gbdt_rows mask.')
    return (power[positions[valid]] / CAPACITY_MW).astype(np.float32), valid


def fit_companion(trainer, inputs, target, names, params, *, seed, threads, weights=None, **kwargs):
    """Use the companion wrapper's callbacks and parameters, with two explicit fit controls."""
    from lightgbm import LGBMRegressor

    original = LGBMRegressor.fit
    fit_signature = signature(original)
    calls = 0

    def fit(model, *args, **fit_kwargs):
        nonlocal calls
        calls += 1
        bound = fit_signature.bind(model, *args, **fit_kwargs)
        existing = model.get_params()
        if existing.get('objective') != 'huber':
            raise ValueError('Expected the companion LightGBM Huber objective.')
        aliases = {name: seed for name in ('seed', 'random_seed') if name in existing}
        model.set_params(random_state=seed, n_jobs=threads, **aliases)
        if bound.arguments.get('sample_weight') is not None:
            raise ValueError('Unexpected existing companion sample_weight.')
        if weights is not None:
            if not np.array_equal(np.asarray(bound.arguments['y']).reshape(-1), target):
                raise ValueError('Training weights require the original companion fit row order.')
            bound.arguments['sample_weight'] = weights
        bound.arguments['callbacks'] = [*(bound.arguments.get('callbacks') or []), lightgbm_progress()]
        return original(*bound.args, **bound.kwargs)

    with patch.object(LGBMRegressor, 'fit', fit):
        model = trainer(inputs, target, names, dict(params, random_state=seed, n_jobs=threads),
                        categorical_features=['station_idx'], **kwargs)
    if calls != 1:
        raise ValueError('Expected one sklearn LightGBM fit per companion training call.')
    effective = model.booster_.params
    if (effective.get('random_state') != seed or effective.get('seed', seed) != seed or
            effective.get('random_seed', seed) != seed or effective.get('num_threads') != threads or
            effective.get('objective') != 'huber'):
        raise ValueError('The fitted LightGBM booster did not retain the registered seed, thread budget and objective.')
    return model, {key: effective[key] for key in sorted(effective)
                   if key.endswith('seed') or key in ('objective', 'random_state', 'num_threads')}


def forecast_keys(days, roster):
    return pd.MultiIndex.from_product([[day.isoformat() for day in days], roster, range(1, 97)],
                                      names=['date', 'station_id', 'point_index']).to_frame(index=False)


def run(args):
    sys.path.insert(0, str(args.companion_code / 'src'))
    backend_version = check_backend(args.backend, args.device)
    from tianji.config import load_config
    from tianji.constants import ALL_STATIONS, REGULAR_IDX
    from tianji.data.dataset import build_gbdt_rows, gbdt_feature_names, load_feature_spec
    if args.backend == 'lightgbm':
        from tianji.models.gbdt import predict, save_model, train_lgbm

    if args.train_block:
        train_days, forecast_days = training_block_dates(args.split, *args.train_block)
    else:
        train_days, forecast_days = date_roles(args.split, args.phase)
    if args.smoke:
        train_days = train_days[:SMOKE_TRAIN_DAYS]
        forecast_days = forecast_days[:SMOKE_VALIDATION_DAYS]
    if not 1 <= args.early_stop_days < len(train_days):
        raise ValueError('Early-stopping days must be a proper subset of training days.')
    stations = list(REGULAR_IDX)
    roster = [ALL_STATIONS[index] for index in stations]
    if set(roster) != set(load_roster(args.data_root)):
        raise ValueError('The companion station roster differs from the qhl submission template.')
    train_paths = cache_paths(args.workdir, train_days)
    prediction_paths = cache_paths(args.workdir, forecast_days)
    versions = {'training': cache_version(train_paths), 'prediction': cache_version(prediction_paths)}
    spec_path = args.workdir / 'stats' / 'feature_spec_train.json'
    spec = load_feature_spec(args.workdir, 'train')
    original_names = list(gbdt_feature_names(spec))
    start = monotonic()
    context, sample_stats, dtype = check_training_sample(
        train_paths, train_days, stations, roster, spec, original_names, build_gbdt_rows, args.data_root,
    )
    print(f'Checking full weather-only prediction grid for {len(forecast_days)} dates', flush=True)
    predict_inputs = weather_rows(prediction_paths, stations, roster, spec, original_names, context, dtype)
    keys = forecast_keys(forecast_days, roster)
    if len(keys) != len(predict_inputs) or keys.duplicated().any():
        raise ValueError('Expected a complete and unique prediction grid.')
    print(f'Prediction grid validated: {len(keys)} keys, {len(roster)} stations.', flush=True)
    if args.check_only:
        return

    config_path = args.config or args.companion_code / 'configs' / 'default.json'
    params = dict(load_config(str(config_path))['gbdt'])
    if args.smoke:
        params['n_estimators'] = min(SMOKE_MAX_ROUNDS, int(params['n_estimators']))
    params.update(random_state=args.seed, n_jobs=args.threads)
    config_hash = sha256(config_path)
    companion_functions = [load_config, build_gbdt_rows, gbdt_feature_names]
    if args.backend == 'lightgbm':
        companion_functions.extend((train_lgbm, predict, save_model))
    source_hashes = {fn.__name__: sha256(Path(getsourcefile(fn))) for fn in companion_functions}
    local_cache = (LocalWindCache(args.data_root, args.local_wind_cache, roster)
                   if 'local_wind' in args.components else None)
    prediction_positions = np.arange(len(keys), dtype=np.intp)
    predict_inputs, feature_names, diagnostic, prediction_local = add_components(
        predict_inputs, original_names, prediction_positions, prediction_paths, local_cache, args.components,
    )

    print(f'Reading training features for {len(train_paths)} selected dates', flush=True)
    weather, _, metadata = build_gbdt_rows(train_paths, stations, CAPACITY_MW, include_invalid=True, verbose=True)
    positions, day_idx, train_key_stats = row_positions(metadata, stations, len(train_days),
                                                       phase='training', feature_rows=len(weather))
    target, valid = training_targets(args.data_root, train_days, roster, positions, 'training')
    positions = positions[valid]
    inner = day_idx[valid] >= len(train_days) - args.early_stop_days
    if not inner.any() or inner.all():
        raise ValueError('Training or inner early-stopping region has no valid power points.')
    del metadata, day_idx
    inputs = np.ascontiguousarray(weather[valid], dtype=np.float32)
    del weather
    inputs, refit_names, _, training_local = add_components(
        inputs, original_names, positions, train_paths, local_cache, args.components,
    )
    if refit_names != feature_names:
        raise ValueError('Prediction and training feature schemas disagree.')
    valid_sample_hash = hashlib.sha256(positions.tobytes() + target.tobytes()).hexdigest()
    selection_weight = refit_weight = None
    if 'station_day_weight' in args.components:
        selection_weight = station_day_weights(positions[~inner], len(stations))
        refit_weight = station_day_weights(positions, len(stations))
    assemble_seconds = monotonic() - start

    def fit(X, y, config, weights, **validation):
        if args.backend == 'lightgbm':
            return fit_companion(train_lgbm, X, y, feature_names, config,
                                 seed=args.seed, threads=args.threads, weights=weights, **validation)
        if config.get('objective', 'huber') != 'huber':
            raise ValueError('Expected the T1 Huber objective.')
        model = train_tree(
            X, y, feature_names, config, categorical_features=['station_idx'], sample_weight=weights,
            early_stopping_rounds=int(config.get('early_stopping_rounds', 100)),
            backend=args.backend, device=args.device, **validation,
        )
        return model, dict(xgboost_parameters(config, args.device), effective_device=fitted_device(model.booster))

    fit_start = monotonic()
    print(f'Selecting iterations inside the {len(train_days)} training dates, seed {args.seed}, '
          f'{args.backend}/{args.device}, up to {params["n_estimators"]} trees', flush=True)
    selection, selection_params = fit(inputs[~inner], target[~inner], params, selection_weight,
                                      X_valid=inputs[inner], y_valid=target[inner])
    best = best_iteration(selection, backend=args.backend, fallback=params['n_estimators'])
    if best < 1:
        raise ValueError('The companion trainer returned an invalid iteration budget.')
    del selection
    print(f'Refitting all {len(train_days)} training dates with {best} trees', flush=True)
    fitted, refit_params = fit(inputs, target, dict(params, n_estimators=best), refit_weight)
    fit_seconds = monotonic() - fit_start
    save = save_model if args.backend == 'lightgbm' else save_tree_model
    save(fitted, args.out / 'companion_T1.joblib')
    del inputs

    prediction_start = monotonic()
    location = 'training-block holdout' if args.train_block else f'phase {args.phase}'
    print(f'Predicting the {len(forecast_days)} {location} dates', flush=True)
    values = (predict(fitted, predict_inputs) if args.backend == 'lightgbm'
              else predict_tree(fitted, predict_inputs, backend=args.backend))
    prediction = np.asarray(values, dtype=float).reshape(-1) * CAPACITY_MW
    if not np.isfinite(prediction).all() or len(prediction) != len(keys):
        raise ValueError('Invalid forecast length or non-finite prediction.')
    keys['prediction'] = np.clip(prediction, 0, MAX_PREDICTION['T1'])
    forecast_csv = args.out / f'companion_T1_{"development" if args.phase == "A" else "sealed"}.csv'
    pending = forecast_csv.with_suffix('.tmp')
    keys.to_csv(pending, index=False)
    pending.replace(forecast_csv)
    diagnostic_info = None
    if args.diagnostic and args.diagnostic in diagnostic:
        path = args.out / 'diagnostic.npy'
        np.save(path, diagnostic[args.diagnostic], allow_pickle=False)
        diagnostic_info = {'component': args.diagnostic, 'sha256': sha256(path), 'file': path.name}

    result = {
        'phase': args.phase, 'seed': args.seed, 'components': args.components, 'smoke': args.smoke,
        'train_block': [day.isoformat() for day in args.train_block] if args.train_block else None,
        'backend': args.backend, 'device': args.device, 'backend_version': backend_version,
        'data_root': str(args.data_root.resolve()), 'feature_workdir': str(args.workdir.resolve()),
        'feature_spec_sha256': sha256(spec_path), 'companion_cache_metadata_sha256': versions,
        'local_wind_cache': {'training': training_local, 'prediction': prediction_local},
        'split_sha256': FROZEN_SPLIT, 'config_sha256': config_hash,
        'companion_source_sha256': source_hashes,
        'threads': args.threads, 'train_days': len(train_paths), 'forecast_days': len(prediction_paths),
        'early_stopping_days_inside_train': args.early_stop_days,
        'early_stopping_start_date': train_days[-args.early_stop_days].isoformat(),
        'train_valid_rows': len(target), 'valid_training_sample_sha256': valid_sample_hash,
        'feature_keys': {'training': train_key_stats, 'training_sample': sample_stats,
                         'prediction': {'rows': len(keys), 'duplicate_rows': 0, 'missing_grid_keys': 0}},
        'early_stopping_valid_rows': int(inner.sum()), 'best_iteration': best, 'final_n_estimators': best,
        'fit_parameters': {'selection': selection_params, 'refit': refit_params},
        'feature_count': len(feature_names), 'feature_names': feature_names,
        'prediction_file': forecast_csv.name, 'prediction_rows': len(keys),
        'clipped_predictions': int(((prediction < 0) | (prediction > MAX_PREDICTION['T1'])).sum()),
        'prediction_sha256': sha256(forecast_csv), 'diagnostic': diagnostic_info,
        'phase_seconds': {'assembly': assemble_seconds, 'fit': fit_seconds,
                          'prediction_and_export': monotonic() - prediction_start},
    }
    if args.backend == 'lightgbm':
        result['lightgbm_version'] = backend_version
        result['lightgbm_fit_parameters'] = result['fit_parameters']
    if versions != {'training': cache_version(train_paths), 'prediction': cache_version(prediction_paths)}:
        raise ValueError('Companion cache files changed while training or predicting.')
    write_json(args.out / 'training_receipt.json', result)
    print(f'Wrote {forecast_csv}', flush=True)
    if args.smoke:
        truth = labels(args.data_root, forecast_days, roster)
        scores = score_track('T1', truth, keys, roster, allow_unscored_stations=True)
        write_json(args.out / 'smoke_score.json', scores)
        print(f'T1 smoke score={scores["readme_unclipped_score"]:.6f}, fit={fit_seconds:.2f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--companion-code', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True, help='Existing verified companion feature workdir in the server.')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--phase', choices=('A', 'B'), default='A')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--components', choices=COMPONENTS, nargs='*', default=[])
    parser.add_argument('--diagnostic', choices=COMPONENTS)
    parser.add_argument('--local-wind-cache', type=Path, default=Path('/workspace/qhl/runs/companion_t1_local_wind'))
    parser.add_argument('--early-stop-days', type=int)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--backend', choices=('lightgbm', 'xgboost'), default='lightgbm')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--smoke', action='store_true', help='Run 20 training days, 10 A days and at most 20 trees.')
    parser.add_argument('--train-block', type=date.fromisoformat, nargs=2, metavar=('FIRST', 'LAST'),
                        help='Predict an all-training-role block, fitting only on earlier training dates.')
    parser.add_argument('--check-only', action='store_true', help='Check the full prediction grid and sampled training keys.')
    args = parser.parse_args()
    if args.smoke and (args.phase != 'A' or args.out is None):
        parser.error('Use --phase A and an explicit --out for --smoke.')
    if args.train_block and (args.phase != 'A' or args.smoke or args.out is None):
        parser.error('Use --phase A and an explicit --out for --train-block.')
    if args.threads is None:
        args.threads = GPU_THREADS if args.device == 'cuda' else DEFAULT_THREADS
    if args.seed is None:
        args.seed = SMOKE_SEED if args.smoke else A_SEED
    if args.early_stop_days is None:
        if args.train_block:
            train, _ = training_block_dates(args.split, *args.train_block)
            args.early_stop_days = block_early_stop_days(train)
        else:
            args.early_stop_days = 4 if args.smoke else EARLY_STOP_DAYS
    if args.out is None:
        suffix = 'v1' if args.backend == 'lightgbm' else f'{args.backend}_{args.device}'
        args.out = Path(f'/workspace/qhl/runs/companion_warm_t1_{suffix}')
    if args.threads < 1:
        parser.error('--threads must be positive.')
    if args.device == 'cuda' and args.threads > GPU_THREADS:
        parser.error(f'Use at most {GPU_THREADS} CPU threads with CUDA.')
    if len(args.components) != len(set(args.components)):
        parser.error('--components cannot contain duplicates.')
    args.components = [component for component in COMPONENTS if component in args.components]
    if not args.smoke and args.seed not in ((A_SEED,) if args.phase == 'A' else B_SEEDS):
        parser.error('Seed is outside the registered seeds for this phase.')
    if args.diagnostic and (args.phase != 'A' or args.diagnostic not in args.components):
        parser.error('--diagnostic must name an enabled component in phase A.')
    if not args.check_only:
        args.out.mkdir(parents=True, exist_ok=True)
        if any((args.out / filename).exists() for filename in
               ('training_receipt.json', 'companion_T1_development.csv', 'companion_T1_sealed.csv')):
            parser.error('Frozen companion predictions already exist. Choose a new output directory for a rerun.')
    run(args)


if __name__ == '__main__':
    main()
