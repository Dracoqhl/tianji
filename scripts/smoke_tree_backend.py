#!/usr/bin/env python3
"""Small tree backend checks, including a training-only T2 calibration smoke."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic

import numpy as np
import pandas as pd

from independent_t2 import KEYS, feature_cache, load_daily_targets
from run_companion_warm_t1 import SPLIT_PATH, date_roles, write_json
from score_forecasts import MAX_PREDICTION, score_track
from tree_model import best_iteration, check_backend, fitted_device, load_tree_model, predict_tree, save_tree_model, train_tree


THREAD_LIMITS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
ROUNDS = 20
SEED = 42


def model_info(model, args, feature_names, trees):
    return {'backend': args.backend, 'device': args.device, 'backend_version': args.version,
            'seed': SEED, 'threads': args.threads, 'final_n_estimators': trees,
            'feature_names': list(feature_names),
            'effective_device': fitted_device(model.booster) if args.backend == 'xgboost' else 'cpu'}


def nvidia_processes():
    executable = shutil.which('nvidia-smi')
    if executable is None:
        return None
    output = subprocess.run(
        [executable, '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=8, check=False,
    )
    return {'python_pid': os.getpid(), 'returncode': output.returncode,
            'processes': output.stdout.strip().splitlines(), 'error': output.stderr.strip()}


def synthetic(args):
    rng = np.random.default_rng(SEED)
    names = ['station_idx', *(f'weather_{column}' for column in range(23))]
    X = rng.normal(size=(50_000, len(names))).astype(np.float32)
    X[:, 0] = rng.integers(0, 12, size=len(X))
    y = 0.5 * X[:, 0] + 2 * X[:, 1] - X[:, 2]**2
    weights = np.ones(len(X), dtype=np.float32)
    weights[::2] = 0.5
    params = {'objective': 'regression', 'n_estimators': ROUNDS, 'learning_rate': 0.05,
              'num_leaves': 31, 'n_jobs': args.threads, 'random_state': SEED}
    start = monotonic()
    model = train_tree(X, y, names, params, categorical_features=['station_idx'], sample_weight=weights,
                       backend=args.backend, device=args.device)
    elapsed = monotonic() - start
    prediction = predict_tree(model, X[:100], backend=args.backend)
    if not np.isfinite(prediction).all():
        raise ValueError('Synthetic predictions contain non-finite values.')
    gpu_processes = nvidia_processes() if args.device == 'cuda' else None
    path = args.out / 'tree_synthetic.joblib'
    save_tree_model(model, path)
    reloaded = load_tree_model(path)
    np.testing.assert_allclose(predict_tree(reloaded, X[:100], backend=args.backend), prediction,
                               rtol=1e-7, atol=1e-7)
    trees = best_iteration(model, backend=args.backend, fallback=ROUNDS)
    if trees != ROUNDS:
        raise ValueError('The fixed-round synthetic refit did not create all requested trees.')
    return dict(model_info(model, args, names, trees), rows=len(X), columns=len(names), fit_seconds=elapsed,
                serialization_verified=True, gpu_processes=gpu_processes)


def fit_station_factors(stations, predicted, actual):
    ratio = np.asarray(predicted, dtype=float) / np.asarray(actual, dtype=float)
    if not (np.isfinite(ratio) & (ratio >= 0)).all():
        raise ValueError('Calibration requires finite nonnegative ratios and positive targets.')
    frame = pd.DataFrame({'station_id': np.asarray(stations), 'ratio': ratio, 'ratio_sq': ratio**2})
    sums = frame.groupby('station_id', sort=True).sum()
    factors = (sums['ratio'] / sums['ratio_sq'].where(sums['ratio_sq'] > 0)).clip(0.6, 1.2).dropna()
    return {str(station): float(factor) for station, factor in factors.items()}


def positive_energy(model, X, backend):
    with np.errstate(over='raise', invalid='raise'):
        energy = np.exp(predict_tree(model, X, backend=backend).astype(float))
    if not (np.isfinite(energy) & (energy > 0)).all():
        raise ValueError('Log-energy prediction is outside a finite positive range.')
    return energy


def t2_matrix(rows, features, backend):
    if backend == 'lightgbm':
        return rows[features]
    return np.column_stack([
        rows[name].cat.codes.to_numpy() if name == 'station_id' else rows[name].to_numpy(dtype=np.float32)
        for name in features
    ]).astype(np.float32, copy=False)


def fit_t2_smoke_model(args, fit_rows, features, params, *, stop_rows=None):
    validation = {}
    if stop_rows is not None:
        validation = {'X_valid': t2_matrix(stop_rows, features, args.backend), 'y_valid': np.log(stop_rows['energy'])}
    return train_tree(t2_matrix(fit_rows, features, args.backend), np.log(fit_rows['energy']), features, params,
                      categorical_features=['station_id'], early_stopping_rounds=5,
                      sample_weight=np.ones(len(fit_rows)), backend=args.backend, device=args.device, **validation)


def t2(args):
    from netCDF4 import Dataset

    all_train, all_valid = date_roles(SPLIT_PATH, 'A')
    train_days, forecast_days = all_train[:20], all_valid[:10]
    folders = [(day.isoformat(), args.data_root / 'train' / f'{day:%Y%m%d}_0000')
               for day in [*train_days, *forecast_days]]
    roster = pd.read_csv(args.data_root / 'submission/T2_dataset/submission_template.csv',
                         usecols=['station_id'], dtype={'station_id': str})['station_id'].drop_duplicates().sort_values()
    stations = pd.read_csv(args.data_root / 'static/power_stations.csv', dtype={'station_id': str})
    stations = roster.to_frame().merge(stations, on='station_id', how='left', validate='one_to_one')
    if len(stations) != 12 or not np.isfinite(stations[['latitude', 'longitude']].to_numpy(dtype=float)).all():
        raise ValueError('T2 smoke requires coordinates for all 12 template stations.')
    weather, cache_version = feature_cache(args.data_root, folders, stations, args.cache, Dataset, print)
    features = [column for column in weather if column != 'date']
    _, eligible = load_daily_targets(folders[:len(train_days)], roster)
    train = weather.merge(eligible, on=KEYS, validate='one_to_one')
    valid = weather.loc[weather['date'].isin([day.isoformat() for day in forecast_days])].copy()
    for rows in (train, valid):
        rows['station_id'] = pd.Categorical(rows['station_id'], categories=roster)
    stop_start, calibration_start = train_days[12].isoformat(), train_days[16].isoformat()
    earlier = train.loc[train['date'].lt(stop_start)]
    stop = train.loc[train['date'].ge(stop_start) & train['date'].lt(calibration_start)]
    before_calibration = train.loc[train['date'].lt(calibration_start)]
    calibration = train.loc[train['date'].ge(calibration_start)]
    if any(rows.empty for rows in (earlier, stop, calibration, valid)):
        raise ValueError('A T2 smoke partition has no rows.')
    params = {'objective': 'regression', 'metric': 'l2', 'n_estimators': ROUNDS, 'learning_rate': 0.04,
              'num_leaves': 15, 'min_child_samples': 40, 'random_state': SEED, 'n_jobs': args.threads}

    start = monotonic()
    print(f'T2 tree selection using {args.backend}/{args.device}', flush=True)
    selected = fit_t2_smoke_model(args, earlier, features, params, stop_rows=stop)
    best = best_iteration(selected, backend=args.backend, fallback=ROUNDS)
    del selected
    params['n_estimators'] = best
    print(f'T2 calibration fit with {best} trees', flush=True)
    calibration_model = fit_t2_smoke_model(args, before_calibration, features, params)
    calibration_pred = positive_energy(calibration_model, t2_matrix(calibration, features, args.backend), args.backend)
    factors = fit_station_factors(calibration['station_id'], calibration_pred, calibration['energy'])
    del calibration_model
    print(f'T2 refit with {best} trees', flush=True)
    model = fit_t2_smoke_model(args, train, features, params)
    fit_seconds = monotonic() - start
    save_tree_model(model, args.out / 'tree_T2.joblib')

    raw = positive_energy(model, t2_matrix(valid, features, args.backend), args.backend)
    multiplier = valid['station_id'].astype(str).map(factors).fillna(1.0).to_numpy(dtype=float)
    cap = MAX_PREDICTION['T2']
    forecasts = {'raw': valid[KEYS].assign(energy=np.clip(raw, 0, cap)),
                 'calibrated': valid[KEYS].assign(energy=np.clip(raw * multiplier, 0, cap))}
    for name, forecast in forecasts.items():
        forecast.to_csv(args.out / f'validation_{name}.csv', index=False)
    truth, _ = load_daily_targets(folders[len(train_days):], roster)
    scores = {name: score_track('T2', truth, forecast, roster, allow_unscored_stations=True)
              for name, forecast in forecasts.items()}
    return dict(model_info(model, args, features, best), smoke_features='independent_t2',
                train_days=[day.isoformat() for day in train_days],
                forecast_days=[day.isoformat() for day in forecast_days],
                early_stop_start=stop_start, calibration_start=calibration_start,
                calibration_rows=len(calibration), station_factors=factors,
                identity_stations=sorted(set(roster) - set(factors)), weather_cache_version=cache_version,
                scores=scores, fit_seconds=fit_seconds,
                energy_ranges={name: [float(frame.energy.min()), float(frame.energy.max())]
                               for name, frame in forecasts.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('synthetic', 't2'))
    parser.add_argument('--backend', choices=('lightgbm', 'xgboost'), default='xgboost')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--expect-version')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error('Use 1 to 4 threads for smoke tests.')
    if (args.out / 'results.json').exists():
        parser.error('Smoke results already exist; choose a fresh output directory.')
    args.version = check_backend(args.backend, args.device)
    if args.expect_version is not None and args.version != args.expect_version:
        parser.error(f'Expected {args.backend} {args.expect_version}, found {args.version}.')
    args.cache = args.cache or args.out
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'command.txt').write_text(subprocess.list2cmdline([sys.executable, *sys.argv]) + '\n')
    write_json(args.out / 'env.json', {'python': sys.version, 'backend': args.backend, 'device': args.device,
                                      'backend_version': args.version, 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
                                      'thread_environment': {name: os.getenv(name) for name in THREAD_LIMITS}})
    status = args.out / 'status.json'
    write_json(status, {'status': 'running', 'case': args.case, 'backend': args.backend, 'device': args.device})
    try:
        result = {'synthetic': synthetic, 't2': t2}[args.case](args)
        write_json(args.out / 'results.json', result)
    except (Exception, KeyboardInterrupt) as error:
        write_json(status, {'status': 'failed', 'case': args.case, 'backend': args.backend, 'device': args.device,
                            'error': str(error)})
        raise
    write_json(status, {'status': 'complete', 'case': args.case, 'backend': args.backend, 'device': args.device})
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
