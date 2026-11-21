#!/usr/bin/env python3
"""Call the server-side companion trainer on the fixed warm T1 split."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import pandas as pd

from score_forecasts import CAPACITY_MW, MAX_PREDICTION
from warm_shared_power import FROZEN_SPLIT, labels, load_roster, manifest_days


def cache_paths(workdir, days):
    paths = [workdir / 'features' / 'train' / f'{day:%Y%m%d}_0000.npz' for day in days]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)} selected daily feature archives, e.g. {missing[:3]}.')
    return paths


def row_positions(meta, station_indices, n_days, *, phase, require_full, feature_rows):
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
    if stats['duplicate_rows'] or (require_full and stats['missing_grid_keys']):
        raise ValueError(f'{phase}: duplicate row keys, or incomplete prediction grid. Key counts printed above.')
    return positions, days, stats


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


def run(args):
    sys.path.insert(0, str(args.companion_code / 'src'))
    from tianji.config import load_config
    from tianji.constants import ALL_STATIONS, REGULAR_IDX
    from tianji.data.dataset import build_gbdt_rows, gbdt_feature_names, load_feature_spec
    from tianji.models.gbdt import predict, save_model, train_lgbm

    split = manifest_days(args.split)
    train_days, dev_days = split['train'], split['development']
    if not 1 <= args.early_stop_days < len(train_days):
        raise ValueError('Early-stopping days must be a proper subset of training days.')
    stations = list(REGULAR_IDX)
    roster = [ALL_STATIONS[index] for index in stations]
    if set(roster) != set(load_roster(args.data_root)):
        raise ValueError('The companion station roster differs from the qhl submission template.')
    train_paths = cache_paths(args.workdir, train_days)
    dev_paths = cache_paths(args.workdir, dev_days)
    start = monotonic()
    # Check the development grid before the large training assembly and before fitting.
    print(f'Checking prediction keys for {len(dev_paths)} development dates', flush=True)
    dev_inputs, _, dev_metadata = build_gbdt_rows(dev_paths, stations, CAPACITY_MW,
                                                include_invalid=True, verbose=True)
    dev_positions, _, dev_key_stats = row_positions(dev_metadata, stations, len(dev_days),
                                                    phase='development', require_full=True, feature_rows=len(dev_inputs))
    del dev_metadata
    if args.check_only:
        sample_indices = sorted({0, len(train_days) // 2, len(train_days) - 1})
        sample_days = [train_days[i] for i in sample_indices]
        sample_paths = [train_paths[i] for i in sample_indices]
        check_matrix, _, check_meta = build_gbdt_rows(sample_paths, stations, CAPACITY_MW,
                                                      include_invalid=True, verbose=False)
        check_positions, _, _ = row_positions(check_meta, stations, len(sample_paths),
                                               phase='sampled training', require_full=False, feature_rows=len(check_matrix))
        training_targets(args.data_root, sample_days, roster, check_positions, 'sampled training')
        print('Development grid and sampled training coverage passed.', flush=True)
        return
    cfg = load_config(str(args.config or args.companion_code / 'configs' / 'default.json'))
    spec_path = args.workdir / 'stats' / 'feature_spec_train.json'
    feature_names = gbdt_feature_names(load_feature_spec(args.workdir, 'train'))
    spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    params = dict(cfg['gbdt'])
    if args.threads:
        params['n_jobs'] = args.threads

    print(f'Reading training features for {len(train_paths)} selected dates', flush=True)
    weather, _, metadata = build_gbdt_rows(train_paths, stations, CAPACITY_MW, include_invalid=True, verbose=True)
    positions, day_idx, train_key_stats = row_positions(metadata, stations, len(train_days),
                                                       phase='training', require_full=False, feature_rows=len(weather))
    target, valid = training_targets(args.data_root, train_days, roster, positions, 'training')
    del positions, metadata
    inputs = np.ascontiguousarray(weather[valid], dtype=np.float32)
    del weather
    inner = day_idx[valid] >= len(train_days) - args.early_stop_days
    if not inner.any() or inner.all():
        raise ValueError('Training or inner early-stopping region has no valid power points.')
    assemble_seconds = monotonic() - start

    fit_start = monotonic()
    print(f'Selecting iterations inside the {len(train_days)} training dates', flush=True)
    selection = train_lgbm(inputs[~inner], target[~inner], feature_names, params,
                           X_valid=inputs[inner], y_valid=target[inner], categorical_features=['station_idx'])
    best = int(getattr(selection, 'best_iteration_', 0) or params['n_estimators'])
    if best < 1:
        raise ValueError('The companion trainer returned an invalid iteration budget.')
    del selection
    fitted = train_lgbm(inputs, target, feature_names, dict(params, n_estimators=best),
                        categorical_features=['station_idx'])
    fit_seconds = monotonic() - fit_start
    save_model(fitted, args.out / 'companion_T1.joblib')
    del inputs

    prediction_start = monotonic()
    print(f'Predicting the {len(dev_paths)} frozen development dates', flush=True)
    output = np.empty(len(dev_positions), dtype=float)
    prediction = np.asarray(predict(fitted, dev_inputs), dtype=float).reshape(-1) * CAPACITY_MW
    if not np.isfinite(prediction).all() or len(prediction) != len(output):
        raise ValueError('Invalid forecast length or non-finite prediction.')
    output[dev_positions] = np.clip(prediction, 0, MAX_PREDICTION['T1'])
    keys = pd.MultiIndex.from_product([[d.isoformat() for d in dev_days], roster, range(1, 97)],
                                      names=['date', 'station_id', 'point_index']).to_frame(index=False)
    keys['prediction'] = output
    forecast_csv = args.out / 'companion_T1_development.csv'
    keys.to_csv(forecast_csv, index=False)

    result = {
        'data_root': str(args.data_root), 'feature_workdir': str(args.workdir),
        'feature_spec_sha256': spec_hash, 'split_sha256': FROZEN_SPLIT,
        'train_days': len(train_paths), 'development_days': len(dev_paths),
        'early_stopping_days_inside_train': args.early_stop_days, 'train_valid_rows': len(target),
        'feature_keys': {'training': train_key_stats, 'development': dev_key_stats},
        'early_stopping_valid_rows': int(inner.sum()), 'best_iteration': best,
        'feature_count': len(feature_names), 'prediction_rows': len(keys),
        'clipped_predictions': int(((prediction < 0) | (prediction > MAX_PREDICTION['T1'])).sum()),
        'prediction_sha256': hashlib.sha256(forecast_csv.read_bytes()).hexdigest(),
        'phase_seconds': {'assembly': assemble_seconds, 'fit': fit_seconds,
                          'prediction_and_export': monotonic() - prediction_start},
    }
    (args.out / 'training_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {forecast_csv}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--companion-code', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True, help='Existing verified companion feature workdir in the server.')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--out', type=Path, default=Path('/workspace/qhl/runs/companion_warm_t1_v1'))
    parser.add_argument('--split', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/warm_date_split_v1.csv')
    parser.add_argument('--early-stop-days', type=int, default=90)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--check-only', action='store_true', help='Check the development grid and three training days, without fitting.')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive.')
    if not args.check_only:
        args.out.mkdir(parents=True, exist_ok=True)
        if (args.out / 'companion_T1_development.csv').exists():
            parser.error('Frozen companion predictions already exist. Choose a new output directory for a rerun.')
    run(args)


if __name__ == '__main__':
    main()
