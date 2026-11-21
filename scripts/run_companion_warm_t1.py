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


def row_positions(meta, station_indices, n_days):
    station_indices = [int(index) for index in station_indices]
    stations = {index: i for i, index in enumerate(station_indices)}
    days = np.asarray(meta['day_idx'], dtype=np.intp)
    points = np.asarray(meta['point_idx'], dtype=np.intp)
    slots = np.fromiter((stations[int(i)] for i in meta['station_idx']), dtype=np.intp, count=len(days))
    if np.any((days < 0) | (days >= n_days)) or np.any((points < 1) | (points > 96)):
        raise ValueError('Companion feature metadata has out-of-range day or point keys.')
    positions = (days * len(station_indices) + slots) * 96 + points - 1
    expected_count = n_days * len(station_indices) * 96
    if len(positions) != expected_count or np.unique(positions).size != expected_count:
        raise ValueError('The companion all-points loader must return a complete, unique station/day/point grid.')
    return positions, days


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
    cfg = load_config(str(args.config or args.companion_code / 'configs' / 'default.json'))
    spec_path = args.workdir / 'stats' / 'feature_spec_train.json'
    feature_names = gbdt_feature_names(load_feature_spec(args.workdir, 'train'))
    spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    params = dict(cfg['gbdt'])
    if args.threads:
        params['n_jobs'] = args.threads

    start = monotonic()
    print(f'Reading training features for {len(train_paths)} selected dates', flush=True)
    weather, _, metadata = build_gbdt_rows(train_paths, stations, CAPACITY_MW, include_invalid=True, verbose=True)
    positions, day_idx = row_positions(metadata, stations, len(train_days))
    truth = labels(args.data_root, train_days, roster)
    target = truth['power'].to_numpy(dtype=float)[positions]
    valid = np.isfinite(target) & truth['power_limit'].eq(1).to_numpy()[positions]
    del truth, positions, metadata
    inputs = np.ascontiguousarray(weather[valid], dtype=np.float32)
    target = (target[valid] / CAPACITY_MW).astype(np.float32)
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
    dev_inputs, _, dev_metadata = build_gbdt_rows(dev_paths, stations, CAPACITY_MW,
                                                include_invalid=True, verbose=True)
    dev_positions, _ = row_positions(dev_metadata, stations, len(dev_days))
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
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive.')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'companion_T1_development.csv').exists():
        parser.error('Frozen companion predictions already exist. Choose a new output directory for a rerun.')
    run(args)


if __name__ == '__main__':
    main()
