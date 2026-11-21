#!/usr/bin/env python3
"""Compare two frozen T1 development forecasts on the fixed warm split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from score_forecasts import CAPACITY_MW, DAY_KEYS, MAX_PREDICTION, MIN_VALID_POINTS, POINT_KEYS, align_predictions, score_track
from warm_shared_power import FROZEN_SPLIT, labels, load_roster, manifest_days


def site_scores(truth, forecasts):
    valid = np.isfinite(truth['power'].to_numpy(dtype=float)) & truth['power_limit'].eq(1).to_numpy()
    keys = truth.loc[valid, DAY_KEYS]
    daily_count = keys.groupby(DAY_KEYS, sort=False).size()
    eligible = daily_count.ge(MIN_VALID_POINTS)
    results = {}
    for name, forecast in forecasts.items():
        prediction = align_predictions(truth, forecast, POINT_KEYS, 'prediction', MAX_PREDICTION['T1'])
        squared_error = ((prediction[valid] - truth.loc[valid, 'power'].to_numpy()) / CAPACITY_MW)**2
        daily_mse = keys.assign(squared_error=squared_error).groupby(DAY_KEYS, sort=False)['squared_error'].mean()
        daily = 1.0 - np.sqrt(daily_mse.loc[eligible])
        results[name] = (100 * daily.groupby(level='station_id').mean()).to_dict()
    return {
        station: {'mlp': score, 'companion': results['companion'][station],
                  'mlp_minus_companion': score - results['companion'][station]}
        for station, score in results['mlp'].items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--split', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/warm_date_split_v1.csv')
    parser.add_argument('--mlp', type=Path, default=Path('/workspace/qhl/runs/warm_shared_power_v1/T1_development.csv'))
    parser.add_argument('--companion', type=Path, required=True, help='CSV with station_id,date,point_index,prediction.')
    args = parser.parse_args()
    development = manifest_days(args.split)['development']
    roster = load_roster(args.data_root)
    paths = {'mlp': args.mlp, 'companion': args.companion}
    digests = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    forecasts = {name: pd.read_csv(path, dtype={'station_id': str, 'date': str}) for name, path in paths.items()}
    truth = labels(args.data_root, development, roster)
    scores = {name: score_track('T1', truth, frame, roster) for name, frame in forecasts.items()}
    per_site = site_scores(truth, forecasts)
    score_fields = ('readme_unclipped_score', 'pdf_clipped_score')
    report = {
        'data_root': str(args.data_root), 'split_sha256': FROZEN_SPLIT, 'development_days': len(development),
        'prediction_sha256': digests, 'scores': scores,
        'mlp_minus_companion': {key: scores['mlp'][key] - scores['companion'][key] for key in score_fields},
        'per_station_readme': per_site,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
