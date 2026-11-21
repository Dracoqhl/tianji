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


SCORE_FIELDS = ('readme_unclipped_score', 'pdf_clipped_score')
PRIMARY_SCORE = SCORE_FIELDS[0]
MINIMUM_GAIN = 0.05


def daily_scores(truth, forecast):
    valid = np.isfinite(truth['power'].to_numpy(dtype=float)) & truth['power_limit'].eq(1).to_numpy()
    keys = truth.loc[valid, DAY_KEYS]
    prediction = align_predictions(truth, forecast, POINT_KEYS, 'prediction', MAX_PREDICTION['T1'])
    squared_error = ((prediction[valid] - truth.loc[valid, 'power'].to_numpy()) / CAPACITY_MW)**2
    losses = keys.assign(squared_error=squared_error).groupby(DAY_KEYS, sort=False)['squared_error']
    return 1.0 - np.sqrt(losses.mean().loc[losses.size().ge(MIN_VALID_POINTS)])


def site_scores(truth, forecasts):
    results = {name: (100 * daily_scores(truth, forecast).groupby(level='station_id').mean()).to_dict()
               for name, forecast in forecasts.items()}
    return {
        station: {'mlp': score, 'companion': results['companion'][station],
                  'mlp_minus_companion': score - results['companion'][station]}
        for station, score in results['mlp'].items()
    }


def compare_pair(truth, baseline, candidate, roster, *, development=False):
    scores = {name: score_track('T1', truth, forecast, roster)
              for name, forecast in (('baseline', baseline), ('candidate', candidate))}
    daily = {name: daily_scores(truth, forecast)
             for name, forecast in (('baseline', baseline), ('candidate', candidate))}
    by_station = pd.DataFrame({name: 100 * values.groupby(level='station_id').mean()
                               for name, values in daily.items()})
    by_station['delta'] = by_station['candidate'] - by_station['baseline']
    result = {
        'scores': scores,
        'delta': {key: scores['candidate'][key] - scores['baseline'][key] for key in SCORE_FIELDS},
        'per_station_readme': by_station.to_dict(orient='index'),
    }
    if development:
        blocks = {}
        for month in ('2024-05', '2024-06', '2024-07'):
            block = {}
            for name, values in daily.items():
                selected = values.loc[values.index.get_level_values('date').str.startswith(month)]
                if selected.empty:
                    raise ValueError(f'No A scoreable station-days in {month}.')
                block[name] = {
                    PRIMARY_SCORE: float(100 * selected.groupby(level='station_id').mean().mean()),
                    SCORE_FIELDS[1]: float(100 * selected.clip(lower=0).groupby(level='station_id').mean().mean()),
                    'scored_station_days': len(selected),
                    'scored_stations': selected.index.get_level_values('station_id').nunique(),
                    'unscored_stations': sorted(set(roster) - set(selected.index.get_level_values('station_id'))),
                }
            block['delta'] = {key: block['candidate'][key] - block['baseline'][key] for key in SCORE_FIELDS}
            blocks[month] = block
        result['blocks'] = blocks
    return result


def diagnostic_errors(truth, baseline, candidate, component, values=None):
    """Point RMSE in predefined groups, restricted to scoreable station-days."""
    truth = truth.reset_index(drop=True)
    valid = np.isfinite(truth['power'].to_numpy(dtype=float)) & truth['power_limit'].eq(1).to_numpy()
    keys = truth.loc[valid, DAY_KEYS]
    eligible = keys.groupby(DAY_KEYS, sort=False)['date'].transform('size').ge(MIN_VALID_POINTS)
    scoreable = keys.loc[eligible]
    point_keys = truth.loc[scoreable.index, POINT_KEYS]
    actual = truth.loc[scoreable.index, 'power'].to_numpy(dtype=float)
    if component == 'station_day_weight':
        days = scoreable.groupby('station_id')['date'].transform('nunique').to_numpy()
        groups = np.select([days <= 14, days <= 28], ['1-14_scored_days', '15-28_scored_days'],
                           default='29-42_scored_days')
    else:
        diagnostic = candidate[POINT_KEYS].assign(value=values).set_index(POINT_KEYS)['value']
        data = diagnostic.reindex(pd.MultiIndex.from_frame(point_keys)).to_numpy()
        if component == 'gpm_missing':
            groups = np.select([data == 5, data >= 6, data == 4],
                               ['all_missing', 'final_missing', 'complete'],
                               default='earlier_missing')
        else:
            groups = np.select([~np.isfinite(data), data < 1, data < 2],
                               ['missing', 'below_1_mps', '1_to_2_mps'], default='at_least_2_mps')

    errors = {}
    for name, forecast in (('baseline', baseline), ('candidate', candidate)):
        aligned = align_predictions(truth, forecast, POINT_KEYS, 'prediction', MAX_PREDICTION['T1'])
        errors[name] = np.square(aligned[scoreable.index] - actual)
    result = {}
    for group in np.unique(groups):
        mask = groups == group
        left = float(np.sqrt(errors['baseline'][mask].mean()))
        right = float(np.sqrt(errors['candidate'][mask].mean()))
        result[group] = {'points': int(mask.sum()), 'baseline_rmse_mw': left,
                         'candidate_rmse_mw': right, 'candidate_minus_baseline_rmse_mw': right - left}
    return result


def b_statistics(deltas, ensemble_delta):
    from scipy.stats import t

    deltas = np.asarray(deltas, dtype=float)
    if deltas.shape != (3,) or not np.isfinite(deltas).all() or not np.isfinite(ensemble_delta):
        raise ValueError('B acceptance requires three finite paired seed differences and an ensemble difference.')
    mean = float(deltas.mean())
    deviation = float(deltas.std(ddof=1))
    if np.all(deltas == deltas[0]):
        mean, deviation = float(deltas[0]), 0.0
    half_width = float(t.ppf(0.975, df=2) * deviation / np.sqrt(3))
    low, high = mean - half_width, mean + half_width
    robust_gain = mean - 0.5 * deviation
    return {'mean_delta': mean, 'sample_sd': deviation, 'paired_t_95_interval': [low, high],
            'identical_seed_deltas': deviation == 0, 'mean_minus_half_sd': robust_gain,
            'interval_passed': low > 0, 'robust_gain_passed': robust_gain >= MINIMUM_GAIN,
            'ensemble_delta': float(ensemble_delta), 'ensemble_passed': ensemble_delta > 0,
            'accepted': (low > 0 or robust_gain >= MINIMUM_GAIN) and ensemble_delta > 0}


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
    report = {
        'data_root': str(args.data_root), 'split_sha256': FROZEN_SPLIT, 'development_days': len(development),
        'prediction_sha256': digests, 'scores': scores,
        'mlp_minus_companion': {key: scores['mlp'][key] - scores['companion'][key] for key in SCORE_FIELDS},
        'per_station_readme': per_site,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
