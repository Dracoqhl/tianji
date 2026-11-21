#!/usr/bin/env python3
"""Score labeled local wind-power validation, using both documented conventions."""

import argparse
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd


CAPACITY_MW = 20.0855
MIN_VALID_POINTS = 72
MIN_ENERGY_MWH = 24.1026
MAX_PREDICTION = {'T1': 24.1026, 'T2': 578.4635, 'T3': 24.1026}
STATION_COUNTS = {'T1': 12, 'T2': 12, 'T3': 6}
DAY_KEYS = ['station_id', 'date']
POINT_KEYS = [*DAY_KEYS, 'point_index']
LABEL_FIELDS = [*POINT_KEYS, 'power', 'power_limit']


def point_indices(frame):
    index = pd.to_numeric(frame['point_index'], errors='raise')
    if not (index.notna() & index.between(1, 96) & (index % 1 == 0)).all():
        raise ValueError('point_index must be an integer from 1 to 96.')
    frame['point_index'] = index.astype('int64')


def align_predictions(truth_keys, predictions, keys, value, maximum):
    predictions = predictions[[*keys, value]].copy()
    if 'point_index' in keys:
        point_indices(predictions)
    index = pd.MultiIndex.from_frame(predictions[keys])
    expected = pd.MultiIndex.from_frame(truth_keys[keys])
    if index.has_duplicates or len(index.difference(expected)) or len(expected.difference(index)):
        raise ValueError('Predictions must cover the exact local validation keys once each.')
    numbers = pd.to_numeric(predictions[value], errors='raise').to_numpy(dtype=float)
    if not (np.isfinite(numbers) & (numbers >= 0) & (numbers <= maximum)).all():
        raise ValueError(f'{value} must be finite and between 0 and {maximum}.')
    return pd.Series(numbers, index=index).reindex(expected).to_numpy()


def score_track(track, labels, predictions, target_stations):
    """Inputs use labeled YYYY-MM-DD dates and the station roster from the track template."""
    stations = set(target_stations)
    if len(stations) != STATION_COUNTS[track] or any(pd.isna(station) or station == '' for station in stations):
        raise ValueError(f'{track} requires {STATION_COUNTS[track]} stations from the submission template.')
    truth = labels.loc[labels['station_id'].isin(stations), LABEL_FIELDS].copy()
    if set(truth['station_id']) != stations or truth['date'].isna().any():
        raise ValueError('Validation labels must contain dated rows for every target station.')
    point_indices(truth)
    day_group = truth.groupby(DAY_KEYS, sort=False)
    if truth.duplicated(POINT_KEYS).any() or not day_group.size().eq(96).all():
        raise ValueError('Validation labels must have one full 96-point grid per target station-day.')
    valid = pd.to_numeric(truth['power_limit'], errors='raise').eq(1)
    truth['power'] = pd.to_numeric(truth['power'], errors='raise')
    finite = np.isfinite(truth['power'].to_numpy(dtype=float, na_value=np.nan))
    if (truth['power'].notna() & ~finite).any():
        raise ValueError('Nonmissing validation power must be finite.')
    valid &= finite
    eligible = truth.loc[valid].copy()
    valid_groups = eligible.groupby(DAY_KEYS, sort=False)['power']
    count = valid_groups.size()
    enough = count.ge(MIN_VALID_POINTS)
    days_with_enough_points = int(enough.sum())

    if track == 'T2':
        pred_keys = truth[DAY_KEYS].drop_duplicates()
        pred_keys['prediction'] = align_predictions(pred_keys, predictions, DAY_KEYS, 'energy', MAX_PREDICTION[track])
        actual = valid_groups.mean().loc[enough] * 24.0
        # Allow floating-point roundoff at exact equality with the published threshold.
        scored = actual.loc[actual.ge(MIN_ENERGY_MWH) | np.isclose(actual, MIN_ENERGY_MWH, atol=1e-12, rtol=0)]
        forecast = pred_keys.set_index(DAY_KEYS)['prediction'].reindex(scored.index)
        residual_sq = ((forecast - scored) / scored) ** 2
        by_station = residual_sq.groupby(level='station_id')
        readme = 1.0 - np.sqrt(by_station.mean())
        pdf = readme.clip(lower=0.0)
        scored_counts = by_station.size()
        changed_unit_count = int(readme.lt(0).sum())
        clipped_unit = 'station'
    else:
        truth['prediction'] = align_predictions(truth, predictions, POINT_KEYS, 'prediction', MAX_PREDICTION[track])
        loss = ((truth.loc[valid, 'prediction'] - truth.loc[valid, 'power']) / CAPACITY_MW) ** 2
        daily = 1.0 - np.sqrt(loss.groupby([eligible[key] for key in DAY_KEYS], sort=False).mean().loc[enough])
        readme = daily.groupby(level='station_id').mean()
        pdf = daily.clip(lower=0.0).groupby(level='station_id').mean()
        scored_counts = daily.groupby(level='station_id').size()
        changed_unit_count = int(daily.lt(0).sum())
        clipped_unit = 'station_day'

    unscored_stations = stations.difference(readme.index)
    if unscored_stations:
        raise ValueError(f'{len(unscored_stations)} target stations have no eligible validation day; choose a window with coverage for all target stations.')
    return {
        'track': track,
        'readme_unclipped_score': 100.0 * float(readme.mean()),
        'pdf_clipped_score': 100.0 * float(pdf.mean()),
        'pdf_clip_unit': clipped_unit,
        'pdf_clip_changed_units': changed_unit_count,
        'validation_station_days': len(truth) // 96,
        'station_days_at_least_72_valid': days_with_enough_points,
        'scored_station_days': int(scored_counts.sum()),
        'scored_days_per_station': {str(key): int(value) for key, value in scored_counts.items()},
    }


def load_labels(path, start, end):
    columns = {'station_id': str, 'date': str}
    if path.is_dir():
        if start is None or end is None:
            raise ValueError('Use --start and --end with a train directory.')
        first = date.fromisoformat(start).strftime('%Y%m%d')
        last = date.fromisoformat(end).strftime('%Y%m%d')
        folders = [folder for folder in sorted(path.glob('????????_0000'))
                   if folder.is_dir() and first <= folder.name[:8] <= last]
        if not folders:
            raise ValueError('No daily label files for the requested validation window.')
        return pd.concat([pd.read_csv(folder / 'power_target.csv', dtype=columns, usecols=LABEL_FIELDS)
                          for folder in folders], ignore_index=True)
    labels = pd.read_csv(path, dtype=columns, usecols=LABEL_FIELDS)
    if start is not None:
        labels = labels.loc[labels['date'].ge(start)]
    if end is not None:
        labels = labels.loc[labels['date'].le(end)]
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', choices=STATION_COUNTS, required=True)
    parser.add_argument('--labels', type=Path, required=True, help='Local label CSV or train directory.')
    parser.add_argument('--predictions', type=Path, required=True, help='Local prediction CSV with date, never date_code.')
    parser.add_argument('--template', type=Path, required=True, help='Track submission template, used only for station IDs.')
    parser.add_argument('--start', help='First validation date, inclusive YYYY-MM-DD.')
    parser.add_argument('--end', help='Last validation date, inclusive YYYY-MM-DD.')
    args = parser.parse_args()
    labels = load_labels(args.labels, args.start, args.end)
    predictions = pd.read_csv(args.predictions, dtype={'station_id': str, 'date': str})
    roster = pd.read_csv(args.template, usecols=['station_id'], dtype={'station_id': str})['station_id'].unique()
    print(json.dumps(score_track(args.track, labels, predictions, roster), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
