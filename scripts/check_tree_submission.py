#!/usr/bin/env python3
"""Check generated T1/T2/T3 CSV files against their submission templates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from score_forecasts import MAX_PREDICTION, STATION_COUNTS, point_indices


COMPETITION_DAYS = 92


def check_candidate(track, path, template_path):
    value = 'energy' if track == 'T2' else 'prediction'
    keys = ['station_id', 'date_code'] + ([] if track == 'T2' else ['point_index'])
    columns = [*keys, value]
    dtype = {'station_id': str, 'date_code': str}
    expected = pd.read_csv(template_path, dtype=dtype)
    candidate = pd.read_csv(path, dtype=dtype)
    stations, points = STATION_COUNTS[track], 1 if track == 'T2' else 96
    if list(expected.columns) != columns or list(candidate.columns) != columns:
        raise ValueError(f'{track}: expected columns in this order: {columns}.')
    if len(candidate) != len(expected) or len(expected) != stations * COMPETITION_DAYS * points:
        raise ValueError(f'{track}: candidate or template row count does not match the competition grid.')
    if track != 'T2':
        point_indices(expected)
        point_indices(candidate)
    if (expected[keys].isna().any().any() or expected.duplicated(keys).any() or
            expected['station_id'].nunique() != stations or expected['date_code'].nunique() != COMPETITION_DAYS or
            not expected.groupby(['station_id', 'date_code']).size().eq(points).all()):
        raise ValueError(f'{track}: template has invalid station-day keys.')
    if not expected[keys].equals(candidate[keys]):
        raise ValueError(f'{track}: candidate keys or row order differ from the template.')
    values = pd.to_numeric(candidate[value], errors='raise').to_numpy(dtype=float)
    if not (np.isfinite(values) & (values >= 0) & (values <= MAX_PREDICTION[track])).all():
        raise ValueError(f'{track}: every {value} must be finite and in [0, {MAX_PREDICTION[track]}].')
    return {'rows': len(candidate), 'columns': columns, 'stations': stations, 'days': COMPETITION_DAYS,
            'prediction_range': [float(values.min()), float(values.max())], 'exact_template_key_order': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    for track in MAX_PREDICTION:
        parser.add_argument(f'--{track.lower()}', type=Path)
    args = parser.parse_args()
    paths = {track: getattr(args, track.lower()) for track in MAX_PREDICTION if getattr(args, track.lower())}
    if not paths:
        parser.error('Specify at least one candidate CSV with --t1, --t2 or --t3.')
    reports = {track: check_candidate(track, path,
                                      args.data_root / 'submission' / f'{track}_dataset/submission_template.csv')
               for track, path in paths.items()}
    print(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
