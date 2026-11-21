#!/usr/bin/env python3
"""Audit the wind competition's daily labels and submission keys as aggregate JSON."""

import argparse
from collections import Counter
import csv
from datetime import date
import json
import math
from pathlib import Path
import re


EXPECTED_TRAIN = {'station_id', 'date', 'point_index', 'power', 'wind', 'power_limit'}
TRACKS = {'T1': (12, 96), 'T2': (12, 1), 'T3': (6, 96)}


def locate_dataset(requested):
    candidates = (requested, requested / 'dataset_v2', requested.parent / 'dataset_v2')
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in ('train', 'test', 'static', 'submission')):
            return candidate
    raise SystemExit('Dataset not found. Pass the directory containing train, test, static and submission.')


def parse_power(value, counts):
    if not value:
        counts['power_missing'] += 1
        return None
    try:
        number = float(value)
    except ValueError:
        counts['power_non_finite_or_invalid'] += 1
        return None
    if not math.isfinite(number):
        counts['power_non_finite_or_invalid'] += 1
        return None
    counts['power_zero'] += number == 0
    counts['power_negative'] += number < 0
    return number


def audit_train(days, station_groups):
    counts = Counter()
    station_days = Counter()
    eligibility = {name: Counter() for name in station_groups}
    eligibility['all'] = Counter()
    for folder in days:
        path = folder / 'power_target.csv'
        if not path.is_file():
            counts['files_missing'] += 1
            continue
        try:
            expected_date = date(int(folder.name[:4]), int(folder.name[4:6]), int(folder.name[6:8])).isoformat()
        except ValueError:
            counts['invalid_date_directories'] += 1
            continue
        seen = set()
        valid = Counter()
        points = Counter()
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = csv.DictReader(stream)
            if not EXPECTED_TRAIN.issubset(rows.fieldnames or []):
                counts['files_with_missing_columns'] += 1
                continue
            for row in rows:
                counts['rows'] += 1
                counts['date_mismatch_rows'] += row['date'] != expected_date
                station = (row['station_id'] or '').strip()
                index = (row['point_index'] or '').strip()
                index_ok = index.isascii() and index.isdigit() and 1 <= int(index) <= 96
                key = (station, int(index) if index_ok else index)
                duplicate = key in seen
                seen.add(key)
                counts['duplicate_rows'] += duplicate
                counts['invalid_station_or_point_rows'] += not station or not index_ok
                if not duplicate and station and index_ok:
                    points[station] += 1
                power = parse_power((row['power'] or '').strip(), counts)
                counts['wind_missing'] += not (row['wind'] or '').strip()
                limit = (row['power_limit'] or '').strip()
                counts['limited_rows'] += limit == '0'
                counts['limit_missing'] += limit == ''
                counts['limit_unrecognized'] += limit not in ('', '0', '1')
                if not duplicate and station and index_ok and power is not None and limit == '1':
                    valid[station] += 1
                    counts['valid_unique_points'] += 1
        for station, number in points.items():
            station_days[station] += 1
            for name, members in [('all', None), *station_groups.items()]:
                if members is None or station in members:
                    eligibility[name]['station_days'] += 1
                    eligibility[name]['incomplete_96_point_grid'] += number != 96
                    eligibility[name]['zero_valid_points'] += valid[station] == 0
                    eligibility[name]['one_to_71_valid_points'] += 0 < valid[station] < 72
                    eligibility[name]['at_least_72_valid_points'] += valid[station] >= 72
    return {'counts': counts, 'stations_by_days_with_rows': dict(sorted(Counter(station_days.values()).items())),
            'station_day_valid_point_buckets_observed': eligibility}


def audit_template(root, track, test_ids):
    expected_stations, points_per_day = TRACKS[track]
    path = root / 'submission' / (track + '_dataset') / 'submission_template.csv'
    meta = {'scoring_readme_exists': (path.parent / 'README.md').is_file()}
    if not path.is_file():
        return dict(meta, status='template_missing'), set()
    columns = (['station_id', 'date_code', 'energy'] if track == 'T2'
               else ['station_id', 'date_code', 'point_index', 'prediction'])
    stations, codes, keys = set(), set(), set()
    rows_read = duplicates = invalid_keys = 0
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream)
        meta['columns_exactly_match'] = rows.fieldnames == columns
        if not set(columns).issubset(rows.fieldnames or []):
            return dict(meta, status='missing_columns'), set()
        for row in rows:
            rows_read += 1
            station = row['station_id'] or ''
            code = row['date_code'] or ''
            index = '1' if track == 'T2' else (row['point_index'] or '')
            if not (station and code and index.isascii() and index.isdigit() and 1 <= int(index) <= points_per_day):
                invalid_keys += 1
                continue
            key = (station, code, int(index))
            duplicates += key in keys
            keys.add(key)
            stations.add(station)
            codes.add(code)
    expected_grid_size = expected_stations * len(test_ids) * points_per_day
    grid_ok = (len(stations) == expected_stations and codes == test_ids and
               len(keys) == expected_grid_size and duplicates == invalid_keys == 0)
    return dict(meta, rows=rows_read, stations=len(stations), test_codes=len(codes),
                missing_test_codes=len(test_ids-codes), extra_test_codes=len(codes-test_ids),
                duplicate_rows=duplicates, invalid_keys=invalid_keys, complete_expected_grid=grid_ok), stations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/data'))
    args = parser.parse_args()
    root = locate_dataset(args.data_root)
    days = sorted(folder for folder in (root / 'train').iterdir() if folder.is_dir())
    dated = [folder for folder in days if re.fullmatch(r'[0-9]{8}_0000', folder.name)]
    test_ids = {folder.name for folder in (root / 'test').iterdir() if folder.is_dir()}
    templates, stations = {}, {}
    for track in TRACKS:
        templates[track], stations[track] = audit_template(root, track, test_ids)
    print(json.dumps({
        'data_root': str(root), 'train_directories': len(days), 'training_date_named_directories': len(dated),
        'test_directories': len(test_ids), 'templates': templates,
        'station_set_checks': {'T1_T2_same': bool(stations['T1']) and stations['T1'] == stations['T2'],
                               'T1_T3_disjoint': bool(stations['T1'] and stations['T3']) and stations['T1'].isdisjoint(stations['T3'])},
        'train_labels': audit_train(dated, {'T1_stations': stations['T1'], 'T3_stations': stations['T3']}),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
