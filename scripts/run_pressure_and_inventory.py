#!/usr/bin/env python3
"""Collect earlier receipts and run the new pressure-profile batch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_companion_warm_t1 import write_json

PRIOR = ('steamlink_followup_gpu_v1', 't1_source_power_gpu_v1')


def summarize_arm(record):
    values = {key: record[key] for key in ('status', 'error', 'B_error') if key in record}
    report = record.get('result', record)
    for key in ('gate', 'A'):
        if key in report:
            values[key] = report[key]
    if isinstance(values.get('A'), dict):
        values['A'] = {name: entry['delta'] if isinstance(entry, dict) and 'delta' in entry else entry
                       for name, entry in values['A'].items()}
    if 'B' in report:
        verification = report['B']
        values['B'] = {key: verification[key] for key in ('scores', 'acceptance', 'seed_deltas') if key in verification}
    if not values:
        values['status'] = 'detail_available'
    return values


def read_result(path):
    if not path.is_file():
        return {'status': 'report_missing'}
    try:
        result = json.loads(path.read_text())
        return result if isinstance(result, dict) else {'status': 'report_unreadable'}
    except (OSError, ValueError):
        return {'status': 'report_unreadable'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args, options = parser.parse_known_args()
    args.out.mkdir(parents=True, exist_ok=True)
    file = args.out / 'batch_overview.json'
    prior = {name: read_result(args.out.parent / name / 'summary.json') for name in PRIOR}
    overview = {'previous': prior}
    write_json(file, overview)
    print('[Batch 1/2] Previous receipts checked. Only the pressure batch starts here.', flush=True)
    for name, result in prior.items():
        if 'status' in result:
            statuses = summarize_arm(result)
        else:
            statuses = {track: summarize_arm(item) for track, item in result.items() if isinstance(item, dict)}
        print(f'[Previous] {name} {json.dumps(statuses, ensure_ascii=False)}', flush=True)

    print('[Batch 2/2] T1 and T3 pressure-profile experiments', flush=True)
    worker = Path(__file__).with_name('run_pressure_profile.py')
    run = subprocess.run([sys.executable, '-u', str(worker), '--out', str(args.out), *options])
    overview['pressure_profile'] = read_result(args.out / 'summary.json')
    overview['pressure_exit_code'] = run.returncode
    write_json(file, overview)
    for track in ('T1', 'T3'):
        score = overview['pressure_profile'].get(track)
        if isinstance(score, dict):
            print(f'[Pressure {track}] {json.dumps(summarize_arm(score), ensure_ascii=False)}', flush=True)
    print(f'[Batch] {file}', flush=True)
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
