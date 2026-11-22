#!/usr/bin/env python3
"""Test joint pre-issue observation vectors on the frozen T1 and T3 recipes."""
import argparse
from datetime import timedelta
import fcntl
from functools import partial
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
import traceback

import numpy as np
import pandas as pd

from run_companion_warm_t1 import B_SEEDS, SPLIT_PATH, sha256, write_json
from run_forecast_blends import freeze_json
from run_fusion_mechanisms import finish_pending, stage
from run_power_hurdle import t1_predict, t3_study
from screen_companion_t1_stations import show_progress
from t1_issue_wind_error import INPUT_HOURS, near_weights, observation_table, observed_vectors


RUNS = Path('/workspace/qhl/runs')
ARM = 'joint_obs_vector_support'
COLUMNS = ('obs_joint_mean_u_4h', 'obs_joint_mean_v_4h', 'obs_joint_latest_u', 'obs_joint_latest_v',
           'obs_joint_mean_support_4h', 'obs_joint_latest_support', 'obs_joint_age_hours', 'obs_joint_coherence_4h')


def farm_features(vectors, near, weight):
    """Jointly mask u/v and normalize each hourly spatial aggregate."""
    values = vectors[near]
    valid = np.isfinite(values).all(axis=-1)
    weighted = weight[..., None] * valid
    mass = weighted.sum(axis=1)
    numerator = (np.where(valid[..., None], values, 0.) * weighted[..., None]).sum(axis=1)
    hourly = np.divide(numerator, mass[..., None], out=np.zeros_like(numerator), where=mass[..., None] > 0)
    populated = mass > 0
    counted = populated.sum(axis=1, keepdims=True)
    means = np.divide(hourly.sum(axis=1), counted, out=np.full_like(hourly[:, 0], np.nan), where=counted > 0)
    index = np.where(populated, np.arange(len(INPUT_HOURS)), -1).max(axis=1)
    row = np.arange(len(near))
    exists = index >= 0
    last = np.where(exists[:, None], hourly[row, index], np.nan)
    coverage = mass / weight.sum(axis=1, keepdims=True)
    age = np.where(exists, len(INPUT_HOURS) - 1 - index, np.nan)
    speeds = np.linalg.norm(np.where(valid[..., None], values, 0.), axis=-1)
    normalizer = (weighted * speeds).sum(axis=(1, 2))
    magnitude = np.linalg.norm(numerator.sum(axis=1), axis=-1)
    coherence = np.divide(magnitude, normalizer, out=np.full(len(row), np.nan), where=normalizer > 0)
    return np.column_stack((means, last, coverage.mean(axis=1),
                            np.where(exists, coverage[row, index], 0.), age, coherence.clip(0, 1))).astype(np.float32)


class ObsFeatures:
    names = COLUMNS

    def __init__(self, data_root, directory):
        self.root = data_root
        static = data_root / 'static'
        self.sites, sensors = observation_table(data_root)
        farms = pd.read_csv(static / 'power_stations.csv', dtype={'station_id': str}).set_index('station_id').sort_index()
        if len(farms) != 18 or not farms.index.is_unique or not np.isfinite(farms[['latitude', 'longitude']]).all().all():
            raise ValueError('Expected eighteen farms with finite static positions.')
        self.farms = farms.index.tolist()
        self.near, self.weight = near_weights(sensors, farms[['latitude', 'longitude']].to_numpy(dtype=float))
        paths = [Path(__file__), Path(observed_vectors.__code__.co_filename), static / 'power_stations.csv',
                 static / 'obs_stations.csv']
        self.version = hashlib.sha256(''.join(sha256(path) for path in paths).encode()).hexdigest()
        self.cache = directory / self.version
        self.cache.mkdir(parents=True, exist_ok=True)

    def input(self, day):
        path = self.root / 'train' / f'{day:%Y%m%d}_0000/obs/obs_input.csv'
        info = path.stat()
        stamp = f'{day}:{info.st_size}:{info.st_mtime_ns}'
        return path, stamp

    def stamp(self, dates):
        digest = hashlib.sha256('\n'.join(self.input(day)[1] for day in dates).encode()).hexdigest()
        return {'extractor': self.version, 'inputs': digest, 'days': len(dates)}

    def daily(self, day):
        path, stamp = self.input(day)
        digest = hashlib.sha256(stamp.encode()).hexdigest()[:20]
        file = self.cache / f'{day}-{digest}.npy'
        if not file.is_file():
            values = observed_vectors(path, day - timedelta(days=1), self.sites, INPUT_HOURS)
            new = farm_features(values, self.near, self.weight)
            if self.input(day)[1] != stamp:
                raise ValueError(f'{path}: observation file changed during extraction.')
            with NamedTemporaryFile(dir=self.cache, suffix='.npy', delete=False) as temp:
                np.save(temp, new, allow_pickle=False)
            Path(temp.name).replace(file)
        result = np.load(file, allow_pickle=False)
        if result.shape != (len(self.farms), len(COLUMNS)) or np.isinf(result).any():
            raise ValueError(f'{file}: invalid joint observation cache.')
        return result

    def __call__(self, matrix, names, dates, roster, positions):
        if set(COLUMNS) & set(names):
            raise ValueError('Joint observation columns must be new to the tree input.')
        rows = [self.farms.index(site) for site in roster]
        daily = []
        for i, day in enumerate(dates, 1):
            daily.append(self.daily(day)[rows])
            if i == 1 or i % 100 == 0 or i == len(dates):
                print(f'[OBS 联合风矢量] {i}/{len(dates)} days', flush=True)
        values = np.concatenate(daily)[np.asarray(positions) // 96]
        return np.column_stack((matrix, values)), [*names, *COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out': RUNS / 'obs_vector_support_gpu_v1', 'data-root': Path('/workspace/qhl/data'), 'split': SPLIT_PATH,
                'workdir': Path('/workspace/cpy/code/workdir'), 'companion-code': Path('/workspace/cpy/code'),
                'dynamic-run': RUNS / 't1_dynamic_gradient_gpu_v1', 'phase-run': RUNS / 'phase_anchor_cpu_v1',
                'ridge-run': RUNS / 'daily_bias_ridge_gpu_v1', 'fusion-run': RUNS / 'fusion_mechanisms_cpu_v1',
                'short-tree': RUNS / 'three_track_gpu_batch_v1/T3', 'transfer-run': RUNS / 'same_issue_followup_v1/T3',
                't3-confirm': RUNS / 't3_transfer_confirmation_gpu_v1'}
    for name, value in defaults.items():
        parser.add_argument(f'--{name}', type=Path, default=value)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        obs = ObsFeatures(args.data_root, args.out / 'observation_cache')
        freeze_json(args.out / 'plan.json', {
            'args': {name: str(value) for name, value in vars(args).items() if name != 'split'}, 'split': sha256(args.split),
            'seeds': list(B_SEEDS), 'columns': list(COLUMNS), 'weather_extractor': obs.version,
            'code': {name: sha256(Path(__file__).with_name(name)) for name in (
                'run_obs_vector_support.py', 'run_power_hurdle.py', 'run_fusion_mechanisms.py',
                'run_companion_warm_t1.py', 'batch_energy_short.py', 'tree_model.py')}})
        model = {'component': ARM, 'additions': obs, 'fractions': {ARM: 1.}}
        studies = {'T1': (partial(t1_predict, args, **model), (ARM,)), 'T3': t3_study(args, **model)}
        start, result, pending = monotonic(), {}, {}
        for i, (track, study) in enumerate(studies.items()):
            show_progress(args.out, '联合观测风', i, len(studies), track)
            try:
                result[track] = {'status': 'complete', **stage(args, track, pending, study=study)}
            except Exception as error:
                traceback.print_exc()
                result[track] = {'status': 'failed', 'error': str(error)}
            write_json(args.out / 'summary.json', result)
        finish_pending(args, result, pending)
        result['seconds'] = round(monotonic() - start, 2)
        write_json(args.out / 'summary.json', result)
        show_progress(args.out, '联合观测风', len(studies), len(studies), '结果已保存')
        print(f'结果摘要 {args.out / "summary.json"}', flush=True)
        return int(any(result[track]['status'] != 'complete' for track in studies))


if __name__ == '__main__':
    raise SystemExit(main())
