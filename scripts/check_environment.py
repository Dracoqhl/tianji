#!/usr/bin/env python3
"""Print a read-only weather/power competition environment inventory as JSON.

Run with Python 3.8+ and pass the competition data directory as --data-root.
The script reports dependency versions, hardware limits, dataset file presence,
CSV headers/counts and the metadata of one training day's forecast files.
"""

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


GIB = 1024 ** 3
PACKAGES = (
    'numpy', 'pandas', 'scipy', 'xarray', 'netCDF4', 'h5netcdf', 'h5py',
    'dask', 'pyarrow', 'scikit-learn', 'lightgbm', 'catboost', 'xgboost', 'torch',
)
FORECASTS = ('tianji_0p125.nc', 'ec_hres_0p10.nc', 'aifs_0p25.nc')
REQUIRED_INPUTS = tuple('nwp/' + filename for filename in FORECASTS) + (
    'gpm/gpm_input.nc', 'fy4b/fy4b_tbb13_input.nc', 'obs/obs_input.csv',
)
TRAIN_ONLY = ('power_target.csv', 'gpm/gpm_target.nc',
              'fy4b/fy4b_tbb13_target.nc', 'obs/obs_target.csv')


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def memory_gib():
    info = read_text('/proc/meminfo') or ''
    match = re.search(r'^MemTotal:\s+(\d+)\s+kB', info, re.MULTILINE)
    return round(int(match.group(1)) * 1024 / GIB, 2) if match else None


def command_output(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=8, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return {'status': type(exc).__name__}
    return {'lines': result.stdout.strip().splitlines()}


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def csv_metadata(path, count_rows=False):
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = csv.reader(stream)
            result = {'columns': next(rows, [])}
            if count_rows:
                result['rows'] = sum(1 for row in rows if row)
            return result
    except (OSError, UnicodeError, csv.Error) as exc:
        return {'status': type(exc).__name__}


def netcdf_metadata(path, versions):
    if not path.is_file():
        return {'status': 'missing'}
    if versions['netCDF4']:
        try:
            import netCDF4

            with netCDF4.Dataset(path, mode='r') as dataset:
                return {'reader': 'netCDF4',
                        'dimensions': {name: len(dim) for name, dim in dataset.dimensions.items()},
                        'variables': sorted(dataset.variables)}
        except Exception as exc:
            netcdf4_error = type(exc).__name__
    else:
        netcdf4_error = 'not_installed'
    if versions['xarray']:
        try:
            import xarray

            with xarray.open_dataset(path, decode_times=False, cache=False) as dataset:
                return {'reader': 'xarray', 'dimensions': dict(dataset.sizes),
                        'variables': sorted(dataset.variables)}
        except Exception as exc:
            return {'status': 'metadata_unavailable', 'netCDF4': netcdf4_error,
                    'xarray': type(exc).__name__}
    return {'status': 'metadata_unavailable', 'netCDF4': netcdf4_error,
            'xarray': 'not_installed'}


def dataset_inventory(root, versions, include_netcdf):
    result = {'root_exists': root.is_dir()}
    if not result['root_exists']:
        return result
    try:
        sections = {}
        for name in ('train', 'test'):
            directory = root / name
            sections[name] = sorted(child for child in directory.iterdir() if child.is_dir()) if directory.is_dir() else []
    except OSError as exc:
        result['status'] = type(exc).__name__
        return result
    train, test = sections['train'], sections['test']
    result['train_sample_directories'] = len(train)
    result['test_sample_directories'] = len(test)
    result['training_date_named_directories'] = sum(
        bool(re.fullmatch(r'\d{8}_0000', directory.name)) for directory in train
    )
    result['official_data_readme_exists'] = (root / '新能源功率预测竞赛_数据说明.md').is_file()
    result['static'] = {
        'terrain_exists': (root / 'static/terrain.nc').is_file(),
        'obs_stations': csv_metadata(root / 'static/obs_stations.csv', True),
        'power_stations': csv_metadata(root / 'static/power_stations.csv', True),
    }
    result['submission'] = {}
    for track in ('T1', 'T2', 'T3'):
        directory = root / 'submission' / (track + '_dataset')
        result['submission'][track] = {
            'scoring_readme_exists': (directory / 'README.md').is_file(),
            'template': csv_metadata(directory / 'submission_template.csv', True),
        }
    if train:
        sample = train[0]
        result['first_training_sample'] = {
            'inputs_present': {name: (sample / name).is_file() for name in REQUIRED_INPUTS},
            'targets_present': {name: (sample / name).is_file() for name in TRAIN_ONLY},
            'power_target': csv_metadata(sample / 'power_target.csv', True),
            'obs_input': csv_metadata(sample / 'obs/obs_input.csv', True),
        }
        if include_netcdf:
            result['first_training_sample']['nwp_metadata'] = {
                name: netcdf_metadata(sample / 'nwp' / name, versions) for name in FORECASTS
            }
    if test:
        sample = test[0]
        result['one_test_sample'] = {
            'inputs_present': {name: (sample / name).is_file() for name in REQUIRED_INPUTS},
            'obs_input': csv_metadata(sample / 'obs/obs_input.csv', True),
        }
    return result


def free_gib(path):
    try:
        return round(shutil.disk_usage(path).free / GIB, 2)
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--scratch-dir', type=Path, default=Path.cwd(),
                        help='Existing location for future experiment artifacts; checks free space.')
    parser.add_argument('--skip-netcdf', action='store_true')
    args = parser.parse_args()
    versions = package_versions()
    gpu = command_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                          '--format=csv,noheader,nounits'])
    print(json.dumps({
        'system': {
            'os': platform.system(), 'machine': platform.machine(),
            'python': platform.python_version(), 'logical_cpus': os.cpu_count(),
            'affinity_cpus': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'visible_system_ram_gib': memory_gib(),
            'cgroup_v2_cpu_max': read_text('/sys/fs/cgroup/cpu.max'),
            'cgroup_v2_memory_max_bytes': read_text('/sys/fs/cgroup/memory.max'),
            'data_mount_free_gib': free_gib(args.data_root),
            'scratch_mount_free_gib': free_gib(args.scratch_dir),
            'gpu_name_memory_mib_driver': gpu,
        },
        'packages': versions,
        'dataset': dataset_inventory(args.data_root, versions, not args.skip_netcdf),
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
