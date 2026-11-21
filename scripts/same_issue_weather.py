"""Compact neural inputs from the companion cache, without loading target arrays."""

from pathlib import Path

import numpy as np
import pandas as pd

from run_companion_warm_t1 import cache_paths, cache_version, sha256
from warm_shared_power import load_roster


NWP_FIELDS = tuple(f'{source}_{field}' for source in ('tianji', 'ec', 'aifs')
                   for field in ('u100', 'v100', 'ws100', 't2m', 'wpd', 'sp'))
CONTEXT_FIELDS = ('time_sin', 'time_cos', 'latitude', 'longitude')


def feature_spec(workdir):
    return Path(workdir) / 'stats/feature_spec_train.json'


def archived_issue_weather(workdir, root, days, roster, progress):
    import json

    fields = json.loads(feature_spec(workdir).read_text())['nwp_names']
    if not set(NWP_FIELDS).issubset(fields):
        raise ValueError('The companion cache is missing registered neural weather columns.')
    indices = [fields.index(name) for name in NWP_FIELDS]
    paths = cache_paths(workdir, days)
    before = cache_version(paths)
    coords = pd.read_csv(root / 'static/power_stations.csv', dtype={'station_id': str}).set_index('station_id')
    coordinates = coords.loc[roster, ['latitude', 'longitude']].to_numpy(dtype=np.float32)
    if not np.isfinite(coordinates).all():
        raise ValueError('Forecast roster is missing valid coordinates.')
    angle = 2 * np.pi * np.arange(96) / 96
    phase = np.column_stack([np.sin(angle), np.cos(angle)])
    output = np.empty((len(days), len(roster), 96, len(indices) + len(CONTEXT_FIELDS)), dtype=np.float32)
    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as source:
            stations = source['station_ids'].tolist()
            array = source['nwp_point']
            if (len(stations) != len(set(stations)) or source['date_code'].tolist() != [path.stem]
                    or array.shape != (len(stations), 96, len(fields))):
                raise ValueError(f'{path}: bad weather cache keys or schema.')
            output[i, ..., :len(indices)] = array[[stations.index(site) for site in roster]][:, :, indices]
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(paths):
            progress(f'天气缓存 {i + 1}/{len(paths)}')
    output[..., -4:-2] = phase
    output[..., -2:] = coordinates[:, None]
    if before != cache_version(paths):
        raise ValueError('Weather cache changed during extraction.')
    return output.reshape(-1, 96, output.shape[-1]), [*NWP_FIELDS, *CONTEXT_FIELDS], before


def warm_archive_weather(workdir, root, train_days, forecast_days, cache_root, workers,
                         progress, dataset_factory=None):
    roster = load_roster(root)
    weather, names, version = archived_issue_weather(workdir, root, [*train_days, *forecast_days], roster, progress)
    return roster, (weather, names, len(train_days) + len(forecast_days), version)


def archive_stamp(workdir, days):
    return {'weather': cache_version(cache_paths(workdir, days)), 'feature_spec': sha256(feature_spec(workdir))}
