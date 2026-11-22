"""Station-day weather descriptors based on one Tianji forecast."""

import json
from pathlib import Path

import numpy as np

from run_companion_warm_t1 import cache_paths


FIELDS = ('tianji_t2m', 'tianji_u100', 'tianji_v100', 'tianji_tp_rate')
NAMES = ['warm_daytime_temperature_gain_k', 'warm_daytime_wind_direction_cosine',
         'warm_day_wet_hour_fraction']


def mean_where(values, valid):
    count = valid.sum(axis=1, keepdims=True)
    total = np.where(valid, values, 0).sum(axis=1, keepdims=True)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count >= 4).squeeze(1)


def warm_block(weather):
    """Convert [station, 96 points, 4 fields] to three station-day features."""
    if weather.ndim != 3 or weather.shape[1:] != (96, len(FIELDS)):
        raise ValueError('Warm-weather inputs must have 96 points and the registered Tianji columns.')
    hours = weather[:, ::4, :].astype(float)
    morning, afternoon = hours[:, 6:12], hours[:, 12:18]
    thermal = [mean_where(block[..., 0], np.isfinite(block[..., 0])) for block in (morning, afternoon)]
    winds = []
    for block in (morning, afternoon):
        uv = block[..., 1:3]
        winds.append(mean_where(uv, np.isfinite(uv).all(axis=2, keepdims=True)))
    speeds = [np.linalg.norm(vector, axis=1) for vector in winds]
    strong = (speeds[0] >= 1) & (speeds[1] >= 1)
    cosine = np.divide(np.sum(winds[0] * winds[1], axis=1), speeds[0] * speeds[1],
                       out=np.full(len(hours), np.nan), where=strong)
    precipitation = hours[..., 3]
    observed = np.isfinite(precipitation) & (precipitation >= 0)
    count = observed.sum(axis=1)
    wet = np.divide(((precipitation >= .1) & observed).sum(axis=1), count, out=np.full(len(hours), np.nan),
                    where=count >= 18)
    return np.column_stack((thermal[1] - thermal[0], np.clip(cosine, -1, 1), wet)).astype(np.float32)


def append_warm_weather(workdir, matrix, names, dates, roster, positions):
    """Append features without reading power, target-weather or evaluation date codes."""
    workdir = Path(workdir)
    fields = json.loads((workdir / 'stats/feature_spec_train.json').read_text())['nwp_names']
    if not set(FIELDS).issubset(fields) or set(names).intersection(NAMES):
        raise ValueError('Warm-weather fields are incomplete or already appended.')
    columns = [fields.index(name) for name in FIELDS]
    stride = len(roster) * 96
    positions = np.asarray(positions)
    if (positions.shape != (len(matrix),) or not np.issubdtype(positions.dtype, np.integer)
            or (positions < 0).any() or (positions >= len(dates) * stride).any()):
        raise ValueError('Warm-weather row coordinates do not match the forecast grid.')
    daily = []
    for path in cache_paths(workdir, dates):
        with np.load(path, allow_pickle=False) as data:
            sites = data['station_ids'].tolist()
            if len(set(sites)) != len(sites) or data['date_code'].tolist() != [path.stem]:
                raise ValueError(f'{path}: unexpected weather keys.')
            if not set(roster).issubset(sites):
                raise ValueError(f'{path}: missing station weather.')
            weather = data['nwp_point']
            if weather.shape != (len(sites), 96, len(fields)):
                raise ValueError(f'{path}: unexpected weather dimensions.')
            rows = [sites.index(site) for site in roster]
            daily.append(warm_block(weather[rows][:, :, columns]))
    extra = np.concatenate(daily, axis=0)[positions // 96]
    return np.column_stack((matrix, extra)), [*names, *NAMES]
