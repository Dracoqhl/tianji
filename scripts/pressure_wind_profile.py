"""Issue-local Tianji pressure-layer wind structure with below-ground masking."""
from pathlib import Path

import numpy as np
import pandas as pd

from independent_t2 import POWER_HOURS, bracket, interpolate
from steamlink_native_wind import append_native
from t1_dynamic_wind import SOURCES as FULL_SOURCES, enclosing_slice, increasing_axis, interpolate_grid

SOURCES = {'tianji': FULL_SOURCES['tianji']}
LAYERS = (1000, 925, 850, 700)
NAMES = ['tianji_pressure_' + field for field in (
    'lowest_speed_minus_100m', 'lowest_direction_cosine', 'max_neighbor_vector_gap',
    'interior_speed_peak', 'available_layers', 'lowest_available_hpa')]


def maximum_known(values):
    found = np.isfinite(values).any(axis=-1)
    return np.where(found, np.max(np.where(np.isfinite(values), values, -np.inf), axis=-1), np.nan)


def profile_stats(surface, horizontal):
    """Summarize sampled vectors with (..., pressure level, horizontal component)."""
    known = np.isfinite(horizontal).all(axis=-1)
    any_known = known.any(axis=-1)
    low = known.argmax(axis=-1)
    nearest = np.take_along_axis(horizontal, low[..., None, None], axis=-2)[..., 0, :]
    nearest = np.where(any_known[..., None], nearest, np.nan)
    near_speed = np.linalg.norm(nearest, axis=-1)
    surface_speed = np.linalg.norm(surface, axis=-1)
    dot = np.sum(surface * nearest, axis=-1)
    length = surface_speed * near_speed
    agreement = np.divide(dot, length, out=np.full(dot.shape, np.nan), where=length >= 1.0)
    speed = np.linalg.norm(np.where(known[..., None], horizontal, np.nan), axis=-1)
    steps = np.linalg.norm(np.diff(horizontal, axis=-2), axis=-1)
    bump = speed[..., 1:-1] - np.maximum(speed[..., :-2], speed[..., 2:])
    return np.stack((near_speed - surface_speed, np.clip(agreement, -1, 1),
                     maximum_known(steps), np.maximum(maximum_known(bump), 0),
                     known.sum(axis=-1), np.where(any_known, np.asarray(LAYERS)[low], np.nan)), axis=-1)


def source_block(path, expected, coordinates, dataset):
    with dataset(path, mode='r') as ds:
        hours, lat, lon = (np.ma.asarray(ds.variables[key][:], dtype=float).filled(np.nan)
                           for key in ('time', 'lat', 'lon'))
        if not np.array_equal(hours, expected):
            raise ValueError(f'{path}: unexpected Tianji native hours.')
        lat, reverse_y = increasing_axis(lat)
        lon, reverse_x = increasing_axis(lon)
        y, x = coordinates.T
        if np.any((y < lat[0]) | (y > lat[-1]) | (x < lon[0]) | (x > lon[-1])):
            raise ValueError(f'{path}: stations lie outside the forecast grid.')
        y_slice, lat = enclosing_slice(lat, y.min(), y.max(), reverse_y)
        x_slice, lon = enclosing_slice(lon, x.min(), x.max(), reverse_x)
        start = np.searchsorted(hours, POWER_HOURS[0], side='right') - 1
        end = np.searchsorted(hours, POWER_HOURS[-1], side='left') + 1
        native = hours[start:end]
        yy = np.broadcast_to(y, (len(native), len(y)))
        xx = np.broadcast_to(x, yy.shape)

        def field(name):
            variable = ds.variables[name]
            if tuple(variable.dimensions) != ('time', 'lat', 'lon'):
                raise ValueError(f'{path}: wrong dimensions for {name}.')
            values = np.ma.asarray(variable[start:end, y_slice, x_slice], dtype=float).filled(np.nan)
            return values[:, ::-1 if reverse_y else 1, ::-1 if reverse_x else 1]

        surface_pressure = field('sp')

        def sample_wind(suffix, keep):
            u, v = [np.where(keep, field(f'{axis}{suffix}'), np.nan) for axis in ('u', 'v')]
            return np.stack([interpolate_grid(lat, lon, vector, yy, xx) for vector in (u, v)], axis=-1)

        surface = sample_wind(100, True)
        horizontal = np.stack([sample_wind(level, np.isfinite(surface_pressure) &
                                            (surface_pressure >= 100*level))
                               for level in LAYERS], axis=-2)
    rows = profile_stats(surface, horizontal)
    before, after, fraction = bracket(native, POWER_HOURS)
    aligned = interpolate(rows[before], rows[after], fraction[:, None, None])
    left = np.searchsorted(native, POWER_HOURS, side='right') - 1
    aligned[..., -2:] = rows[left, ..., -2:]
    return aligned.transpose(1, 0, 2).astype(np.float32)


def extract(paths, coordinates):
    from netCDF4 import Dataset

    (path, hours), = paths
    return source_block(path, hours, coordinates, Dataset)


def append_pressure(root, cache, matrix, names, dates, roster, positions):
    station_ids = pd.read_csv(root / 'static/power_stations.csv', dtype={'station_id': str},
                              usecols=['station_id']).station_id.tolist()
    whole = sorted(set(station_ids))
    if len(whole) != 18 or len(station_ids) != len(whole):
        raise ValueError('Expected 18 distinct power sites.')
    day_station, point = np.divmod(positions, len(POWER_HOURS))
    day, slots = np.divmod(day_station, len(roster))
    mapped = np.array([whole.index(site) for site in roster])[slots]
    positions_full = (day * len(whole) + mapped) * len(POWER_HOURS) + point
    return append_native(root, cache, matrix, names, dates, whole, positions_full, sources=SOURCES,
                         extractor=extract, columns=NAMES, source_version=Path(__file__).read_bytes())
