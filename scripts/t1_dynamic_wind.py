"""Dynamics from hourly 100 m NWP grids on a fixed physical stencil."""
from pathlib import Path

import numpy as np

from independent_t2 import POWER_HOURS, SOURCES as NWP_SOURCES, bracket, interpolate
from steamlink_native_wind import append_native

SOURCES = {source: NWP_SOURCES[source] for source in ('tianji', 'ec')}
RADIUS_M = 25_000.
EARTH_M = 6_371_000.
NAMES = [f'{source}_dynamic_{field}' for source in SOURCES for field in (
    'divergence_per_hour', 'speed_advection_mps_per_hour', 'upwind_speed_gap_mps')]


def increasing_axis(axis):
    step = np.diff(axis)
    if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
        raise ValueError('Expected a finite one-dimensional weather grid.')
    if np.all(step > 0):
        return axis, False
    if np.all(step < 0):
        return axis[::-1], True
    raise ValueError('Expected a strictly monotonic weather grid.')


def angular_offsets(coordinates):
    latitude = coordinates[..., 0]
    north = np.full(latitude.shape, np.rad2deg(RADIUS_M / EARTH_M))
    east = north / np.cos(np.deg2rad(latitude))
    return north, east


def enclosing_slice(axis, low, high, descending):
    start = np.clip(np.searchsorted(axis, low, side='right') - 1, 0, len(axis)-2)
    end = np.clip(np.searchsorted(axis, high, side='left') + 1, start+2, len(axis))
    part = axis[start:end]
    box = slice(len(axis)-end, len(axis)-start) if descending else slice(start, end)
    return box, part


def interpolate_grid(lat, lon, array, latitude, longitude):
    """Bilinear interpolation, with one row of query coordinates per forecast hour."""
    inside = ((latitude >= lat[0]) & (latitude <= lat[-1])
              & (longitude >= lon[0]) & (longitude <= lon[-1]))
    south, north, yfrac = bracket(lat, np.where(inside, latitude, lat[0]))
    west, east, xfrac = bracket(lon, np.where(inside, longitude, lon[0]))
    hour = np.arange(len(array))[:, None]
    low = interpolate(array[hour, south, west], array[hour, south, east], xfrac)
    high = interpolate(array[hour, north, west], array[hour, north, east], xfrac)
    return np.where(inside, interpolate(low, high, yfrac), np.nan)


def dynamic_block(lat, lon, u, v, coordinates):
    """Return the three registered fields with shape (hour, station, field)."""
    north, east = angular_offsets(coordinates)
    y, x = coordinates.T
    local_lat = np.broadcast_to(np.stack((y, y, y, y-north, y+north)), (len(u), 5, len(y)))
    local_lon = np.broadcast_to(np.stack((x, x-east, x+east, x, x)), (len(u), 5, len(y)))
    shape = (len(u), len(y))

    def sample(array):
        return interpolate_grid(lat, lon, array, local_lat.reshape(len(u), -1),
                                local_lon.reshape(len(u), -1)).reshape(len(u), 5, len(y)).transpose(1, 0, 2)

    u0, uw, ue, us, un = sample(u)
    v0, vw, ve, vs, vn = sample(v)
    center = np.hypot(u0, v0)
    divergence = ((ue-uw) + (vn-vs)) * (3600 / (2*RADIUS_M))
    speed_dx = (np.hypot(ue, ve)-np.hypot(uw, vw)) / (2*RADIUS_M)
    speed_dy = (np.hypot(un, vn)-np.hypot(us, vs)) / (2*RADIUS_M)
    advection = -(u0*speed_dx + v0*speed_dy) * 3600
    moving = center >= 1.
    uy = np.divide(v0, center, out=np.full(shape, np.nan), where=moving)
    ux = np.divide(u0, center, out=np.full(shape, np.nan), where=moving)
    upstream_y, upstream_x = y - north*uy, x - east*ux
    upstream = np.hypot(interpolate_grid(lat, lon, u, upstream_y, upstream_x),
                        interpolate_grid(lat, lon, v, upstream_y, upstream_x))
    return np.stack((divergence, advection, upstream-center), axis=-1)


def source_block(path, expected, coordinates, dataset):
    with dataset(path, mode='r') as ds:
        hours, lat, lon = (np.ma.asarray(ds.variables[k][:], dtype=float).filled(np.nan)
                           for k in ('time', 'lat', 'lon'))
        if not np.array_equal(hours, expected):
            raise ValueError(f'{path}: unexpected forecast times.')
        lat, reverse_y = increasing_axis(lat)
        lon, reverse_x = increasing_axis(lon)
        north, east = angular_offsets(coordinates)
        y, x = coordinates.T
        if np.any((y < lat[0]) | (y > lat[-1]) | (x < lon[0]) | (x > lon[-1])):
            raise ValueError(f'{path}: the weather grid does not cover each requested station.')
        y_box, lat = enclosing_slice(lat, (y-north).min(), (y+north).max(), reverse_y)
        x_box, lon = enclosing_slice(lon, (x-east).min(), (x+east).max(), reverse_x)
        start = np.searchsorted(hours, POWER_HOURS[0], side='right') - 1
        end = np.searchsorted(hours, POWER_HOURS[-1], side='left') + 1
        native = hours[start:end]
        values = []
        for variable in ('u100', 'v100'):
            data = ds.variables[variable]
            if tuple(data.dimensions) != ('time', 'lat', 'lon'):
                raise ValueError(f'{path}: wrong dimensions for {variable}.')
            field = np.ma.asarray(data[start:end, y_box, x_box], dtype=float).filled(np.nan)
            field = field[:, ::-1 if reverse_y else 1, ::-1 if reverse_x else 1]
            values.append(field)
    components = dynamic_block(lat, lon, *values, coordinates)
    before, after, fraction = bracket(native, POWER_HOURS)
    aligned = interpolate(components[before], components[after], fraction[:, None, None])
    return aligned.transpose(1, 0, 2)


def extract(paths, coordinates):
    from netCDF4 import Dataset

    return np.concatenate([source_block(path, hours, coordinates, Dataset) for path, hours in paths],
                          axis=-1).astype(np.float32)


def append_dynamic(root, cache, matrix, names, dates, roster, positions):
    return append_native(root, cache, matrix, names, dates, roster, positions, sources=SOURCES,
                         extractor=extract, columns=NAMES, source_version=Path(__file__).read_bytes())
