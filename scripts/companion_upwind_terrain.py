"""Two deterministic DEM features from the same issue's Tianji 100 m wind."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


COMPONENT = 'upwind_terrain'
DISTANCE_KM = (3, 9)
COLUMNS = tuple(f'tianji_upwind_terrain_diff_{km}km' for km in DISTANCE_KM)
EARTH_RADIUS_KM = 6371.0088
MIN_WIND = 1.0


def endpoints(latitude, longitude, bearing, km):
    lat, lon, angle = np.deg2rad(latitude), np.deg2rad(longitude), np.deg2rad(bearing)
    distance = np.asarray(km) / EARTH_RADIUS_KM
    lat2 = np.arcsin(np.sin(lat) * np.cos(distance) + np.cos(lat) * np.sin(distance) * np.cos(angle))
    lon2 = lon + np.arctan2(np.sin(angle) * np.sin(distance) * np.cos(lat),
                           np.cos(distance) - np.sin(lat) * np.sin(lat2))
    return np.rad2deg(lat2), (np.rad2deg(lon2) + 180) % 360 - 180


def make_lookup(latitude, longitude, elevation, coordinates):
    from scipy.interpolate import RegularGridInterpolator

    if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all()
            or elevation.shape != (len(latitude), len(longitude))):
        raise ValueError('Unexpected station coordinates or DEM axes.')
    interpolate = RegularGridInterpolator((latitude, longitude), elevation, bounds_error=False, fill_value=np.nan)
    base = interpolate(coordinates)
    lat, lon = coordinates.T
    bearing = np.arange(360)[None, :, None]
    positions = endpoints(lat[:, None, None], lon[:, None, None], bearing, np.asarray(DISTANCE_KM))
    points = np.stack(positions, axis=-1)
    return (interpolate(points) - base[:, None, None]).astype(np.float32)


def wind_bearing(u, v):
    """Degrees clockwise from north, pointing toward the origin of the air."""
    finite = np.isfinite(u) & np.isfinite(v)
    use = finite & (np.hypot(np.where(finite, u, 0.0), np.where(finite, v, 0.0)) >= MIN_WIND)
    bearing = np.full(len(u), np.nan, dtype=float)
    bearing[use] = np.degrees(np.arctan2(-u[use], -v[use])) % 360
    # Roundoff near north can yield exactly 360 degrees.
    bearing[use] %= 360
    return bearing


def directional_values(lookup, slots, bearing):
    values = np.full((len(bearing), len(DISTANCE_KM)), np.nan, dtype=np.float32)
    valid = np.isfinite(bearing)
    angle = bearing[valid]
    low = np.floor(angle).astype(np.intp)
    fraction = (angle - low)[:, None]
    left = lookup[slots[valid], low]
    right = lookup[slots[valid], (low + 1) % 360]
    values[valid] = np.where(fraction == 0, left, left + fraction * (right - left))
    return values


class UpwindTerrain:
    def __init__(self, path, roster):
        with np.load(path, allow_pickle=False) as cached:
            cached_roster = cached['station_ids'].tolist()
            lookup = cached['terrain']
        if (len(set(cached_roster)) != len(cached_roster) or set(cached_roster) != set(roster)
                or lookup.shape != (len(roster), 360, len(DISTANCE_KM)) or np.isinf(lookup).any()):
            raise ValueError('The terrain lookup does not match the companion station roster or direction grid.')
        self.lookup = lookup[[cached_roster.index(station) for station in roster]]
        self.roster = roster

    def append(self, data, names, positions, *, with_bearing=False):
        slots = (positions // 96) % len(self.roster)
        bearing = wind_bearing(data[:, names.index('tianji_u100')], data[:, names.index('tianji_v100')])
        values = directional_values(self.lookup, slots, bearing)
        metadata = {'finite_rows_per_feature': np.isfinite(values).sum(axis=0).tolist(), 'rows': len(data)}
        diagnostic = np.column_stack((bearing, values)) if with_bearing else None
        return np.column_stack((data, values)), [*names, *COLUMNS], metadata, diagnostic


def source_stamp(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lookup_inputs(root):
    return {'terrain_sha256': source_stamp(root / 'static/terrain.nc'),
            'coordinates_sha256': source_stamp(root / 'static/power_stations.csv'),
            'implementation_sha256': source_stamp(Path(__file__))}


def read_dem(path):
    def values(dataset):
        return tuple(np.ma.asarray(dataset.variables[name][:], dtype=float).filled(np.nan)
                     for name in ('lat', 'lon', 'elevation'))

    def classic_value(variable):
        raw = np.asarray(variable[:])
        missing = np.zeros(raw.shape, dtype=bool)
        for attribute in ('_FillValue', 'missing_value'):
            if hasattr(variable, attribute):
                missing |= np.isin(raw, getattr(variable, attribute))
        calibrated = raw.astype(float) * getattr(variable, 'scale_factor', 1) + getattr(variable, 'add_offset', 0)
        return np.where(missing, np.nan, calibrated)

    errors = []
    for reader in ('netCDF4', 'xarray', 'scipy'):
        try:
            if reader == 'netCDF4':
                from netCDF4 import Dataset

                with Dataset(path) as dataset:
                    return values(dataset)
            if reader == 'xarray':
                import xarray

                with xarray.open_dataset(path, decode_times=False, cache=False) as dataset:
                    return tuple(dataset[name].values.astype(float) for name in ('lat', 'lon', 'elevation'))
            from scipy.io import netcdf_file

            with netcdf_file(path, mmap=False) as dataset:
                return tuple(classic_value(dataset.variables[name]) for name in ('lat', 'lon', 'elevation'))
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            errors.append(f'{reader}: {type(error).__name__}: {error}')
    raise RuntimeError('Cannot read the static DEM. ' + ' | '.join(errors))


def create_lookup(root, destination):
    from warm_shared_power import load_roster

    stamp = lookup_inputs(root)
    roster = load_roster(root)
    sites = pd.read_csv(root / 'static/power_stations.csv', dtype={'station_id': str}).set_index('station_id')
    coordinates = sites.loc[roster, ['latitude', 'longitude']].to_numpy(dtype=float)
    grid = read_dem(root / 'static/terrain.nc')
    values = make_lookup(*grid, coordinates)
    if not np.isfinite(values).any():
        raise ValueError('None of the registered T1 sites has valid DEM elevation differences.')
    if stamp != lookup_inputs(root):
        raise ValueError('Static terrain inputs changed while computing the lookup.')
    with TemporaryDirectory(dir=destination.parent) as folder:
        pending = Path(folder) / 'lookup.npz'
        np.savez_compressed(pending, terrain=values, station_ids=roster, inputs=json.dumps(stamp, sort_keys=True))
        pending.replace(destination)


def coverage(path, root):
    with np.load(path, allow_pickle=False) as cache:
        stamp = json.loads(cache['inputs'].item())
        if stamp != lookup_inputs(root):
            raise ValueError('The stored lookup was calculated from different static inputs.')
        roster, values = cache['station_ids'].tolist(), cache['terrain']
    result = {}
    for site, table in zip(roster, values):
        stats = {}
        for i, km in enumerate(DISTANCE_KM):
            good = table[:, i][np.isfinite(table[:, i])]
            stats[str(km)] = {'covered_directions': len(good),
                              'elevation_difference_range_m': [float(good.min()), float(good.max())]
                              if len(good) else None}
        result[site] = stats
    return {'inputs': stamp, 'distance_km': list(DISTANCE_KM), 'sites': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    create_lookup(args.data_root, args.out)


if __name__ == '__main__':
    main()
