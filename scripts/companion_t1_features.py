"""Fixed, per-row feature and weighting changes for companion warm T1."""

import hashlib
from inspect import getsource
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from independent_t2 import POWER_HOURS, SOURCES, bracket, interpolate


COMPONENTS = ('disagreement', 'gpm_missing', 'station_day_weight', 'local_wind')
WIND_SOURCES = ('tianji', 'ec', 'aifs')
DISAGREEMENT_COLUMNS = tuple(f'wind100_source_{stat}' for stat in ('mean', 'std', 'range', 'vector_rms'))
GPM_POINTS = tuple(f'gpm_point_{step}' for step in range(4))
GPM_SUMMARIES = ('gpm_total_4h', 'gpm_hours_since_rain')
PATCH_COLUMNS = tuple(f'{source}_ws100_patch_{stat}' for source in ('ec', 'aifs')
                      for stat in ('mean', 'std', 'min', 'max'))


def wind_disagreement(u, v):
    """Summarize three equally weighted forecasts, requiring all six components."""
    u = np.where(np.isfinite(u), u, np.nan)
    v = np.where(np.isfinite(v), v, np.nan)
    speed = np.hypot(u, v)
    vector_rms = np.hypot(u.std(axis=-1), v.std(axis=-1))
    return np.column_stack((speed.mean(axis=-1), speed.std(axis=-1), np.ptp(speed, axis=-1), vector_rms))


def gpm_summaries(points):
    finite = np.isfinite(points)
    total = np.where(finite, points, 0).sum(axis=-1, dtype=np.float64)
    total[~finite.any(axis=-1)] = np.nan
    dry = finite & (points < 0.1)
    # Prefix product over newest-to-oldest frames stops at a missing or rainy hour.
    trailing = np.cumprod(dry[:, ::-1], axis=1).sum(axis=1).astype(float)
    trailing[~finite[:, -1]] = np.nan
    return total, trailing


def station_day_weights(positions, n_stations):
    """Normalize to mean one over the valid points in this fit partition."""
    station_days = positions // 96
    groups, inverse, point_counts = np.unique(station_days, return_inverse=True, return_counts=True)
    station_slots = groups % n_stations
    days_per_station = np.bincount(station_slots, minlength=n_stations)
    weights = 1.0 / (days_per_station[station_slots[inverse]] * point_counts[inverse])
    return weights / weights.mean()


def patch_statistics(speed):
    valid = np.isfinite(speed)
    cells = valid.sum(axis=(1, 2))
    mean = np.divide(np.where(valid, speed, 0).sum(axis=(1, 2)), cells,
                     out=np.full(len(speed), np.nan), where=cells > 0)
    variance = np.divide(np.where(valid, (speed - mean[:, None, None])**2, 0).sum(axis=(1, 2)), cells,
                         out=np.full(len(speed), np.nan), where=cells > 0)
    low = np.where(valid, speed, np.inf).min(axis=(1, 2))
    high = np.where(valid, speed, -np.inf).max(axis=(1, 2))
    low[cells == 0] = high[cells == 0] = np.nan
    return np.column_stack((mean, np.sqrt(variance), low, high))


def source_patches(path, coordinates, expected_times, dataset_factory):
    with dataset_factory(path, mode='r') as ds:
        hours, lat, lon = (np.ma.asarray(ds.variables[key][:], dtype=float).filled(np.nan)
                           for key in ('time', 'lat', 'lon'))
        if not np.array_equal(hours, expected_times):
            raise ValueError(f'{path}: unexpected forecast hours.')
        for axis in (lat, lon):
            if (axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all() or
                    not (np.all(np.diff(axis) > 0) or np.all(np.diff(axis) < 0))):
                raise ValueError(f'{path}: invalid grid axis.')
        if (not ((lat.min() <= coordinates[:, 0]) & (coordinates[:, 0] <= lat.max())).all() or
                not ((lon.min() <= coordinates[:, 1]) & (coordinates[:, 1] <= lon.max())).all()):
            raise ValueError(f'{path}: grid must cover all T1 stations.')
        u, v = (ds.variables[name] for name in ('u100', 'v100'))
        if any(tuple(var.dimensions) != ('time', 'lat', 'lon') for var in (u, v)):
            raise ValueError(f'{path}: expected u100/v100 dimensions time,lat,lon.')
        before, after, fraction = bracket(hours, POWER_HOURS)
        rows = []
        for slat, slon in coordinates:
            i, j = int(np.argmin(abs(lat - slat))), int(np.argmin(abs(lon - slon)))
            region = (slice(None), slice(max(0, i - 1), i + 2), slice(max(0, j - 1), j + 2))
            u_cells, v_cells = (np.ma.asarray(var[region], dtype=float).filled(np.nan) for var in (u, v))
            stats = patch_statistics(np.hypot(u_cells, v_cells))
            rows.append(interpolate(stats[before], stats[after], fraction[:, None]))
    return np.asarray(rows, dtype=np.float32)


class LocalWindCache:
    """Store only the eight EC/AIFS columns, separately from companion archives."""

    def __init__(self, data_root, cache_root, roster, dataset_factory=None):
        if dataset_factory is None:
            from netCDF4 import Dataset

            dataset_factory = Dataset
        self.dataset_factory = dataset_factory
        self.root = data_root
        table = pd.read_csv(data_root / 'static/power_stations.csv', dtype={'station_id': str})
        self.coordinates = table.set_index('station_id').loc[roster, ['latitude', 'longitude']].to_numpy(dtype=float)
        if self.coordinates.shape != (len(roster), 2) or not np.isfinite(self.coordinates).all():
            raise ValueError('Invalid T1 station coordinates.')
        contract = [(key, SOURCES[key][0], SOURCES[key][1].tolist()) for key in ('ec', 'aifs')]
        extraction = Path(__file__).read_bytes() + (getsource(bracket) + getsource(interpolate)).encode()
        self.version = hashlib.sha256(extraction + self.coordinates.tobytes() + POWER_HOURS.tobytes() +
                                      json.dumps([str(data_root.resolve()), roster, contract]).encode()).hexdigest()
        self.directory = cache_root / self.version
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shape = (len(roster), len(POWER_HOURS), len(PATCH_COLUMNS))

    def day(self, day):
        folder = self.root / 'train' / day / 'nwp'
        paths = [(folder / SOURCES[key][0], SOURCES[key][1]) for key in ('ec', 'aifs')]
        stats = [path.stat() for path, _ in paths]
        fingerprint = hashlib.sha256(repr([(stat.st_size, stat.st_mtime_ns) for stat in stats]).encode()).hexdigest()
        path = self.directory / f'{day}-{fingerprint}.npz'
        hit = path.is_file()
        if hit:
            with np.load(path, allow_pickle=False) as archive:
                values = archive['features']
        else:
            values = np.concatenate([source_patches(source, self.coordinates, hours, self.dataset_factory)
                                     for source, hours in paths], axis=-1)
        if values.shape != self.shape or values.dtype != np.float32 or np.isinf(values).any():
            raise ValueError(f'{path}: invalid local wind cache shape or values.')
        if not hit:
            with TemporaryDirectory(dir=self.directory) as temporary:
                staging = Path(temporary) / 'features.npz'
                np.savez_compressed(staging, features=values)
                staging.replace(path)
        return values, hit, fingerprint


def add_components(inputs, feature_names, positions, day_paths, local_cache, components):
    columns = {name: index for index, name in enumerate(feature_names)}
    names = list(feature_names)
    pieces = [inputs]
    diagnostics = {}
    cache_info = None
    if 'disagreement' in components:
        u, v = (inputs[:, [columns[f'{source}_{axis}100'] for source in WIND_SOURCES]]
                for axis in ('u', 'v'))
        summary = wind_disagreement(u, v)
        pieces.append(summary)
        names.extend(DISAGREEMENT_COLUMNS)
        diagnostics['disagreement'] = summary[:, 1]
    if 'gpm_missing' in components:
        points = inputs[:, [columns[name] for name in GPM_POINTS]]
        total, trailing = gpm_summaries(points)
        inputs[:, columns[GPM_SUMMARIES[0]]] = total
        inputs[:, columns[GPM_SUMMARIES[1]]] = trailing
        diagnostics['gpm_missing'] = np.isfinite(points).sum(axis=1) + 5 * ~np.isfinite(points[:, -1])
    if 'local_wind' in components:
        patches, hits, fingerprints = [], 0, []
        for index, path in enumerate(day_paths, 1):
            values, hit, fingerprint = local_cache.day(path.stem)
            patches.append(values)
            hits += hit
            fingerprints.append((path.stem, fingerprint))
            if index == 1 or index % 50 == 0 or index == len(day_paths):
                print(f'Local wind {index}/{len(day_paths)} days, cache hits {hits}', flush=True)
        patch_rows = np.concatenate(patches).reshape(-1, len(PATCH_COLUMNS))[positions]
        pieces.append(patch_rows)
        names.extend(PATCH_COLUMNS)
        diagnostics['local_wind'] = np.maximum(patch_rows[:, 1], patch_rows[:, 5])
        cache_info = {'version': local_cache.version, 'hits': hits, 'days': len(day_paths),
                      'source_fingerprint': hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest()}
    return (inputs if len(pieces) == 1 else np.concatenate(pieces, axis=1)), names, diagnostics, cache_info
