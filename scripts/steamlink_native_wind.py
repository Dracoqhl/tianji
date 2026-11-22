"""Small, issue-local summaries of native weather trajectories."""
from pathlib import Path
from tempfile import NamedTemporaryFile
import hashlib

import numpy as np
import pandas as pd

from independent_t2 import POWER_HOURS, SOURCES, sample_variable, station_sampling

FIELDS = 'du dv ds ramp_up ramp_down up_hour down_hour valid_segments'.split()
NAMES = [f'native_{source}_{field}' for source in SOURCES for field in FIELDS]
NAMES += ['native_active_sources', 'native_hourly_up_gap', 'native_hourly_down_gap']


def native_block(hours, u, v):
    """Return (station, point, column) with exact segment durations and no gap bridging."""
    speeds = np.hypot(u, v)
    dt = np.diff(hours)
    adjacent = (np.isfinite(speeds[:-1]) & np.isfinite(speeds[1:]))
    slopes = np.where(adjacent[..., None],
                      np.stack([np.diff(values, axis=0) for values in (u, v, speeds)], axis=-1)
                      / dt[:, None, None], np.nan)
    segment = np.searchsorted(hours, POWER_HOURS, side='right') - 1
    inside = (hours[:-1] < POWER_HOURS[-1]) & (hours[1:] > POWER_HOURS[0])
    eligible = adjacent & inside[:, None]
    changes = slopes[..., 2]
    up = np.argmax(np.where(eligible, changes, -np.inf), axis=0)
    down = np.argmin(np.where(eligible, changes, np.inf), axis=0)
    sites = np.arange(u.shape[1])
    high, low = changes[up, sites], changes[down, sites]
    has_up = eligible.any(axis=0) & (high > 0)
    has_down = eligible.any(axis=0) & (low < 0)
    stats = np.column_stack((np.where(has_up, high, np.nan), np.where(has_down, low, np.nan),
                             np.where(has_up, hours[up + 1], np.nan),
                             np.where(has_down, hours[down + 1], np.nan), eligible.sum(axis=0)))
    return np.concatenate((slopes[segment].transpose(1, 0, 2),
                           np.broadcast_to(stats[:, None], (len(sites), len(POWER_HOURS), len(FIELDS)-3))),
                          axis=-1)


def extract(paths, coordinates):
    from netCDF4 import Dataset

    blocks = []
    for (path, expected) in paths:
        with Dataset(path, mode='r') as ds:
            hours, lat, lon = (np.ma.asarray(ds.variables[key][:], dtype=float).filled(np.nan)
                               for key in ('time', 'lat', 'lon'))
            if not np.array_equal(hours, expected):
                raise ValueError(f'{path}: unexpected native forecast times.')
            sampling = station_sampling(lat, lon, coordinates)
            indices = np.arange(len(hours))
            u, v = (sample_variable(ds.variables[name], sampling, (indices, indices, np.zeros(len(hours))))
                    for name in ('u100', 'v100'))
        blocks.append(native_block(hours, u, v))
    active = sum(np.isfinite(part[..., 2]) for part in blocks)
    # Compare timing only between hourly sources; AIFS retains its own coarse summaries.
    gaps = np.abs(blocks[0][..., 5:7] - blocks[1][..., 5:7])
    return np.concatenate([*blocks, active[..., None], gaps], axis=-1).astype(np.float32)


def append_native(root, cache, matrix, names, dates, roster, positions, *,
                  sources=SOURCES, extractor=extract, columns=NAMES, source_version=b''):
    table = pd.read_csv(root / 'static/power_stations.csv', dtype={'station_id': str}).set_index('station_id')
    coords = table.loc[roster, ['latitude', 'longitude']].to_numpy()
    if not np.isfinite(coords).all():
        raise ValueError('Nonfinite station coordinates.')
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() + source_version
                                 + coords.tobytes() + repr(roster).encode()).hexdigest()
    cache = cache / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    feature = np.empty((len(positions), len(columns)), np.float32)
    stride = 96 * len(roster)
    order = np.argsort(positions)
    sorted_positions = positions[order]
    rows = np.searchsorted(sorted_positions, np.arange(len(dates) + 1) * stride)
    if rows[0] or rows[-1] != len(positions) or np.any(np.diff(sorted_positions) == 0):
        raise ValueError('Native feature rows must have unique in-range keys.')
    for index, day in enumerate(dates):
        folder = root / 'train' / f'{day:%Y%m%d}_0000' / 'nwp'
        paths = [(folder / name, hours) for name, hours, _ in sources.values()]
        stamps = [(p.stat().st_size, p.stat().st_mtime_ns) for p, _ in paths]
        key = hashlib.sha256(repr((str(folder.resolve()), stamps)).encode()).hexdigest()[:24]
        file = cache / f'{day}-{key}.npy'
        if not file.exists():
            values = extractor(paths, coords)
            with NamedTemporaryFile(dir=cache, suffix='.npy', delete=False) as pending:
                np.save(pending, values, allow_pickle=False)
            Path(pending.name).replace(file)
        values = np.load(file, allow_pickle=False)
        if values.shape != (len(roster), 96, len(columns)) or np.isinf(values).any():
            raise ValueError(f'{file}: invalid native weather cache.')
        take = slice(*rows[index:index+2])
        feature[order[take]] = values.reshape(stride, len(columns))[sorted_positions[take] % stride]
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(dates):
            print(f'[T1 原生天气] {index + 1}/{len(dates)}', flush=True)
    return np.column_stack((matrix, feature)), [*names, *columns]
