"""Geographic and flow-directed neighbor features from companion weather caches."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


SPATIAL_COMPONENTS = ('geo_neighbors', 'upwind_neighbors')
WIND_SOURCES = ('tianji', 'ec', 'aifs')


def geometry(coordinates):
    """Choose two nearest neighbors using target-latitude local east/north distances."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 3 or not np.isfinite(coordinates).all():
        raise ValueError('Valid locations for at least three fixed-roster stations are required.')
    latitude, longitude = coordinates.T
    dy = latitude[None, :] - latitude[:, None]
    dx = (longitude[None, :] - longitude[:, None]) * np.cos(np.deg2rad(latitude[:, None]))
    distance = np.hypot(dx, dy)
    np.fill_diagonal(distance, np.inf)
    near = np.argsort(distance, axis=1, kind='stable')[:, :2]
    rows = np.arange(len(coordinates))[:, None]
    return near, dx[rows, near], dy[rows, near]


def neighbor_features(u, v, near, east, north):
    """Return [variant, station, point, feature] using only issued forecast winds."""
    u = np.where(np.isfinite(u), u, np.nan)
    v = np.where(np.isfinite(v), v, np.nan)
    speed = np.hypot(u, v)
    neighbor_speed = speed[near]
    projection = u[:, None] * east[..., None, None] + v[:, None] * north[..., None, None]
    has_wind = np.isfinite(neighbor_speed)

    def summarize(valid):
        count = valid.sum(axis=1)
        total = np.where(valid, neighbor_speed, 0).sum(axis=1)
        mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        return np.concatenate((mean, mean - speed), axis=-1)

    return np.stack((summarize(has_wind), summarize(has_wind & (projection < 0)))).astype(np.float32)


class StationNeighborCache:
    def __init__(self, root, cache_root, roster, stations, spec):
        coordinates = pd.read_csv(root / 'static/power_stations.csv', dtype={'station_id': str})
        coordinates = coordinates.set_index('station_id').loc[roster, ['latitude', 'longitude']].to_numpy(dtype=float)
        self.roster = roster
        self.stations = np.asarray(stations, dtype=np.intp)
        self.nwp_names = list(spec['nwp_names'])
        self.near, self.east, self.north = geometry(coordinates)
        self.columns = {name: [f'{name}_{stat}_{source}' for stat in ('mean', 'minus_local')
                               for source in WIND_SOURCES] for name in SPATIAL_COMPONENTS}
        recipe = json.dumps([str(root.resolve()), roster, self.stations.tolist(), self.nwp_names]).encode()
        self.version = hashlib.sha256(Path(__file__).read_bytes() + coordinates.tobytes() + recipe).hexdigest()
        self.directory = cache_root / self.version
        self.directory.mkdir(parents=True, exist_ok=True)

    def day(self, companion_path):
        stat = companion_path.stat()
        fingerprint = f'{stat.st_size}-{stat.st_mtime_ns}'
        cached = self.directory / f'{companion_path.stem}-{fingerprint}.npz'
        hit = cached.is_file()
        if hit:
            with np.load(cached, allow_pickle=False) as data:
                features = data['features']
        else:
            with np.load(companion_path, allow_pickle=False) as data:
                if (data['date_code'].tolist() != [companion_path.stem] or
                        data['station_ids'][self.stations].tolist() != self.roster):
                    raise ValueError(f'{companion_path}: invalid weather date or station order.')
                nwp = data['nwp_point'][self.stations]
            u, v = (nwp[..., [self.nwp_names.index(f'{source}_{axis}100') for source in WIND_SOURCES]]
                    for axis in ('u', 'v'))
            features = neighbor_features(u, v, self.near, self.east, self.north)
            with TemporaryDirectory(dir=self.directory) as temporary:
                pending = Path(temporary) / 'forecast.npz'
                np.savez_compressed(pending, features=features)
                pending.replace(cached)
        if features.shape != (len(SPATIAL_COMPONENTS), len(self.roster), 96, 6) or np.isinf(features).any():
            raise ValueError(f'{cached}: invalid spatial weather features.')
        return features, hit


def extend_spatial(inputs, names, positions, paths, cache, components):
    if cache is None:
        return inputs, names, None
    daily, hits = [], 0
    for index, path in enumerate(paths, 1):
        values, hit = cache.day(path)
        daily.append(values)
        hits += hit
        if index == 1 or index % 50 == 0 or index == len(paths):
            print(f'Neighbor weather {index}/{len(paths)} days, cache hits {hits}', flush=True)
    stacked = np.stack(daily, axis=1)
    parts = [inputs]
    columns = list(names)
    for i, name in enumerate(SPATIAL_COMPONENTS):
        if name in components:
            parts.append(stacked[i].reshape(-1, 6)[positions])
            columns.extend(cache.columns[name])
    return np.concatenate(parts, axis=1), columns, {'version': cache.version, 'hits': hits, 'days': len(paths)}
