"""Issue-time observation wind error fitted entirely within historical training."""
from datetime import timedelta
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from t1_dynamic_wind import increasing_axis, interpolate_grid

INPUT_HOURS = (17, 18, 19, 20)
INITIAL_HOUR = 21
FIELDS = ['station_id', 'date', 'point_index', 'wind_speed', 'wind_direction']
FOLD_DAYS = 90


def observed_vectors(path, day, sites, hours):
    frame = pd.read_csv(path, usecols=FIELDS, dtype={'station_id': str, 'date': str})
    wanted = pd.MultiIndex.from_product((sites, hours), names=('station_id', 'point_index'))
    chosen = frame.loc[frame.date.eq(str(day)) & frame.point_index.isin(hours)].set_index(wanted.names)
    if len(chosen) != len(wanted) or chosen.index.has_duplicates or not wanted.isin(chosen.index).all():
        raise ValueError(f'{path}: incorrect pre-issue wind keys for {day}/{hours}.')
    data = chosen.reindex(wanted)
    speed = data.wind_speed.to_numpy(dtype=float)
    direction = data.wind_direction.to_numpy(dtype=float)
    usable = np.isfinite(speed) & (speed >= 0) & np.isfinite(direction) & (direction >= 0) & (direction <= 360)
    angle = np.deg2rad(np.where(usable, direction, 0))
    vectors = -speed[:, None] * np.column_stack((np.sin(angle), np.cos(angle)))
    vectors = np.where(usable[:, None], vectors, np.nan)
    vectors[speed == 0] = 0
    return vectors.reshape(len(sites), len(hours), 2)


def initial_nwp(path, coordinates, dataset):
    with dataset(path, mode='r') as src:
        hours = np.ma.asarray(src.variables['time'][:], dtype=float).filled(np.nan)
        if not len(hours) or hours[0] != 0:
            raise ValueError(f'{path}: initial NWP time must be 0.')
        lat, rev_lat = increasing_axis(np.asarray(src.variables['lat'][:], dtype=float))
        lon, rev_lon = increasing_axis(np.asarray(src.variables['lon'][:], dtype=float))
        values = []
        for name in ('u10', 'v10'):
            variable = src.variables[name]
            if tuple(variable.dimensions) != ('time', 'lat', 'lon'):
                raise ValueError(f'{path}: unexpected dimensions for {name}.')
            raw = np.ma.asarray(variable[:1], dtype=float).filled(np.nan)
            raw = raw[:, ::-1 if rev_lat else 1, ::-1 if rev_lon else 1]
            values.append(interpolate_grid(lat, lon, raw, coordinates[None, :, 0], coordinates[None, :, 1])[0])
    return np.stack(values, axis=-1)


def observation_table(root):
    table = pd.read_csv(root / 'static/obs_stations.csv', dtype={'station_id': str}).sort_values('station_id')
    coordinates = table[['latitude', 'longitude']].to_numpy(dtype=float)
    if table.station_id.duplicated().any() or len(table) != 87 or not np.isfinite(coordinates).all():
        raise ValueError('Expected the documented 87 distinct weather observation sites.')
    return table.station_id.tolist(), coordinates


def forecast_inputs(root, cache, dates, sites, coordinates, dataset):
    token = hashlib.sha256(Path(__file__).read_bytes() + coordinates.tobytes() + repr(sites).encode()).hexdigest()
    cache = cache / token
    cache.mkdir(parents=True, exist_ok=True)
    result, stamps = [], []
    for index, date in enumerate(dates):
        folder = root / 'train' / f'{date:%Y%m%d}_0000'
        obs, nwp = folder / 'obs/obs_input.csv', folder / 'nwp/tianji_0p125.nc'
        signature = tuple((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in (obs, nwp))
        digest = hashlib.sha256(repr(signature).encode()).hexdigest()[:24]
        file = cache / f'{date}-{digest}.npy'
        if not file.is_file():
            values = observed_vectors(obs, date - timedelta(days=1), sites, INPUT_HOURS)
            model = initial_nwp(nwp, coordinates, dataset)
            combined = np.concatenate((values.reshape(len(sites), 8), model, coordinates), axis=1)
            with NamedTemporaryFile(dir=cache, suffix='.npy', delete=False) as tmp:
                np.save(tmp, combined, allow_pickle=False)
            Path(tmp.name).replace(file)
        inputs = np.load(file, allow_pickle=False)
        if inputs.shape != (len(sites), 12) or np.isinf(inputs).any():
            raise ValueError(f'{file}: invalid cached issue inputs.')
        result.append(inputs)
        stamps.append(digest)
        if index == 0 or (index + 1) % 50 == 0 or index+1 == len(dates):
            print(f'[OBS/Tianji] {index+1}/{len(dates)} days', flush=True)
    return np.stack(result), hashlib.sha256(repr(stamps).encode()).hexdigest()


def training_truth(root, dates, sites):
    return np.stack([observed_vectors(root / 'train' / f'{d:%Y%m%d}_0000/obs/obs_target.csv',
                                      d-timedelta(days=1), sites, (INITIAL_HOUR,))[:, 0] for d in dates])


def supported(values):
    return np.isfinite(values[..., :8].reshape(*values.shape[:-1], 4, 2)).all(axis=-1).any(axis=-1) & (
        np.isfinite(values[..., 8:10]).all(axis=-1))


def initial_residual_fit(train, truth, predict):
    train_residual = truth - train[..., 8:10]
    valid = supported(train) & np.isfinite(train_residual).all(axis=-1)
    x, y = train[valid], train_residual[valid]
    if len(x) < 12:
        raise ValueError('Insufficient valid observational initial wind samples.')
    finite = np.isfinite(x)
    n = finite.sum(axis=0).clip(min=1)
    center = np.where(finite, x, 0).sum(axis=0) / n
    scale = np.sqrt(np.where(finite, (x-center)**2, 0).sum(axis=0) / n).clip(min=1e-4)

    def normalized(values):
        known = np.isfinite(values)
        return np.concatenate((np.where(known, (values-center)/scale, 0), ~known), axis=-1)

    x = normalized(x)
    mean = y.mean(axis=0)
    coef = np.linalg.solve(x.T@x / len(x) + .1*np.eye(x.shape[1]), x.T@(y-mean) / len(x))
    result = normalized(predict) @ coef + mean
    return np.where(supported(predict)[..., None], result, np.nan), int(len(x))


def near_weights(sensors, farms):
    delta = np.deg2rad(sensors[None] - farms[:, None])
    dot = (np.sin(delta[..., 0] / 2)**2 + np.cos(np.deg2rad(farms[:, None, 0]))
           * np.cos(np.deg2rad(sensors[None, :, 0])) * np.sin(delta[..., 1] / 2)**2)
    km = 12742 * np.arcsin(np.sqrt(np.clip(dot, 0, 1)))
    indices = np.argsort(km, axis=1, kind='stable')[:, :5]
    distance = np.take_along_axis(km, indices, axis=1)
    return indices, 1. / (1. + distance)


def farm_residual(residual, near, weight):
    values = residual[:, near]
    valid = np.isfinite(values).all(axis=-1)
    mass = (valid * weight).sum(axis=-1)
    numerator = (np.where(valid[..., None], values, 0) * weight[..., None]).sum(axis=2)
    return np.divide(numerator, mass[..., None], out=np.full(numerator.shape, np.nan),
                     where=mass[..., None] > 0)


def wind_response(measured, uncorrected, regressor, future, estimate):
    corrected = estimate.copy()
    record = {}
    for site in range(measured.shape[1]):
        for lead in range(0, 96, 24):
            span = slice(lead, lead+24)
            reference = uncorrected[:, site, span]
            true = measured[:, site, span]
            x = regressor[:, site]
            mask = np.isfinite(true) & (true >= 0) & np.isfinite(reference) & np.isfinite(x).all(axis=-1)[:, None]
            fitted = np.zeros(2)
            if mask.any():
                predictors = np.broadcast_to(x[:, None, :], (*true.shape, 2))[mask]
                response = (true-reference)[mask]
                fitted = np.linalg.solve(predictors.T@predictors/len(response) + np.eye(2),
                                         predictors.T@response / len(response))
                update = np.nan_to_num(future[:, site] @ fitted, nan=0.)
                corrected[:, site, span] = np.maximum(estimate[:, site, span] + update[:, None], 0)
            record[f'{site}/{lead}'] = {'paired_points': int(mask.sum()), 'coefficients': fitted.tolist()}
    return corrected, record
