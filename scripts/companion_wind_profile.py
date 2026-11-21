"""Predeclared 10–100 metre wind shear/veer features from issued vectors."""

import numpy as np


SOURCES = ('tianji', 'aifs')
SHEAR = 'relative_shear'
VEER = 'relative_shear_veer'
PROFILE_COMPONENTS = (SHEAR, VEER)
SHEAR_NAMES = tuple(f'{source}_shear_exponent_10_100' for source in SOURCES)
VEER_NAMES = tuple(f'{source}_veer_{part}_10_100' for source in SOURCES for part in ('sin', 'cos'))
MIN_SPEED = 1.0


def vertical_profile(lower, upper):
    """Return the shear exponent and signed upper-minus-lower rotation."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    low_speed, high_speed = np.hypot(*lower.T), np.hypot(*upper.T)
    valid = (np.isfinite(low_speed) & np.isfinite(high_speed)
             & (low_speed >= MIN_SPEED) & (high_speed >= MIN_SPEED))
    result = np.full((len(lower), 3), np.nan, dtype=np.float32)
    low_u, low_v = (lower[valid] / low_speed[valid, None]).T
    high_u, high_v = (upper[valid] / high_speed[valid, None]).T
    result[valid] = np.column_stack((
        (np.log(high_speed[valid]) - np.log(low_speed[valid])) / np.log(10.0),
        low_u * high_v - low_v * high_u,
        low_u * high_u + low_v * high_v,
    ))
    return result


def profile_columns(inputs, column_map, mode):
    parts = []
    for source in SOURCES:
        lower, upper = (inputs[:, [column_map[f'{source}_{axis}{height}'] for axis in ('u', 'v')]]
                        for height in (10, 100))
        parts.append(vertical_profile(lower, upper))
    shear = np.column_stack([part[:, 0] for part in parts])
    if mode == SHEAR:
        return shear, SHEAR_NAMES, shear
    if mode == VEER:
        return np.column_stack((shear, *(part[:, 1:] for part in parts))), SHEAR_NAMES + VEER_NAMES, shear
    raise ValueError(f'Unknown wind profile component {mode!r}.')


def training_shear_thresholds(shear):
    thresholds = {}
    for i, source in enumerate(SOURCES):
        finite = np.abs(shear[np.isfinite(shear[:, i]), i])
        thresholds[source] = float(np.quantile(finite, 0.90)) if finite.size else None
    return thresholds
