"""Chronology-safe, station-shrunk monotone power curves for companion T1."""

from dataclasses import dataclass

import numpy as np


FEATURE, RESIDUAL = 'power_curve_feature', 'power_curve_residual'
CURVE_COMPONENTS = (FEATURE, RESIDUAL)
CURVE_COLUMN = 'station_power_curve_prior'
SOURCES = ('tianji', 'ec', 'aifs')
BINS = np.arange(0.25, 30.51, 0.5)
SHRINKAGE = 32.0


def mean_wind(matrix, names):
    fields = [names.index(f'{source}_{axis}100') for source in SOURCES for axis in ('u', 'v')]
    vectors = matrix[:, fields].reshape(len(matrix), len(SOURCES), 2)
    valid = np.isfinite(vectors).all(axis=2)
    safe = np.where(valid[..., None], vectors, 0)
    speed = np.hypot(safe[..., 0], safe[..., 1])
    count = valid.sum(axis=1)
    return np.divide(speed.sum(axis=1), count, out=np.full(len(matrix), np.nan), where=count > 0)


def monotone(values, weights):
    """Weighted least-squares projection onto increasing values."""
    weights = list(map(float, weights))
    totals = list(np.asarray(values, dtype=float) * weights)
    pooled = []
    for index, (weight, total) in enumerate(zip(weights, totals, strict=True)):
        pooled.append((index, index + 1, weight, total))
        while len(pooled) > 1 and pooled[-2][3] / pooled[-2][2] > pooled[-1][3] / pooled[-1][2]:
            _, end, right_weight, right_total = pooled.pop()
            start, _, left_weight, left_total = pooled.pop()
            pooled.append((start, end, left_weight + right_weight, left_total + right_total))
    result = np.empty(len(values))
    for start, end, weight, total in pooled:
        result[start:end] = total / weight
    return result


def fit_curve(wind, sites, target, n_sites):
    n_bins = len(BINS)
    valid = np.isfinite(wind)
    bins = np.minimum((wind[valid] / 0.5).astype(np.intp), n_bins - 1)
    flat = sites[valid] * n_bins + bins
    counts = np.bincount(flat, minlength=n_sites * n_bins).reshape(n_sites, n_bins)
    totals = np.bincount(flat, weights=target[valid], minlength=n_sites * n_bins).reshape(n_sites, n_bins)
    size = np.bincount(sites, minlength=n_sites)
    site_total = np.bincount(sites, weights=target, minlength=n_sites)
    site_weight = 1 / np.maximum(size, 1)
    global_mean = float((site_total * site_weight).sum() / (size > 0).sum())
    count, total = site_weight @ counts, site_weight @ totals
    adequate = count > 0
    if adequate.any():
        shared = np.interp(BINS, BINS[adequate], monotone(total[adequate] / count[adequate], count[adequate]))
    else:
        shared = np.full(n_bins, global_mean)
    weight = counts + SHRINKAGE
    values = (totals + shared * SHRINKAGE) / weight
    curves = np.stack([monotone(row, row_weights) for row, row_weights in zip(values, weight, strict=True)])
    missing = (site_total + SHRINKAGE * global_mean) / (size + SHRINKAGE)
    return curves, missing


def apply_curve(curve, wind, sites):
    values, fallback = curve
    estimates = fallback[sites].copy()
    for site, row in enumerate(values):
        mask = (sites == site) & np.isfinite(wind)
        estimates[mask] = np.interp(wind[mask], BINS, row)
    return estimates


def out_of_fold(wind, sites, target, dates, n_sites):
    unique = np.unique(dates)
    if len(unique) < 3:
        raise ValueError('Power-curve cross-fitting requires at least three fit dates.')
    values = np.empty(len(target))
    for held_dates in np.array_split(unique, 3):
        held = np.isin(dates, held_dates)
        fit = ~held
        curve = fit_curve(wind[fit], sites[fit], target[fit], n_sites)
        values[held] = apply_curve(curve, wind[held], sites[held])
    return values


@dataclass
class CurveInputs:
    selection: np.ndarray
    refit: np.ndarray
    forecast: np.ndarray
    frozen: tuple[np.ndarray, np.ndarray]


def prepare_curves(wind, sites, power, dates, inner, prediction_wind, prediction_sites, n_sites):
    earlier = ~inner
    selector = fit_curve(wind[earlier], sites[earlier], power[earlier], n_sites)
    selection = np.empty(len(power))
    selection[earlier] = out_of_fold(wind[earlier], sites[earlier], power[earlier], dates[earlier], n_sites)
    selection[inner] = apply_curve(selector, wind[inner], sites[inner])
    refit = out_of_fold(wind, sites, power, dates, n_sites)
    final = fit_curve(wind, sites, power, n_sites)
    return CurveInputs(selection, refit, apply_curve(final, prediction_wind, prediction_sites), final)
