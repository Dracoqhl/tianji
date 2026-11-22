#!/usr/bin/env python3
"""Screen weather-conditioned T1 residuals and T3 short-period shape transfer."""
import argparse
import fcntl
from functools import partial
from pathlib import Path
from time import monotonic
import traceback

import numpy as np

from experiment_companion_t1 import read_json
from run_companion_warm_t1 import B_SEEDS, SPLIT_PATH, forecast_keys, sha256, write_json
from run_forecast_blends import freeze_json
from run_fusion_mechanisms import disagreement, finish_pending, stage
from run_phase_anchor import T1_ARMS as PHASE_ARMS, historical_reference, phase_forecasts
from run_steamlink_followup import finite_mean
from run_t1_issue_wind_error import dynamic_baseline
from run_t3_nearby_donors import v3_reference
from run_wind_attribution import SOURCES, STATION, read_arm
from same_issue_weather import archived_issue_weather
from score_forecasts import MAX_PREDICTION, MIN_VALID_POINTS, POINT_KEYS
from screen_companion_t1_stations import show_progress
from warm_shared_power import labels


RUNS = Path('/workspace/qhl/runs')
T1_ARMS = ('station_shape', 'weather_conditioned_shape')
T3_ARMS = {'fast_shape_25': .25, 'fast_shape_50': .50}
FAST_WINDOW = 9
N_BASIS = 6
N_POINTS = 96
BASIS = np.sqrt(2) * np.cos(np.pi * (np.arange(N_POINTS)[:, None] + .5)
                           * np.arange(1, N_BASIS+1)[None, :] / N_POINTS)
MULTIPLIER = .50
POINT_LIMIT = 1.
PENALTY = .10
CONTEXT = ('mean_forecast_mw', 'sd_forecast_mw', 'mean_wind', 'wind_tail_minus_head',
           'mean_source_range', 'wind_hourly_change', 'source_coverage')


def weather_summary(data, columns):
    speed = np.stack([np.hypot(data[..., columns.index(f'{source}_u100')],
                               data[..., columns.index(f'{source}_v100')]) for source in SOURCES])
    pooled = finite_mean(speed)
    def mean_points(values):
        return finite_mean(values.T)
    return np.column_stack((mean_points(pooled),
                            mean_points(pooled[:, -24:]) - mean_points(pooled[:, :24]),
                            mean_points(disagreement(speed)),
                            mean_points(np.abs(pooled[:, 4:] - pooled[:, :-4])),
                            np.isfinite(speed).mean(axis=(0, 2))))


def weather_context(args, days, sites):
    data, columns, lock = archived_issue_weather(args.workdir, args.data_root, days, sites, print)
    return weather_summary(data, columns), lock


def context_matrix(forecast, summary):
    curve = forecast.prediction.to_numpy().reshape(-1, N_POINTS)
    return np.column_stack((curve.mean(axis=1), curve.std(axis=1), summary))


def standardized(matrix, mean, scale):
    return np.clip(np.where(np.isfinite(matrix), (matrix-mean)/scale, 0.), -3, 3)


def project_history(history, truth):
    aligned = history.merge(truth[POINT_KEYS + ['power', 'power_limit']], on=POINT_KEYS, how='left',
                            sort=False, validate='one_to_one', indicator=True)
    if not aligned['_merge'].eq('both').all():
        raise ValueError('T1 residual truth and replay predictions do not have identical keys.')
    sites = aligned.station_id.iloc[::N_POINTS].to_numpy()
    actual = aligned.power.to_numpy().reshape(-1, N_POINTS)
    mask = np.isfinite(actual) & aligned.power_limit.eq(1).to_numpy().reshape(actual.shape)
    kept = (mask.sum(axis=1) >= MIN_VALID_POINTS) & (sites != STATION)
    if not kept.any():
        raise ValueError('No eligible station-day for T1 shape regression.')
    valid = mask[kept]
    residual = (actual - aligned.prediction.to_numpy().reshape(actual.shape))[kept]
    inverse_count = 1 / valid.sum(axis=1)
    basis_mean = (valid @ BASIS) * inverse_count[:, None]
    centered = BASIS[None, :, :] - basis_mean[:, None, :]
    gram = np.einsum('dti,dtj,dt,d->dij', centered, centered, valid, inverse_count)
    rhs = np.einsum('dti,dt,d->di', centered, np.where(valid, residual, 0), inverse_count)
    coefficients = np.linalg.solve(gram + .02 * np.eye(N_BASIS), rhs[..., None])[..., 0]
    return sites, kept, coefficients


def learn_residuals(history, truth, summary, roster):
    sites, kept, coefficients = project_history(history, truth)
    station = (sites[kept, None] == np.asarray(roster)[None, :]).astype(float)
    counts = station.sum(axis=0)
    weights = 1 / (station @ counts)
    weights /= weights.sum()
    matrix = context_matrix(history, summary)[kept]
    finite = np.isfinite(matrix)
    size = finite.sum(axis=0).clip(1)
    mean = np.where(finite, matrix, 0).sum(axis=0) / size
    scale = np.sqrt(np.where(finite, (matrix-mean)**2, 0).sum(axis=0) / size).clip(.001)
    features = np.column_stack((station, standardized(matrix, mean, scale)))

    def ridge(design):
        lhs = design.T @ (design * weights[:, None]) + PENALTY * np.eye(design.shape[1])
        return np.linalg.solve(lhs, design.T @ (coefficients * weights[:, None])).tolist()

    return {'roster': list(roster), 'context_columns': list(CONTEXT), 'center': mean.tolist(), 'scale': scale.tolist(),
            'station_shape': ridge(station), 'weather_conditioned_shape': ridge(features),
            'eligible_by_station': {site: int(n) for site, n in zip(roster, counts)}}


def apply_residuals(base, summary, model, names):
    sites = base.station_id.iloc[::N_POINTS].to_numpy()
    station = (sites[:, None] == np.asarray(model['roster'])[None, :]).astype(float)
    context = standardized(context_matrix(base, summary), model['center'], model['scale'])
    arms = {}
    for name in names:
        features = station if name == T1_ARMS[0] else np.column_stack((station, context))
        delta = MULTIPLIER * (features @ model[name]) @ BASIS.T
        limit = np.maximum(1, np.abs(delta).max(axis=1) / POINT_LIMIT)
        delta = np.where(sites[:, None] == STATION, 0., delta / limit[:, None])
        value = np.clip(base.prediction.to_numpy() + delta.ravel(), 0, MAX_PREDICTION['T1'])
        arms[name] = base.assign(prediction=value)
    return arms


def t1_v4_reference(args, phase, keys, seed):
    old, old_lock = dynamic_baseline(args, phase, keys, seed)
    root = args.phase_run / 'T1'
    gate = read_json(root / 'gate.json')
    winner = PHASE_ARMS[1]
    if gate['selected'] != winner or not gate['promoted']:
        raise ValueError('T1 reference requires the adopted symmetric smoothing.')
    if phase == 'A':
        folder = root / 'A/forecasts'
        expected = gate['screen_locks']['A']
        name = winner
    else:
        folder = root / 'B' / str(seed)
        ledger = root / 'B/predictions_frozen.json'
        report = read_json(root / 'B/b_score.json')
        if not report['acceptance']['accepted'] or report['prediction_lock_sha256'] != sha256(ledger):
            raise ValueError('T1 v4 B reference does not have an accepted and verified prediction lock.')
        expected = read_json(ledger)[str(seed)]
        name = 'candidate'
    marker = folder / 'predictions_frozen.json'
    lock = sha256(marker)
    meta = read_json(marker)['sources']
    if phase == 'B':
        if meta['gate'] != gate:
            raise ValueError('T1 v4 B predictions use a different gate.')
        meta = meta['source']
    fit = root / 'phase_fit.json'
    if lock != expected or meta != {'v3': old_lock, 'phase': sha256(fit)}:
        raise ValueError('T1 v4 forecast is not linked to its original source and calibration.')
    current = read_arm(folder, name, keys, 'T1')
    replay = phase_forecasts(old, read_json(fit)['sites'], (winner,))[winner]
    if not np.allclose(current.prediction, replay.prediction, rtol=0, atol=1e-6):
        raise ValueError('T1 smoothing replay does not reproduce v4.')
    return current, lock


def t1_study(args):
    fitted = None
    by_phase = {}

    def predict(phase, train, days, sites, seed, names):
        nonlocal fitted
        if fitted is None:
            historical_days, history, origin = historical_reference(args, train, sites)
            weather, cache = weather_context(args, historical_days, sites)
            model = learn_residuals(history, labels(args.data_root, historical_days, sites), weather, sites)
            fitted = model, freeze_json(args.out / 'T1/residual_fit.json',
                                       {'replay': origin, 'weather_cache': cache, 'model': model})
        if phase not in by_phase:
            by_phase[phase] = weather_context(args, days, sites)
        summary, weather = by_phase[phase]
        keys = forecast_keys(days, sites)
        base, lock = t1_v4_reference(args, phase, keys, seed)
        model, stamp = fitted
        return keys, {'baseline': base, **apply_residuals(base, summary, model, names)}, {
            'v4_reference': lock, 'fit': stamp, 'weather_cache': weather}

    return predict, T1_ARMS


def fast_shape(difference):
    pad = FAST_WINDOW // 2
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(difference, ((0, 0), (pad, pad)), mode='edge'),
                                                       FAST_WINDOW, axis=1)
    fast = difference - windows.mean(axis=-1)
    return fast - fast.mean(axis=1, keepdims=True)


def t3_predict(args, window, train, days, sites, seed, names):
    keys, tree, ridge, base, lock = v3_reference(args, window, train, days, sites, seed)
    difference = (ridge.prediction - tree.prediction).to_numpy().reshape(-1, N_POINTS)
    fast = fast_shape(difference).ravel()
    arms = {'baseline': base}
    for name in names:
        candidate = base.prediction + (T3_ARMS[name] - .375) * fast
        arms[name] = base.assign(prediction=candidate.clip(0, MAX_PREDICTION['T3']))
    return keys, arms, {'v3_reference': lock, 'fast_window': FAST_WINDOW}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out': RUNS / 'intraday_residuals_cpu_v1', 'data-root': Path('/workspace/qhl/data'),
                'workdir': Path('/workspace/cpy/code/workdir'), 'split': SPLIT_PATH, 'phase-run': RUNS / 'phase_anchor_cpu_v1',
                'dynamic-run': RUNS / 't1_dynamic_gradient_gpu_v1', 'warm-run': RUNS / 'strict_warm_replay_gpu_v1',
                'ridge-run': RUNS / 'daily_bias_ridge_gpu_v1', 'fusion-run': RUNS / 'fusion_mechanisms_cpu_v1',
                'short-tree': RUNS / 'three_track_gpu_batch_v1/T3', 'transfer-run': RUNS / 'same_issue_followup_v1/T3',
                't3-confirm': RUNS / 't3_transfer_confirmation_gpu_v1'}
    for name, value in defaults.items():
        parser.add_argument(f'--{name}', type=Path, default=value)
    parser.add_argument('--device', choices=('cuda',), default='cuda', help='Expected device of frozen source models.')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_json(args.out / 'plan.json', {
            'args': {k: str(v) for k, v in vars(args).items()}, 'split': sha256(args.split), 'seeds': list(B_SEEDS),
            'basis': N_BASIS, 'ridge': PENALTY, 'multiplier': MULTIPLIER, 'limit_mw': POINT_LIMIT,
            'fast_window': FAST_WINDOW, 'shape_fractions': T3_ARMS,
            'code': {name: sha256(Path(__file__).with_name(name)) for name in
                     ('run_intraday_residuals.py', 'run_phase_anchor.py', 'run_fusion_mechanisms.py',
                      'run_t3_nearby_donors.py')}})
        studies = {'T1': t1_study(args), 'T3': (partial(t3_predict, args), tuple(T3_ARMS))}
        start, result, pending = monotonic(), {}, {}
        for i, (track, study) in enumerate(studies.items()):
            show_progress(args.out, '日内残差', i, len(studies), track)
            try:
                result[track] = {'status': 'complete', **stage(args, track, pending, study=study)}
            except Exception as error:
                traceback.print_exc()
                result[track] = {'status': 'failed', 'error': str(error)}
            write_json(args.out / 'summary.json', result)
        finish_pending(args, result, pending)
        result['seconds'] = round(monotonic() - start, 2)
        write_json(args.out / 'summary.json', result)
        show_progress(args.out, '日内残差', len(studies), len(studies), '结果已保存')
        print(f'结果摘要 {args.out / "summary.json"}', flush=True)
        return int(any(result[track]['status'] != 'complete' for track in studies))


if __name__ == '__main__':
    raise SystemExit(main())
