#!/usr/bin/env python3
"""Evaluate low-cost T1 daily bias correction and T3 ridge regularization."""

import argparse
import fcntl
from functools import partial
from pathlib import Path
import subprocess
from tempfile import mkdtemp
from time import monotonic
import traceback

import numpy as np
import pandas as pd

from advance_batch_b import read_frozen_forecast
from batch_energy_short import roster_for, t3_windows
from compare_t2_mlp import select_mlp_runtime
from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, chronological_dates, forecast_keys, sha256, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from run_pressure_profile import summarize_b, t3_stage
from run_steamlink_followup import confirmed_t3
from run_strict_warm_replay import replay_dates
from run_t1_issue_wind_error import dynamic_baseline
from run_t1_t3_batch import paired_report
from run_wind_attribution import STATION, read_arm
from same_issue_weather import archived_issue_weather, archive_stamp
from score_forecasts import DAY_KEYS, MAX_PREDICTION, MIN_VALID_POINTS, POINT_KEYS
from screen_companion_t1_stations import block_comparisons, show_progress
from screen_short_transfer import NETWORK, RIDGE, adjust_short, targets
from warm_shared_power import POWER_CAP, labels, load_roster


RUNS = Path('/workspace/qhl/runs')
T1_ARMS = ('station_bias', 'day_context_bias')
MAX_OFFSET = .75
RIDGE_NEW = .20


def daily_features(frame):
    return frame.groupby(DAY_KEYS, sort=False).prediction.agg(['mean','std']).reset_index()


def learn_bias(forecast, truth, sites):
    daily = daily_features(forecast)
    data = truth.merge(forecast, on=POINT_KEYS, validate='one_to_one')
    valid = data.power_limit.eq(1) & np.isfinite(data.power)
    residual = data.loc[valid].assign(error=lambda f: f.power - f.prediction)
    groups = residual.groupby(DAY_KEYS).error.agg(['mean','size']).rename(columns={'mean':'error'}).reset_index()
    fit = daily.merge(groups.loc[groups['size'].ge(MIN_VALID_POINTS)], on=DAY_KEYS, validate='one_to_one')
    fit = fit.loc[~fit.station_id.eq(STATION)]
    if fit.empty:
        raise ValueError('No eligible historical station-day for bias calibration.')
    station = (fit.station_id.to_numpy()[:, None] == np.asarray(sites)[None, :]).astype(float)
    context = fit[['mean','std']].to_numpy()
    center = context.mean(axis=0)
    scale = np.maximum(context.std(axis=0), 1e-3)
    normalized = (context-center)/scale

    def coefficients(matrix):
        return np.linalg.solve(matrix.T @ matrix + 20 * np.eye(matrix.shape[1]), matrix.T @ fit.error)

    return {'sites': list(sites), 'center': center.tolist(), 'scale': scale.tolist(),
            'station_beta': coefficients(station).tolist(),
            'context_beta': coefficients(np.column_stack((station, normalized))).tolist(),
            'n_days_by_station': {k: int(v) for k,v in fit.groupby('station_id').size().items()}}


def apply_bias(forecast, model, names):
    daily = daily_features(forecast)
    station = (daily.station_id.to_numpy()[:, None] == np.asarray(model['sites'])[None, :]).astype(float)
    context = (daily[['mean','std']].to_numpy()-model['center']) / model['scale']
    offset = {'station_bias': station @ model['station_beta'],
              'day_context_bias': np.column_stack((station, context)) @ model['context_beta']}
    for name in names:
        correction = np.where(daily.station_id.eq(STATION), 0., np.clip(offset[name], -MAX_OFFSET, MAX_OFFSET))
        delta = daily[DAY_KEYS].assign(offset=correction)
        rows = forecast.merge(delta, on=DAY_KEYS, how='left', validate='many_to_one', sort=False)
        yield name, forecast.assign(prediction=(rows.prediction + rows.offset).clip(0, MAX_PREDICTION['T1']).to_numpy())


def t1_stage(args):
    out = args.out / 'T1'
    sites = load_roster(args.data_root)
    fit, days = chronological_dates(args.split, 'A')
    historical_fit, historical = replay_dates(args.split)
    if not set(historical_fit + historical).issubset(fit):
        raise ValueError('T1 calibration dates escape the current A fit.')
    history = args.warm_run / 'T1'
    source = history / 'forecasts'
    lock = sha256(source / 'predictions_frozen.json')
    if read_json(history / 'score.json')['prediction_lock'] != lock:
        raise ValueError('Warm replay baseline is not associated with its saved score.')
    contract = read_json(args.warm_run / 'plan.json')['dates']
    if (contract['train'] != list(map(str, historical_fit)) or
            contract['forecast'] != list(map(str, historical)) or
            contract['data_root'] != str(args.data_root.resolve())):
        raise ValueError('Wrong strict warm replay source dates or root.')
    hist_keys = forecast_keys(historical, sites)
    base = read_arm(source, 'baseline', hist_keys, 'T1')
    model = learn_bias(base, labels(args.data_root, historical, sites), sites)
    stamp = freeze_json(out / 'bias_fit.json', {'params': model, 'warm_source': lock, 'max_offset_mw': MAX_OFFSET})

    def locked_pair(phase, seed, keys, names):
        base, base_lock = dynamic_baseline(args, phase, keys, seed)
        return {'baseline': base, **dict(apply_bias(base, model, names))}, {
            'reference_lock': base_lock, 'bias_fit': stamp}

    keys = forecast_keys(days, sites)
    frames, config = locked_pair('A', A_SEED, keys, T1_ARMS)
    frames, lock_a = freeze_window(out / 'A/forecasts', 'T1', keys, frames, config)
    a_details = block_comparisons(labels(args.data_root, days, sites), frames, sites, T1_ARMS)
    a = {arm: row[PRIMARY_SCORE] for arm, row in a_details['scores'].items()}
    winner = max(T1_ARMS, key=a.get)
    gain = a[winner] - a['baseline']
    gate = {'candidate': winner, 'delta': gain, 'promoted': gain > 0., 'forecast_lock': lock_a}
    freeze_json(out / 'A/score.json', {**a_details, 'gate': gate})
    print(f'[T1 A] {a}; {gate}', flush=True)
    result = {'A': a, 'gate': gate}
    if gate['promoted']:
        fit_b, days_b = chronological_dates(args.split, 'B')
        if fit_b != fit + days:
            raise ValueError('T1 B fit must add all A dates.')
        keys = forecast_keys(days_b, sites)
        pairs, locks = {}, {}
        for seed in B_SEEDS:
            arms, config = locked_pair('B', seed, keys, (winner,))
            arms['candidate'] = arms.pop(winner)
            pairs[seed], locks[str(seed)] = freeze_window(out / 'B' / str(seed), 'T1', keys, arms, {**config,'gate':gate})
        lock_b = freeze_json(out / 'B/predictions_frozen.json', locks)
        result['B'] = summarize_b(paired_report(out / 'B', 'T1', labels(args.data_root, days_b, sites),
                                                sites, keys, pairs, lock_b))
    return result


def encoder_source(args, window, seed):
    if seed == A_SEED:
        directory = args.transfer_run / window
        record = directory / 'predictions_frozen.json'
        payload = read_json(record)
        return directory / 'frozen_encoder.pt', record, payload['fit'], payload['plan']
    directory = args.t3_confirm / 'T3' / str(seed)
    root = directory / 'predictions_frozen.json'
    record = directory / 'transfer' / 'receipt.json'
    value = read_json(record)
    expected = read_json(root)['sources']['fit']['transfer']
    if value != expected or value['seed'] != seed:
        raise ValueError('T3 frozen encoder receipt differs from the seed lock.')
    return directory / 'transfer' / 'frozen_encoder.pt', record, value['fit'], None


def worker(args):
    import torch

    _, train, days = next(w for w in t3_windows() if w[0] == args.worker_window)
    sites = roster_for(args.data_root, 'T3')
    checkpoint, receipt, fit, plan = encoder_source(args, args.worker_window, args.seed)
    if plan:
        if (plan['fit_dates'] != list(map(str, train)) or plan['score_dates'] != list(map(str, days))
                or plan['seed'] != args.seed or plan['target_sites'] != sites or plan['ridge'] != RIDGE):
            raise ValueError('T3 stored encoder has a different fit contract.')
    torch.set_num_threads(4)
    weather, _, cache = archived_issue_weather(args.workdir, args.data_root, train + days, sites, print)
    stamp = archive_stamp(args.workdir, train + days)
    if stamp['weather'] != cache:
        raise ValueError('T3 weather cache changed during the ridge replay.')
    mean, scale = (np.asarray(fit['source_normalization'][name]) for name in ('mean','scale'))
    valid = np.isfinite(weather)
    normalized = np.concatenate([np.where(valid, (weather-mean)/scale, 0.), ~valid], axis=-1).astype(np.float32)
    model = NETWORK(normalized.shape[-1]).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    model.eval()
    with torch.inference_mode():
        raw = [torch.sigmoid(model(torch.as_tensor(part, device=args.device)).squeeze(-1)).cpu().numpy()
               for part in np.array_split(normalized, max(1, (len(normalized)+63)//64))]
    raw = np.clip(np.concatenate(raw).astype(float) * POWER_CAP, 0, POWER_CAP)
    actual, mask = targets(args.data_root, train, sites)
    cut = len(train)*len(sites)
    keys = forecast_keys(days, sites)
    hashes, norms = {}, {}
    for strength in (RIDGE, RIDGE_NEW):
        frame, heads = adjust_short(model, lambda: raw, normalized[:cut], normalized[cut:], actual, mask, sites,
                                    args.device, ridge=strength)
        name = f'ridge_{strength:.2f}'
        path = args.out / f'{name}.csv'
        keys.assign(prediction=frame.ravel()).to_csv(path, index=False)
        hashes[name] = sha256(path)
        norms[name] = {site: float(np.linalg.norm(rows['coefficients'][:-1]))
                       for site, rows in heads.items() if rows['points']}
    write_json(args.out / 'worker_receipt.json', {'checkpoint': sha256(checkpoint), 'old_fit_receipt': sha256(receipt),
                                                 'weather': stamp, 'hashes': hashes, 'coefficient_norms': norms})


def t3_pair(args, window, train, days, short, _all_sites, _source, seed):
    keys, tree, neural, baseline, source = confirmed_t3(args, window, train, days, short, seed)
    target = args.out / 'T3' / window / str(seed)
    target.mkdir(parents=True, exist_ok=True)
    saved = target / 'heads'
    checkpoint, prior_receipt, _, _ = encoder_source(args, window, seed)
    if not saved.is_dir():
        pending = Path(mkdtemp(dir=target, prefix='.heads-attempt-'))
        options = [str(args.torch_python), '-u', str(Path(__file__).resolve()), '--out', str(pending),
                   '--data-root', str(args.data_root), '--workdir', str(args.workdir),
                   '--device', args.device, '--transfer-run', str(args.transfer_run), '--split', str(args.split),
                   '--t3-confirm', str(args.t3_confirm), '--worker-window', window, '--seed', str(seed)]
        subprocess.run(options, check=True)
        pending.rename(saved)
    receipt = read_json(saved / 'worker_receipt.json')
    if (receipt['old_fit_receipt'] != sha256(prior_receipt) or receipt['checkpoint'] != sha256(checkpoint)
            or receipt['weather'] != archive_stamp(args.workdir, train + days)):
        raise ValueError('T3 candidate ridge replay no longer matches its inputs.')

    def old_or_new(strength):
        name = f'ridge_{strength:.2f}'
        return read_frozen_forecast(saved / f'{name}.csv', receipt['hashes'][name], keys, 'T3')

    gap = float(np.max(np.abs(old_or_new(RIDGE).prediction-neural.prediction)))
    if gap > 2e-4:
        raise ValueError(f'T3 ridge baseline replay differs from frozen predictions by {gap:.6f} MW.')
    arms = {'baseline': baseline, 'candidate': blend(tree, old_or_new(RIDGE_NEW), .25, 'T3')}
    locked, digest = freeze_window(target / 'forecasts', 'T3', keys, arms, {
        'reference_lock': source, 'fit_receipt': sha256(saved/'worker_receipt.json'),
        'old_max_abs_reproduction_error_mw': gap})
    return keys, locked, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out': RUNS/'daily_bias_ridge_gpu_v1', 'data-root': Path('/workspace/qhl/data'),
        'workdir': Path('/workspace/cpy/code/workdir'), 'companion-code': Path('/workspace/cpy/code'),
        'dynamic-run': RUNS/'t1_dynamic_gradient_gpu_v1', 'warm-run': RUNS/'strict_warm_replay_gpu_v1',
        'transfer-run': RUNS/'same_issue_followup_v1/T3', 'short-tree': RUNS/'three_track_gpu_batch_v1/T3',
        't3-confirm': RUNS/'t3_transfer_confirmation_gpu_v1', 'split': SPLIT_PATH, 'mlp-run': RUNS/'warm_shared_power_v1'}
    for key, val in defaults.items():
        parser.add_argument(f'--{key}', type=Path, default=val)
    for name in ('config', 'mlp-python', 'mlp-weather-python'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1,5), default=4)
    parser.add_argument('--worker-window', choices=tuple(w[0] for w in t3_windows()))
    parser.add_argument('--seed', type=int, choices=B_SEEDS, default=A_SEED)
    args = parser.parse_args()
    if args.worker_window:
        worker(args)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start, result = monotonic(), {}
        freeze_json(args.out/'plan.json', {
            'inputs': {name: str(getattr(args, name).resolve()) for name in (
                'data_root', 'workdir', 'dynamic_run', 'warm_run', 'transfer_run', 'short_tree', 't3_confirm')},
            'split_sha256': sha256(args.split), 'seeds': list(B_SEEDS), 'device': args.device,
            'daily_bias_penalty': 20, 'max_offset_mw': MAX_OFFSET,
            'ridge': [RIDGE, RIDGE_NEW], 'code': {name: sha256(Path(__file__).with_name(name))
                for name in ('run_daily_bias_ridge.py', 'screen_short_transfer.py', 'run_strict_warm_replay.py')}})

        def short_stage(args):
            args.torch_python = select_mlp_runtime(args, require_weather=False)[0]
            return t3_stage(args, predict_pair=t3_pair, title='T3 岭正则')

        stages = (('T1', t1_stage), ('T3', short_stage))
        for index, (track, stage) in enumerate(stages):
            show_progress(args.out, '日均与岭头', index, len(stages), track)
            (args.out/track).mkdir(exist_ok=True)
            try:
                result[track] = {'status':'complete', **stage(args)}
            except Exception as exc:
                traceback.print_exc()
                result[track] = {'status':'failed','error':str(exc)}
            write_json(args.out/'summary.json', result)
        result['seconds'] = round(monotonic()-start,2)
        write_json(args.out/'summary.json', result)
        show_progress(args.out, '日均与岭头', len(stages), len(stages), '已保存')
        print(f'结果摘要 {args.out / "summary.json"}', flush=True)
        return int(any(result[t]['status'] != 'complete' for t,_ in stages))


if __name__ == '__main__':
    raise SystemExit(main())
