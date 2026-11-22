#!/usr/bin/env python3
"""Verify and replay the adopted T1 v4, T2 v3 and T3 v3 on local windows."""
import argparse
import fcntl
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from batch_energy_short import roster_for, t3_windows
from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, chronological_dates, forecast_keys, sha256
from run_forecast_blends import blend, freeze_json, freeze_window
from run_intraday_residuals import t1_v4_reference
from run_phase_anchor import t2_v3_reference
from run_steamlink_followup import energy_for
from run_t1_issue_wind_error import dynamic_baseline
from run_t3_nearby_donors import v3_reference
from run_wind_attribution import read_arm
from score_forecasts import score_track
from advance_batch_b import keys_for
from warm_shared_power import labels


TRACKS = ('T1', 'T2', 'T3')
EXPECTED = {
    ('A', 'T1', 'A'): 87.80072396682495,
    ('A', 'T2', 'A'): 67.19750029596793,
    ('A', 'T3', 'inner_feb'): 86.89076329169495,
    ('A', 'T3', 'inner_mar'): 90.45629726707361,
    ('B', 'T1', 'B'): 88.59377282347674,
    ('B', 'T2', 'B'): 67.5741989772363,
    ('B', 'T3', 'fixed_50_10'): 90.0848486531422,
}
SOURCE_DIRS = {
    'phase_run': 'phase_anchor_cpu_v1', 'dynamic_run': 't1_dynamic_gradient_gpu_v1',
    't2_prior': 'steamlink_followup_gpu_v1', 'fusion_run': 'fusion_mechanisms_cpu_v1',
    'ridge_run': 'daily_bias_ridge_gpu_v1', 't3_confirm': 't3_transfer_confirmation_gpu_v1',
    'transfer_run': 'same_issue_followup_v1/T3', 'short_tree': 'three_track_gpu_batch_v1/T3',
}


def windows(split, tracks, phase):
    for track in tracks:
        if track == 'T3':
            short = list(t3_windows())
            yield from ((track, *window) for window in (short[:2] if phase == 'A' else short[-1:]))
        else:
            fit, days = chronological_dates(split, phase)
            if phase == 'B':
                first, a = chronological_dates(split, 'A')
                if fit != first+a:
                    raise ValueError(f'{track} B requires training plus every A day.')
            yield track, phase, fit, days


def reference(args, track, window, fit, days, sites, seed):
    if track == 'T1':
        keys = forecast_keys(days, sites)
        return t1_v4_reference(args, window, keys, seed)
    if track == 'T2':
        keys, calibrated, forecast, origin = t2_v3_reference(args, window, days, sites, seed)
        power, receipt = dynamic_baseline(args, window, forecast_keys(days, sites), seed)
        reconstructed = blend(calibrated, energy_for(power, keys), .10, 'T2')
        if receipt != origin['anchor'] or not np.allclose(reconstructed.energy, forecast.energy, rtol=0, atol=1e-6):
            raise ValueError('Recomputed T2 90/10 blend differs from the adopted forecast.')
        return forecast, origin['v3']
    _, _, _, forecast, lock = v3_reference(args, window, fit, days, sites, seed)
    return forecast, lock


def prediction_keys(track, days, sites):
    return keys_for(days, sites, 'T2') if track == 'T2' else forecast_keys(days, sites)


def ensemble_for(track, frames, keys):
    value = 'energy' if track == 'T2' else 'prediction'
    return keys.assign(**{value: np.mean([f[value].to_numpy() for f in frames.values()], axis=0)})


def predict(args, out, scopes, seeds):
    ledger = {}
    for track, window, fit, days in scopes:
        sites = roster_for(args.data_root, track)
        keys = prediction_keys(track, days, sites)
        frames, locks = {}, {}
        for seed in seeds:
            name = f'seed_{seed}'
            frames[name], locks[name] = reference(args, track, window, fit, days, sites, seed)
        frames['ensemble'] = ensemble_for(track, frames, keys)
        _, lock = freeze_window(out/track/window, track, keys, frames,
                                {'source_locks': locks, 'train_days': len(fit), 'score_days': len(days)})
        ledger[f'{track}/{window}'] = lock
        print(f'[{track} {window}] {len(keys)} keys, {len(seeds)} seed forecast(s) verified', flush=True)
    return freeze_json(out/'predictions_frozen.json', ledger)


def score(args, out, scopes, seeds):
    path = out/'predictions_frozen.json'
    ledger = read_json(path)
    expected_windows = {f'{track}/{window}' for track, window, _, _ in scopes}
    if set(ledger) != expected_windows:
        raise ValueError('Prediction ledger covers a different set of tracks or windows.')
    names = tuple(f'seed_{seed}' for seed in seeds)
    verified = {}
    # Verify every forecast in the request before opening the first label file.
    for track, window, _, days in scopes:
        folder = out/track/window
        lock = folder/'predictions_frozen.json'
        key = f'{track}/{window}'
        if sha256(lock) != ledger[key] or set(read_json(lock)['hashes']) != {*names, 'ensemble'}:
            raise ValueError(f'{key}: forecast lock changed.')
        sites = roster_for(args.data_root,track)
        keys = prediction_keys(track, days, sites)
        frames = {name: read_arm(folder, name, keys, track) for name in (*names,'ensemble')}
        ensemble = ensemble_for(track, {name:frames[name] for name in names}, keys)
        column = 'energy' if track == 'T2' else 'prediction'
        if not np.allclose(ensemble[column], frames['ensemble'][column], rtol=0, atol=1e-10):
            raise ValueError(f'{key}: ensemble does not equal the mean of the locked seed predictions.')
        verified[key] = sites,frames

    reports = {}
    for track, window, _, days in scopes:
        sites, frames = verified[f'{track}/{window}']
        truth = labels(args.data_root, days, sites)
        results = {name:score_track(track, truth, frame, sites, allow_unscored_stations=True)
                   for name,frame in frames.items()}
        target = EXPECTED[args.phase,track,window]
        observed = results['ensemble'][PRIMARY_SCORE]
        reports[f'{track}/{window}'] = {
            'expected_readme_score':target,'replayed_readme_score':observed,
            'delta_from_archived':observed-target,'matches_archived':abs(observed-target)<=1e-6,'scores':results}
        print(f'[{track} {window}] {observed:.8f}  archived {target:.8f}', flush=True)
    freeze_json(out/'scores.json',{'prediction_lock':sha256(path),'results':reports})
    return all(result['matches_archived'] for result in reports.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('predict','score'))
    parser.add_argument('--phase',choices=('A','B'),default='B')
    parser.add_argument('--tracks',nargs='+',choices=TRACKS,default=list(TRACKS))
    parser.add_argument('--data-root',type=Path,default=Path('/workspace/qhl/data'))
    parser.add_argument('--runs',type=Path,default=Path('/workspace/qhl/runs'))
    parser.add_argument('--out',type=Path,default=Path('/workspace/qhl/runs/current_baselines_replay_v1'))
    parser.add_argument('--split',type=Path,default=SPLIT_PATH)
    raw = parser.parse_args()
    if len(set(raw.tracks)) != len(raw.tracks):
        parser.error('--tracks must not contain duplicates')
    args = SimpleNamespace(**vars(raw), device='cuda', **{name: raw.runs / path for name,path in SOURCE_DIRS.items()})
    out = args.out / args.phase
    out.mkdir(parents=True,exist_ok=True)
    seeds = (A_SEED,) if args.phase == 'A' else B_SEEDS
    scopes = list(windows(args.split, args.tracks, args.phase))
    receipt = {'data_root':str(args.data_root.resolve()),'source_runs':str(args.runs.resolve()),
               'split':sha256(args.split),'phase':args.phase,'seeds':list(seeds),'code':sha256(Path(__file__)),
               'windows':{f'{track}/{window}':{'fit':list(map(str,fit)), 'score':list(map(str,days))}
                          for track,window,fit,days in scopes}}
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out/'plan.json'
        if args.stage == 'predict':
            freeze_json(path,receipt)
            forecast_lock = predict(args,out,scopes,seeds)
            print(f'Frozen predictions {out/"predictions_frozen.json"} ({forecast_lock})', flush=True)
            return 0
        if read_json(path) != receipt:
            raise ValueError('Scoring must use the same run and window plan as prediction.')
        return int(not score(args,out,scopes,seeds))


if __name__=='__main__':
    raise SystemExit(main())
