#!/usr/bin/env python3
"""Register and run the upwind terrain T1 comparison and its automatic B gate."""

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from advance_batch_b import before_scoring, choose_gate, keys_for, lock_predictions, make_t1, paired_scores
from companion_upwind_terrain import COMPONENT, coverage
from compare_t2_mlp import candidate_pythons
from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import code_version, event, load_run, read_json
from run_companion_warm_t1 import A_SEED, SPLIT_PATH, date_roles, forecast_keys, sha256, write_json
from score_forecasts import DAY_KEYS, MIN_VALID_POINTS, POINT_KEYS
from screen_companion_t1_stations import A_REPORT, FREEZE, confirm_a, frozen_pooled_forecast, frozen_pooled_origin
from screen_companion_t1_stations import screen, show_progress, study_setup
from warm_shared_power import labels, load_roster


SCRIPT_DIR = Path(__file__).resolve().parent


def lookup_for_run(args):
    lookup = args.out / 'terrain_lookup.npz'
    if not lookup.is_file():
        interpreters = [args.terrain_python] if args.terrain_python else candidate_pythons(args)
        seen = set()
        for item in interpreters:
            if not item:
                continue
            python = Path(item).absolute()
            if python in seen or not python.is_file():
                continue
            seen.add(python)
            argv = [str(python), '-u', str(SCRIPT_DIR / 'companion_upwind_terrain.py'),
                    '--data-root', str(args.data_root), '--out', str(lookup)]
            try:
                result = subprocess.run(argv, capture_output=True, text=True, timeout=45)
            except (OSError, subprocess.TimeoutExpired) as error:
                print(f'[DEM] {python} {type(error).__name__}', flush=True)
                continue
            if result.returncode == 0 and lookup.is_file():
                print(f'[DEM] 使用 {python} 读取静态高程', flush=True)
                break
            lines = result.stderr.strip().splitlines()
            detail = lines[-1] if lines else f'退出码 {result.returncode}'
            print(f'[DEM] {python} {detail}', flush=True)
        if not lookup.is_file():
            raise RuntimeError('未找到能读取 terrain.nc 的环境。可用 --terrain-python 指定已有 Python。')
    report = coverage(lookup, args.data_root)
    report['lookup_sha256'] = sha256(lookup)
    write_json(args.out / 'terrain_coverage.json', report)
    for site, distances in report['sites'].items():
        print(f'[DEM] {site} 覆盖方向数/360、相对高差范围 m {distances}', flush=True)
    return lookup


def changes(frame, values):
    return values.groupby([frame[name] for name in DAY_KEYS], sort=False).diff()


def diagnostics(history, truth, baseline, candidate, weather):
    """Use observed ramp outcomes solely for error grouping after forecasts are saved."""
    observed_history = history.power.where(history.power_limit.eq(1) & np.isfinite(history.power))
    thresholds = changes(history, observed_history).abs().groupby(history.station_id).quantile(0.90).dropna()
    valid = truth.power_limit.eq(1) & np.isfinite(truth.power)
    scored = valid & valid.groupby([truth[name] for name in DAY_KEYS], sort=False).transform('sum').ge(MIN_VALID_POINTS)
    actual = truth.power.where(scored)
    sector_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    directions = pd.Series('missing_or_weak', index=truth.index)
    has_wind = np.isfinite(weather.bearing)
    directions.loc[has_wind] = np.take(sector_names, np.floor((weather.bearing.loc[has_wind] + 22.5) / 45)
                                     .to_numpy(dtype=int) % 8)
    baseline_error = baseline.prediction - actual
    terrain_error = candidate.prediction - actual
    actual_delta = changes(truth, actual)
    absolute_change = actual_delta.abs()
    ramp_cut = truth.station_id.map(thresholds)
    large_ramp = absolute_change.gt(0) & absolute_change.ge(ramp_cut)
    adjacent = absolute_change.notna() & ramp_cut.notna()

    def summarize(mask, *, ramp=False):
        if not mask.any():
            return {'points': 0}
        result = {'points': int(mask.sum()), 'station_days': len(truth.loc[mask, DAY_KEYS].drop_duplicates()),
                  'terrain_both_available': int(weather.loc[mask, ['h3', 'h9']].notna().all(axis=1).sum()),
                  'baseline_rmse_mw': float(np.sqrt((baseline_error.loc[mask]**2).mean())),
                  'terrain_rmse_mw': float(np.sqrt((terrain_error.loc[mask]**2).mean()))}
        if ramp:
            for name, frame in (('baseline', baseline), ('terrain', candidate)):
                error = changes(truth, frame.prediction) - actual_delta
                result[f'{name}_change_mae_mw'] = float(error.loc[mask].abs().mean())
        return result

    return {'wind_sector': {sector: summarize(scored & directions.eq(sector)) for sector in [
                *sector_names, 'missing_or_weak']},
            'ramps': {'large': summarize(adjacent & large_ramp, ramp=True),
                      'other': summarize(adjacent & ~large_ramp, ramp=True)},
            'training_p90_abs_15min_change_mw': thresholds.to_dict()}


def report_a_diagnostics(args, a):
    train, days = date_roles(args.split, 'A')
    roster = load_roster(args.data_root)
    keys = forecast_keys(days, roster)
    directory = args.out / 'a' / COMPONENT
    receipt, candidate, _ = load_run(directory, keys, 'A', A_SEED, [COMPONENT], backend=args.backend, device=args.device)
    if (receipt != a['training'][COMPONENT] or receipt[COMPONENT]['lookup_sha256'] != sha256(args.terrain_lookup)):
        raise ValueError('The A terrain receipt differs from the frozen comparison.')
    diagnostic_file = directory / receipt['diagnostic']['file']
    digest = sha256(diagnostic_file)
    if digest != receipt['diagnostic']['sha256']:
        raise ValueError('The A terrain diagnostic feature hash changed.')
    raw_keys = pd.read_csv(directory / receipt['prediction_file'], usecols=POINT_KEYS,
                           dtype={'station_id': str, 'date': str})
    values = np.load(diagnostic_file, allow_pickle=False)
    if values.shape != (len(keys), 3):
        raise ValueError('The A terrain diagnostics do not cover the full forecast grid.')
    weather = keys.merge(raw_keys.assign(bearing=values[:, 0], h3=values[:, 1], h9=values[:, 2]),
                         how='left', on=POINT_KEYS, validate='one_to_one')
    origin = frozen_pooled_origin(args.baseline_run)
    _, baseline, _ = frozen_pooled_forecast(args.baseline_run, origin, 'A', train, days, keys, device=args.device)
    report_path = args.out / 'A_terrain_diagnostics.json'
    hashes = {'a_report_sha256': sha256(args.out / A_REPORT), 'features_sha256': digest}
    if report_path.is_file():
        stored = read_json(report_path)
        if stored['hashes'] != hashes:
            raise ValueError('The existing A diagnostic report was computed from another prediction.')
        return stored
    truth = labels(args.data_root, days, roster)
    history = labels(args.data_root, train, roster)
    result = diagnostics(history, truth, baseline, candidate, weather)
    result['hashes'] = hashes
    write_json(report_path, result)
    return result


def b_gate(a, freeze, roster):
    scores = {'baseline': a['scores']['baseline'], 'station_routes': a['scores']['candidate'],
              'all_stations': a['full_terrain']['scores']['candidate']}
    gate = choose_gate(scores, 'baseline', ('baseline', 'station_routes', 'all_stations'))
    gate['routes'] = (dict.fromkeys(roster, COMPONENT) if gate['candidate'] == 'all_stations' else freeze['routes'])
    gate['recipe'] = freeze['recipe']
    return gate


def run_b(args, a):
    roster = load_roster(args.data_root)
    gate = b_gate(a, read_json(args.out / FREEZE), roster)
    gate['a_report_sha256'] = sha256(args.out / A_REPORT)
    gate_path = args.out / 'b_gate.json'
    if gate_path.is_file() and read_json(gate_path) != gate:
        raise ValueError('The A-based B enrollment changed.')
    if not gate_path.is_file():
        write_json(gate_path, gate)
    print(f'[B 准入] {gate["candidate"]} A 差 {gate["a_delta"]:+.6f}，{gate["status"]}', flush=True)
    if gate['status'] != 'promoted':
        return
    train, days = date_roles(args.split, 'B')
    if (len(train), len(days)) != (628, 92):
        raise ValueError('Expected the frozen 628/92 B split.')
    b_args = SimpleNamespace(**{**vars(args), 'out': args.out / 'B'})
    b_args.out.mkdir(exist_ok=True)
    runtime = {'request_sha256': sha256(Path(__file__)), 'code_sha256': code_version(),
               'terrain_lookup_sha256': sha256(args.terrain_lookup), 'split_sha256': sha256(args.split)}
    digest = lock_predictions(b_args, 'T1', gate, runtime, train, days, roster, build_fit=make_t1)
    folder = b_args.out / 'T1'
    forecasts = before_scoring(folder, 'T1', keys_for(days, roster, 'T1'), digest)
    path = folder / 'score.json'
    if path.is_file():
        result = read_json(path)
        if result['prediction_lock_sha256'] != digest:
            raise ValueError('The B score uses a different prediction lock.')
    else:
        event(b_args.out, 'B', status='all_predictions_verified')
        result = paired_scores(b_args, 'T1', gate, days, roster, forecasts, digest)
        write_json(path, result)
    print(f'[B] 集成差 {result["ensemble"]["delta"][PRIMARY_SCORE]:+.6f}，'
          f'按规则采用 {result["acceptance"]["accepted"]}，报告 {path}', flush=True)


def execute(args):
    show_progress(args.out, '上风地形', 0, 4, '核对 DEM 和场站覆盖')
    args.terrain_lookup = lookup_for_run(args)
    if args.check_only:
        return
    settings = study_setup(args)
    settings_path = args.out / 'setup.json'
    if settings_path.is_file() and read_json(settings_path) != settings:
        raise ValueError('The registered experiment configuration changed.')
    if not settings_path.is_file():
        write_json(settings_path, settings)
    show_progress(args.out, '上风地形', 1, 4, '三个训练内块逐站配对')
    screen(args, settings)
    show_progress(args.out, '上风地形', 2, 4, 'A 集全站与逐站组合')
    confirm_a(args, settings)
    a = read_json(args.out / A_REPORT)
    result = report_a_diagnostics(args, a)
    print(f'[A] 全站地形差 {a["full_terrain"]["delta"][PRIMARY_SCORE]:+.6f}，诊断 {args.out / "A_terrain_diagnostics.json"}',
          flush=True)
    print(f'[A] 大爬坡 {result["ramps"]["large"]}', flush=True)
    show_progress(args.out, '上风地形', 3, 4, '按 A 结果决定 B')
    run_b(args, a)
    show_progress(args.out, '上风地形', 4, 4, '完成')
    print(f'报告目录 {args.out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/workspace/qhl/runs/t1_upwind_terrain_gpu_v1'))
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--baseline-run', type=Path, default=Path('/workspace/qhl/runs/t1_champion_st_tree_gpu_v1'))
    parser.add_argument('--local-wind-cache', type=Path, default=Path('/workspace/qhl/runs/companion_t1_local_wind'))
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--terrain-python', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    args.backend = 'xgboost'
    args.candidate_set = 'upwind-terrain'
    args.mlp_python = args.mlp_weather_python = None
    args.mlp_run = Path('/workspace/qhl/runs/warm_shared_power_v1')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args)
        except (Exception, KeyboardInterrupt) as error:
            event(args.out, 'terrain', status='failed', detail=str(error))
            raise
        event(args.out, 'terrain', status='complete')


if __name__ == '__main__':
    main()
