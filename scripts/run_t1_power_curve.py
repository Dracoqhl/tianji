#!/usr/bin/env python3
"""Compare a cross-fitted monotone power-curve feature with residual learning."""

import argparse
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd

from advance_batch_b import frozen_station_b
from companion_power_curve import CURVE_COMPONENTS, FEATURE, RESIDUAL
from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import code_version, event, load_run, read_json
from run_companion_warm_t1 import A_SEED, SPLIT_PATH, date_roles, forecast_keys, sha256, write_json
from run_t1_wind_profile import b_gate, print_b_decision
from score_forecasts import DAY_KEYS, MIN_VALID_POINTS, POINT_KEYS
from screen_companion_t1_stations import A_REPORT, FREEZE, confirm_a, frozen_pooled_forecast, frozen_pooled_origin
from screen_companion_t1_stations import screen, show_progress, study_setup
from warm_shared_power import labels, load_roster


def point_groups(truth, wind, priors, forecasts, limits):
    valid = truth.power_limit.eq(1) & np.isfinite(truth.power)
    counts = valid.groupby([truth[field] for field in DAY_KEYS], sort=False).transform('sum')
    scored = valid & counts.ge(MIN_VALID_POINTS)
    bins = np.digitize(wind, limits, right=True) if limits is not None else np.zeros(len(wind), dtype=int)
    results = {}
    for index, name in enumerate(('low', 'middle', 'high', 'extreme', 'missing')):
        group = wind.isna() if name == 'missing' else wind.notna() & (bins == index)
        take = scored & group
        row = {'points': int(take.sum()), 'station_days': len(truth.loc[take, DAY_KEYS].drop_duplicates())}
        if take.any():
            actual = truth.power.loc[take]
            predictions = {'curve_only': priors, **{name: frame.prediction for name, frame in forecasts.items()}}
            row['rmse_mw'] = {model: float(np.sqrt(((pred.loc[take] - actual)**2).mean()))
                              for model, pred in predictions.items()}
        results[name] = row
    return results


def a_diagnostics(args, a, roster):
    train, days = date_roles(args.split, 'A')
    keys = forecast_keys(days, roster)
    origin = frozen_pooled_origin(args.baseline_run)
    _, baseline, _ = frozen_pooled_forecast(args.baseline_run, origin, 'A', train, days, keys, device=args.device)
    forecasts = {'baseline': baseline}
    values = {}
    limits = {}
    for mode in CURVE_COMPONENTS:
        folder = args.out / 'a' / mode
        receipt, forecasts[mode], _ = load_run(folder, keys, 'A', A_SEED, [mode], backend=args.backend,
                                              device=args.device)
        if receipt != a['training'][mode]:
            raise ValueError(f'{mode}: the A prediction receipt changed.')
        meta = receipt['diagnostic']
        archive = folder / meta['file']
        if sha256(archive) != meta['sha256']:
            raise ValueError(f'{mode}: the A curve diagnostic hash changed.')
        array = np.load(archive, allow_pickle=False)
        if array.shape != (len(keys), 2):
            raise ValueError(f'{mode}: the curve diagnostic grid is incomplete.')
        original = pd.read_csv(folder / receipt['prediction_file'], usecols=POINT_KEYS,
                               dtype={'date': str, 'station_id': str})
        values[mode] = keys.merge(original.assign(wind=array[:, 0], prior_mw=array[:, 1]), on=POINT_KEYS,
                                  how='left', validate='one_to_one')
        limits[mode] = meta['training_wind_p25_p75_p95_mps']

    if not values[FEATURE].equals(values[RESIDUAL]) or limits[FEATURE] != limits[RESIDUAL]:
        raise ValueError('The power-curve alternatives do not use the same frozen weather prior.')
    uses = values[FEATURE]
    truth = labels(args.data_root, days, roster)
    diagnostic = {'a_report_sha256': sha256(args.out / A_REPORT), 'training_wind_quantiles_mps': limits[FEATURE],
                  'by_wind_group': point_groups(truth, uses.wind, uses.prior_mw, forecasts, limits[FEATURE])}
    write_json(args.out / 'A_power_curve_diagnostics.json', diagnostic)


def execute(args):
    settings = study_setup(args)
    setup = args.out / 'setup.json'
    if setup.is_file() and read_json(setup) != settings:
        raise ValueError('The registered power-curve setup changed.')
    if not setup.is_file():
        write_json(setup, settings)
    show_progress(args.out, '功率曲线', 0, 3, '三个训练内块逐站配对')
    screen(args, settings)
    show_progress(args.out, '功率曲线', 1, 3, 'A 集曲线输入与残差对照')
    confirm_a(args, settings)
    a = read_json(args.out / A_REPORT)
    roster = load_roster(args.data_root)
    a_diagnostics(args, a, roster)
    deltas = {name: row['delta'][PRIMARY_SCORE] for name, row in a['full_components'].items()}
    print(f'[A] 全站分差 {deltas}，残差模型相对曲线输入 {deltas[RESIDUAL] - deltas[FEATURE]:+.6f}', flush=True)
    show_progress(args.out, '功率曲线', 2, 3, '按 A 结果自动决定 B')
    gate = b_gate(a, read_json(args.out / FREEZE), roster, CURVE_COMPONENTS)
    gate['a_report_sha256'] = sha256(args.out / A_REPORT)
    runtime = {'launcher_sha256': sha256(Path(__file__)), 'code_sha256': code_version(),
               'split_sha256': sha256(args.split)}
    print_b_decision(args, frozen_station_b(args, gate, runtime), gate)
    show_progress(args.out, '功率曲线', 3, 3, '完成')
    print(f'报告目录 {args.out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/workspace/qhl/runs/t1_power_curve_gpu_v1'))
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--baseline-run', type=Path, default=Path('/workspace/qhl/runs/t1_champion_st_tree_gpu_v1'))
    parser.add_argument('--local-wind-cache', type=Path, default=Path('/workspace/qhl/runs/companion_t1_local_wind'))
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.backend = 'xgboost'
    args.candidate_set = 'power-curve'
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args)
        except (Exception, KeyboardInterrupt) as error:
            event(args.out, 'power-curve', status='failed', detail=str(error))
            raise
        event(args.out, 'power-curve', status='complete')


if __name__ == '__main__':
    main()
