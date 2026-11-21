#!/usr/bin/env python3
"""Run the matched wind-profile comparisons, with a separate per-station B decision."""

import argparse
import fcntl
from math import isclose
from pathlib import Path

import numpy as np
import pandas as pd

from advance_batch_b import choose_gate, frozen_station_b
from companion_wind_profile import PROFILE_COMPONENTS, SOURCES
from compare_warm_t1 import MINIMUM_GAIN, PRIMARY_SCORE, b_statistics
from experiment_companion_t1 import code_version, event, load_run, read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, date_roles, forecast_keys, sha256, write_json
from score_forecasts import DAY_KEYS, MIN_VALID_POINTS, POINT_KEYS
from screen_companion_t1_stations import A_REPORT, FREEZE, confirm_a, frozen_pooled_forecast, frozen_pooled_origin
from screen_companion_t1_stations import screen, show_progress, study_setup
from warm_shared_power import labels, load_roster


def meets_gain(value):
    return value >= MINIMUM_GAIN or isclose(value, MINIMUM_GAIN, rel_tol=0, abs_tol=1e-12)


def b_gate(report, frozen, roster):
    """A confirms training-screened sites; B assesses their gains individually."""
    baseline = report['scores']['baseline'][PRIMARY_SCORE]
    site_evidence = {station: report['per_station_readme'][station]['delta'] for station in frozen['routes']}
    confirmed = {site: arm for site, arm in frozen['routes'].items() if meets_gain(site_evidence[site])}
    if confirmed:
        delta = sum(site_evidence[site] for site in confirmed) / len(roster)
        return {'baseline': 'baseline', 'candidate': 'confirmed_stations', 'a_baseline': baseline,
                'a_candidate': baseline + delta, 'a_delta': delta, 'status': 'promoted',
                'routes': confirmed, 'recipe': frozen['recipe'], 'a_screened_station_deltas': site_evidence}

    scores = {'baseline': report['scores']['baseline'],
              **{name: report['full_components'][name]['scores']['candidate'] for name in PROFILE_COMPONENTS}}
    gate = choose_gate(scores, 'baseline', ('baseline', *PROFILE_COMPONENTS))
    return {**gate, 'routes': dict.fromkeys(roster, gate['candidate']), 'recipe': frozen['recipe'],
            'a_screened_station_deltas': site_evidence}


def b_station_decisions(result, sites):
    comparisons = {}
    for site, component in sites.items():
        ensemble = result['ensemble']['per_station_readme'][site]['delta']
        pairs = result['pairs']
        deltas = [(pairs[str(seed)] if str(seed) in pairs else pairs[seed])['per_station_readme'][site]['delta']
                  for seed in B_SEEDS]
        rule = b_statistics(deltas, ensemble)
        comparisons[site] = {'component': component, 'seed_deltas': deltas, 'statistics': rule,
                             'accepted': rule['accepted'] and meets_gain(ensemble)}
    return comparisons


def increments(frame, values):
    return values.groupby([frame[field] for field in DAY_KEYS], sort=False).diff()


def profile_diagnostics(truth, history, baseline, candidates, shear, threshold):
    valid = truth.power_limit.eq(1) & np.isfinite(truth.power)
    counts = valid.groupby([truth[field] for field in DAY_KEYS], sort=False).transform('sum')
    scored = valid & counts.ge(MIN_VALID_POINTS)
    actual = truth.power.where(scored)
    change = increments(truth, actual)
    history_power = history.power.where(history.power_limit.eq(1) & np.isfinite(history.power))
    ramp_limits = increments(history, history_power).abs().groupby(history.station_id).quantile(.90).dropna()
    large_ramp = change.abs().gt(0) & change.abs().ge(truth.station_id.map(ramp_limits))
    day = truth.point_index.between(25, 72)
    any_shear = shear.notna().any(axis=1)
    large_shear = pd.Series(False, index=shear.index)
    for source, limit in threshold.items():
        if limit is not None:
            large_shear |= shear[source].abs().ge(limit)
    groups = {'day': scored & day, 'night': scored & ~day, 'large_shear': scored & large_shear,
              'other_available_shear': scored & any_shear & ~large_shear,
              'missing_shear': scored & ~any_shear, 'large_ramp': large_ramp}
    outputs = {'training_abs_shear_p90': threshold, 'training_ramp_p90_mw': ramp_limits.to_dict(), 'groups': {}}
    forecasts = {'baseline': baseline, **candidates}
    for group, mask in groups.items():
        row = {'points': int(mask.sum()), 'station_days': len(truth.loc[mask, DAY_KEYS].drop_duplicates())}
        if mask.any():
            row['point_rmse_mw'] = {name: float(np.sqrt(((frame.prediction - actual).loc[mask]**2).mean()))
                                    for name, frame in forecasts.items()}
            if group == 'large_ramp':
                row['change_mae_mw'] = {name: float((increments(truth, frame.prediction) - change).loc[mask].abs().mean())
                                        for name, frame in forecasts.items()}
        outputs['groups'][group] = row
    return outputs


def diagnostics(args, report, roster):
    train, days = date_roles(args.split, 'A')
    keys = forecast_keys(days, roster)
    figure = {}
    candidates = {}
    threshold = None
    for name in PROFILE_COMPONENTS:
        directory = args.out / 'a' / name
        receipt, candidates[name], _ = load_run(directory, keys, 'A', A_SEED, [name], backend=args.backend, device=args.device)
        if receipt != report['training'][name]:
            raise ValueError(f'{name}: the A fit receipt changed.')
        meta = receipt['diagnostic']
        path = directory / meta['file']
        if sha256(path) != meta['sha256']:
            raise ValueError(f'{name}: the A shear feature hash changed.')
        values = np.load(path, allow_pickle=False)
        if values.shape != (len(keys), len(SOURCES)):
            raise ValueError(f'{name}: the shear diagnostic grid is incomplete.')
        raw = pd.read_csv(directory / receipt['prediction_file'], usecols=POINT_KEYS,
                          dtype={'station_id': str, 'date': str})
        for i, source in enumerate(SOURCES):
            raw[source] = values[:, i]
        figure[name] = keys.merge(raw, on=POINT_KEYS, how='left', validate='one_to_one')[list(SOURCES)]
        limits = meta['training_abs_shear_p90']
        if threshold is not None and limits != threshold:
            raise ValueError('The two alternatives use different training shear cutoffs.')
        threshold = limits

    shear = figure[PROFILE_COMPONENTS[0]]
    if not shear.equals(figure[PROFILE_COMPONENTS[1]]):
        raise ValueError('The two alternatives use different A shear values.')
    origin = frozen_pooled_origin(args.baseline_run)
    _, baseline, _ = frozen_pooled_forecast(args.baseline_run, origin, 'A', train, days, keys, device=args.device)
    hashes = {'a_report_sha256': sha256(args.out / A_REPORT)}
    path = args.out / 'A_wind_profile_diagnostics.json'
    if path.is_file():
        result = read_json(path)
        if result['hashes'] != hashes:
            raise ValueError('The stored A diagnostics refer to a different fit.')
        return result
    result = profile_diagnostics(labels(args.data_root, days, roster), labels(args.data_root, train, roster),
                                 baseline, candidates, shear, threshold)
    result['hashes'] = hashes
    write_json(path, result)
    return result


def execute(args):
    settings = study_setup(args)
    setup = args.out / 'setup.json'
    if setup.is_file() and read_json(setup) != settings:
        raise ValueError('The registered wind-profile setup changed.')
    if not setup.is_file():
        write_json(setup, settings)
    show_progress(args.out, '风廓线', 0, 3, '三个训练内块逐站配对')
    screen(args, settings)
    show_progress(args.out, '风廓线', 1, 3, 'A 集两组消融与逐站确认')
    confirm_a(args, settings)
    a = read_json(args.out / A_REPORT)
    roster = load_roster(args.data_root)
    report = diagnostics(args, a, roster)
    deltas = {name: row['delta'][PRIMARY_SCORE] for name, row in a['full_components'].items()}
    print(f'[A] 全站分差 {deltas}，转角额外增益 {deltas[PROFILE_COMPONENTS[1]]-deltas[PROFILE_COMPONENTS[0]]:+.6f}',
          flush=True)
    print(f'[A] 大爬坡 {report["groups"]["large_ramp"]}', flush=True)
    show_progress(args.out, '风廓线', 2, 3, '按 A 结果自动决定 B')
    gate = b_gate(a, read_json(args.out / FREEZE), roster)
    gate['a_report_sha256'] = sha256(args.out / A_REPORT)
    runtime = {'launcher_sha256': sha256(Path(__file__)), 'code_sha256': code_version(),
               'split_sha256': sha256(args.split)}
    result = frozen_station_b(args, gate, runtime)
    if result:
        if gate['candidate'] == 'confirmed_stations':
            decisions = b_station_decisions(result, gate['routes'])
            accepted = {site: row['component'] for site, row in decisions.items() if row['accepted']}
            write_json(args.out / 'B/T1/station_acceptance.json', {'decisions': decisions, 'adopted_routes': accepted,
                                                                 'prediction_lock_sha256': result['prediction_lock_sha256']})
            print(f'[B] 逐站采纳 {accepted}', flush=True)
        else:
            print(f'[B] 全站采纳 {result["acceptance"]["accepted"]}', flush=True)
        print(f'[B] 集成差 {result["ensemble"]["delta"][PRIMARY_SCORE]:+.6f}', flush=True)
    show_progress(args.out, '风廓线', 3, 3, '完成')
    print(f'报告目录 {args.out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/workspace/qhl/runs/t1_wind_profile_gpu_v1'))
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
    args.candidate_set = 'wind-profile'
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args)
        except (Exception, KeyboardInterrupt) as error:
            event(args.out, 'wind-profile', status='failed', detail=str(error))
            raise
        event(args.out, 'wind-profile', status='complete')


if __name__ == '__main__':
    main()
