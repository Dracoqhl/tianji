#!/usr/bin/env python3
"""Compare pre-registered small forecast blends on the three fixed track windows."""

import argparse
import fcntl
from pathlib import Path
from types import SimpleNamespace

from advance_batch_b import before_scoring, choose_gate, keys_for, lock_predictions, paired_scores
from advance_batch_b import pooled_curve, read_frozen_forecast
from batch_energy_short import energy_arm_report, roster_for, save_csv, t2_windows, t3_windows, weather_day_targets
from compare_t2_mlp import select_mlp_runtime
from compare_warm_t1 import PRIMARY_SCORE, compare_pair
from experiment_companion_t1 import RECIPE_FIELDS, assert_matched, code_version, event, phase_versions, read_json
from independent_t2 import load_daily_targets
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, cache_paths, cache_version, date_roles
from run_companion_warm_t1 import forecast_keys, sha256, write_json
from run_same_issue_followup import CONTEXT_RUNNER, POLICY, call
from score_forecasts import DAY_KEYS, MAX_PREDICTION, POINT_KEYS, score_track
from screen_companion_t1_stations import block_comparisons, frozen_pooled_forecast, frozen_pooled_origin
from screen_companion_t1_stations import show_progress, station_routes, station_scores
from screen_same_issue_context import CONTEXTS
from run_t1_wind_profile import b_station_decisions, meets_gain
from warm_shared_power import labels, load_roster


T1_ARM = 'tree_context_80_20'
T2_BASE = 'mlp_relative_75_25'
T2_ARM = 'current_t2_tree_integral_90_10'
T3_ARMS = {'tree_scratch_75_25': 'scratch_context', 'tree_transfer_75_25': 'frozen_context_ridge'}
FRACTIONS = {T1_ARM: 0.20, T2_ARM: 0.10, **dict.fromkeys(T3_ARMS, 0.25)}
SCRIPTS = Path(__file__).resolve().parent


def blend(baseline, other, fraction, track, sites=None):
    fields, value = (DAY_KEYS, 'energy') if track == 'T2' else (POINT_KEYS, 'prediction')
    if not baseline[fields].equals(other[fields]):
        raise ValueError(f'{track}: blend inputs have different ordered keys.')
    mixed = baseline[value] * (1 - fraction) + other[value] * fraction
    if sites is not None:
        mixed = baseline[value].where(~baseline.station_id.isin(sites), mixed)
    return baseline.assign(**{value: mixed.clip(0, MAX_PREDICTION[track])})


def freeze_json(path, record):
    if path.is_file():
        if read_json(path) != record:
            raise ValueError(f'{path}: registered inputs changed.')
    else:
        write_json(path, record)
    return sha256(path)


def freeze_window(directory, track, keys, arms, sources):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'predictions_frozen.json'
    if not marker.is_file():
        hashes = {name: save_csv(directory / f'{name}.csv', frame) for name, frame in arms.items()}
        freeze_json(marker, {'sources': sources, 'hashes': hashes})
    stored = read_json(marker)
    if stored['sources'] != sources or set(stored['hashes']) != set(arms):
        raise ValueError(f'{marker}: forecast sources or arms changed.')
    checked = {name: read_frozen_forecast(directory / f'{name}.csv', digest, keys, track)
               for name, digest in stored['hashes'].items()}
    return checked, sha256(marker)


def context_file(directory, options, train, days, seed, tag, keys, *, source_score=None):
    info = read_json(directory / 'frozen.json')
    expected = {'data_root': str(options.data_root.resolve()), 'device': options.device, 'seed': seed,
                'fit_dates': [str(day) for day in train], 'forecast_dates': [str(day) for day in days],
                'epochs': 16, 'batch_station_days': 64, 'dilations': list(CONTEXTS['full_context']),
                'station_coordinates_sha256': sha256(options.data_root / 'static/power_stations.csv'),
                'code_sha256': {name: sha256(SCRIPTS / name) for name in (
                    'screen_same_issue_context.py', 'warm_shared_power.py', 'same_issue_weather.py')}}
    assert_matched(expected, info['settings'], expected)
    if source_score is not None and source_score['prediction_sha256']['full_context'] != info['forecast_sha256']:
        raise ValueError('The saved context prediction differs from the original score receipt.')
    forecast = read_frozen_forecast(directory / f'T1_{tag}.csv', info['forecast_sha256'], keys, 'T1')
    return info, forecast


def t1_screen(args, roster, origin):
    inner, source_locks = [], {}
    for name, train, days in t2_windows(args.split):
        target = args.out / 'T1' / name
        keys = forecast_keys(days, roster)
        directory = args.context_run / 'T1' / name
        previous = read_json(directory / 'score.json')
        tree, baseline, receipt = frozen_pooled_forecast(args.tree_run, origin, name, train, days,
                                                        keys, device=args.device)
        assert_matched(tree, args.runtime, ('data_root', 'split_sha256', 'device', 'threads'))
        model, neural = context_file(directory / 'full_context', args, train, days, A_SEED, 'development', keys,
                                     source_score=previous)
        if previous['prediction_sha256']['pooled_xgboost'] != tree['prediction_sha256']:
            raise ValueError('The context study used a different pooled tree.')
        full = blend(baseline, neural, FRACTIONS[T1_ARM], 'T1')
        arms = {'baseline': baseline, T1_ARM: full}
        sources = {'tree_receipt': receipt, 'context': model['forecast_sha256'], 'fraction': FRACTIONS[T1_ARM]}
        if name == 'A':
            routes, comparisons = station_routes(roster, inner, (T1_ARM,))
            freeze_digest = freeze_json(args.out / 'T1' / 'station_freeze.json',
                                        {'routes': routes, 'comparisons': comparisons, 'locks': source_locks})
            sources['station_freeze_sha256'] = freeze_digest
            arms['training_routes'] = blend(baseline, neural, FRACTIONS[T1_ARM], 'T1', routes)
        arms, digest = freeze_window(target, 'T1', keys, arms, sources)
        truth = labels(args.data_root, days, roster)
        if name != 'A':
            score = block_comparisons(truth, arms, roster, (T1_ARM,))
            inner.append(score)
            source_locks[name] = digest
            write_json(target / 'score.json', {**score, 'prediction_lock_sha256': digest})
            print(f'[T1 训练块] {name} 逐站计分完成', flush=True)
            continue

        full_report = compare_pair(truth, arms['baseline'], arms[T1_ARM], roster, development=True)
        route_report = compare_pair(truth, arms['baseline'], arms['training_routes'], roster, development=True)
        confirmed = {site: T1_ARM for site in routes
                     if meets_gain(full_report['per_station_readme'][site]['delta'])}
        report = {'prediction_lock_sha256': digest, 'full': full_report, 'training_routes': route_report,
                  'confirmed_sites': confirmed}
        print(f'[T1 A] 全站分差 {full_report["delta"][PRIMARY_SCORE]:+.6f}，'
              f'训练期筛选场站 {list(routes)}，A 确认场站 {list(confirmed)}', flush=True)
        score_path = target / 'score.json'
        write_json(score_path, report)
        gate = {'baseline': 'baseline', 'a_score_sha256': sha256(score_path)}
        if confirmed:
            gain = sum(full_report['per_station_readme'][site]['delta'] for site in confirmed) / len(roster)
            baseline_score = full_report['scores']['baseline'][PRIMARY_SCORE]
            gate.update(candidate='confirmed_stations', routes=confirmed, a_delta=gain,
                        a_baseline=baseline_score, a_candidate=baseline_score + gain, status='promoted')
        else:
            gate.update(choose_gate({'baseline': full_report['scores']['baseline'],
                                     T1_ARM: full_report['scores']['candidate']}, 'baseline', ('baseline', T1_ARM)))
        return gate
    raise ValueError('The warm A phase is missing.')


def t2_screen(args, days, roster):
    source = args.adopted_t2 / 'A'
    marker = source / 'predictions_frozen.json'
    locked = read_json(marker)
    report = read_json(source / 'score.json')
    policy = read_json(args.t2_policy)
    if (policy['name'] != T2_BASE or policy['components'] != {'mlp_integral': 0.75, 'relative_mse': 0.25}
            or policy['split_sha256'] != args.runtime['split_sha256']
            or report['prediction_lock_sha256'] != sha256(marker)
            or report['window'] != [str(day) for day in days]):
        raise ValueError('The adopted T2 A report does not match the registered baseline.')
    keys = keys_for(days, roster, 'T2')
    hashes = {arm: locked['hashes'][arm] for arm in (T2_BASE, 't1_integral')}
    base, tree = [read_frozen_forecast(source / f'{name}.csv', digest, keys, 'T2')
                  for name, digest in hashes.items()]
    t1_a = args.out / 'T1' / 'A'
    tree_digest = read_json(t1_a / 'predictions_frozen.json')['hashes']['baseline']
    curve = read_frozen_forecast(t1_a / 'baseline.csv', tree_digest, forecast_keys(days, roster), 'T1')
    if (tree.energy - integral(curve, keys).energy).abs().max() > 1e-9:
        raise ValueError('The T2 tree integral used a different pooled T1 forecast.')
    directory = args.out / 'T2' / 'A'
    arms, digest = freeze_window(directory, 'T2', keys,
                                 {'baseline': base, T2_ARM: blend(base, tree, FRACTIONS[T2_ARM], 'T2')},
                                 {'source_lock_sha256': sha256(marker), 'fraction': FRACTIONS[T2_ARM]})
    truth, eligible = load_daily_targets(weather_day_targets(args.data_root, days), roster)
    scores = energy_arm_report(truth, eligible, arms, roster)
    expected = report['scores'][T2_BASE]
    if (abs(scores['baseline'][PRIMARY_SCORE] - expected[PRIMARY_SCORE]) > 1e-7
            or scores['baseline']['scored_station_days'] != expected['scored_station_days']):
        raise ValueError('The T2 baseline score or scoring population changed.')
    months = {}
    for month in dict.fromkeys(day.strftime('%Y-%m') for day in days):
        month_arms = {name: frame.loc[frame.date.str.startswith(month)] for name, frame in arms.items()}
        item = energy_arm_report(truth.loc[truth.date.str.startswith(month)],
                                 eligible.loc[eligible.date.str.startswith(month)], month_arms, roster)
        months[month] = choose_gate(item, 'baseline', ('baseline', T2_ARM))['a_delta']
    base_sites = scores['baseline']['by_station']
    site_delta = {site: row['score'] - base_sites[site]['score']
                  for site, row in scores[T2_ARM]['by_station'].items()}
    score_path = directory / 'score.json'
    write_json(score_path, {'prediction_lock_sha256': digest, 'scores': scores, 'monthly_deltas': months,
                            'station_deltas': site_delta})
    return {**choose_gate(scores, 'baseline', ('baseline', T2_ARM)), 'a_score_sha256': sha256(score_path)}


def t3_screen(args):
    roster = roster_for(args.data_root, 'T3')
    previous = read_json(args.short_tree / 'results.json')
    assert_matched(previous['settings'], args.runtime, ('data_root', 'split_sha256'))
    results = {}
    for name, train, days in t3_windows():
        directory = args.context_run / 'T3' / name
        keys = forecast_keys(days, roster)
        marker = directory / 'predictions_frozen.json'
        locked = read_json(marker)
        plan = locked['plan']
        expected = {'target_sites': roster, 'fit_dates': [str(day) for day in train],
                    'score_dates': [str(day) for day in days], 'seed': A_SEED,
                    'station_coordinates_sha256': sha256(args.data_root / 'static/power_stations.csv')}
        assert_matched(plan, expected, expected)
        reference = previous['windows'][name]
        tree_hash = reference['predictions_sha256']['short_only']
        if plan['reference_sha256'] != tree_hash or reference['window'] != expected['score_dates']:
            raise ValueError(f'{name}: the T3 source windows or baseline changed.')
        base = read_frozen_forecast(args.short_tree / name / 'short_only.csv', tree_hash, keys, 'T3')
        arms = {'baseline': base}
        for candidate, source in T3_ARMS.items():
            other = read_frozen_forecast(directory / f'{source}.csv', locked['forecasts'][source], keys, 'T3')
            arms[candidate] = blend(base, other, FRACTIONS[candidate], 'T3')
        arms, digest = freeze_window(args.out / 'T3' / name, 'T3', keys, arms,
                                     {'transfer_lock_sha256': sha256(marker), 'tree_sha256': tree_hash,
                                      'fractions': {arm: FRACTIONS[arm] for arm in T3_ARMS}})
        truth = labels(args.data_root, days, roster)
        scores = {arm: dict(score_track('T3', truth, frame, roster, allow_unscored_stations=True),
                            by_station=station_scores(truth, frame)) for arm, frame in arms.items()}
        base_score = scores['baseline']
        if abs(base_score[PRIMARY_SCORE] - reference['scores']['short_only'][PRIMARY_SCORE]) > 1e-7:
            raise ValueError(f'{name}: the short-only score changed.')
        deltas = {arm: scores[arm][PRIMARY_SCORE] - base_score[PRIMARY_SCORE] for arm in T3_ARMS}
        site_deltas = {arm: {site: row['score'] - base_score['by_station'][site]['score']
                             for site, row in scores[arm]['by_station'].items()} for arm in T3_ARMS}
        result = {'prediction_lock_sha256': digest, 'scores': scores, 'deltas': deltas,
                  'station_deltas': site_deltas}
        write_json(args.out / 'T3' / name / 'score.json', result)
        results[name] = deltas
        print(f'[T3 {name}] 基线 {base_score[PRIMARY_SCORE]:.6f}，分差 {deltas}', flush=True)
    write_json(args.out / 'T3' / 'summary.json', results)


def adopted_b(args, train, days, roster):
    source = args.adopted_t2 / 'B' / 'T2'
    plan = read_json(source / 'plan.json')
    assert_matched(plan['runtime'], args.runtime, ('data_root', 'split_sha256', 'device', 'threads'))
    if (plan['gate']['candidate'] != T2_BASE
            or plan['gate']['a_score_sha256'] != sha256(args.adopted_t2 / 'A' / 'score.json')
            or plan['runtime']['blend_trees_fraction'] != 0.25 or plan['seeds'] != list(B_SEEDS)
            or (plan['train_days'], plan['prediction_days']) != (len(train), len(days))
            or plan['fit_cache_version'] != cache_version(cache_paths(args.workdir, train))
            or plan['forecast_cache_version'] != cache_version(cache_paths(args.workdir, days))):
        raise ValueError('The saved adopted T2 B components do not match the 628-day fit.')
    digest = sha256(source / 'predictions_frozen.json')
    return before_scoring(source, 'T2', keys_for(days, roster, 'T2'), digest), digest


def integral(curve, keys):
    daily = (curve.groupby(DAY_KEYS, sort=False).prediction.sum() / 4).rename('energy').reset_index()
    energy = keys.merge(daily, on=DAY_KEYS, how='left', validate='one_to_one')
    return energy.assign(energy=energy.energy.clip(0, MAX_PREDICTION['T2']))


def run_b(args, gates, roster, tree_reference):
    promoted = {track: gate for track, gate in gates.items() if gate['status'] == 'promoted'}
    if not promoted:
        print('[B] 两条赛道均未获准，跳过', flush=True)
        return
    train, days = date_roles(args.split, 'B')
    a_train, a_days = date_roles(args.split, 'A')
    if len(days) != 92 or len(train) != 628 or train != sorted(a_train + a_days):
        raise ValueError('B must use exactly training + A, with 92 held-out dates.')
    options = SimpleNamespace(**{**vars(args), 'out': args.out / 'B'})
    options.out.mkdir(exist_ok=True)
    curve_keys = forecast_keys(days, roster)
    versions = phase_versions(args, train, days)
    if 'T1' in promoted:
        interpreter = select_mlp_runtime(args, require_weather=False)[0]
    if 'T2' in promoted:
        current, adopted_lock = adopted_b(args, train, days, roster)
        energy_keys = keys_for(days, roster, 'T2')

    def build_t1(_options, _train, _days, _roster, gate):
        def fit(seed, directory):
            info, baseline = pooled_curve(options, seed, curve_keys, versions)
            assert_matched(tree_reference, info, RECIPE_FIELDS)
            work = directory / 'context'
            call(interpreter, CONTEXT_RUNNER, '--predict-b', seed, '--data-root', args.data_root,
                 '--workdir', args.workdir, '--split', args.split, '--device', args.device, '--out', work)
            model, alternative = context_file(work / 'full_context', args, train, days, seed, 'sealed', curve_keys)
            candidate = blend(baseline, alternative, FRACTIONS[T1_ARM], 'T1', gate.get('routes'))
            return {'baseline': baseline, 'candidate': candidate}, {'pooled': info, 'full_context': model}

        return fit

    def build_t2(_options, _train, _days, _roster, _gate):
        def fit(seed, _directory):
            info, curve = pooled_curve(options, seed, curve_keys, versions)
            assert_matched(tree_reference, info, RECIPE_FIELDS)
            baseline = current[seed]['candidate']
            candidate = blend(baseline, integral(curve, energy_keys), FRACTIONS[T2_ARM], 'T2')
            return {'baseline': baseline, 'candidate': candidate}, {'pooled': info, 'adopted_lock': adopted_lock}

        return fit

    builders = {'T1': build_t1, 'T2': build_t2}
    locks = {}
    for track, gate in promoted.items():
        runtime = {**args.runtime, 't1_tree_reference': tree_reference}
        if track == 'T2':
            runtime['adopted_b_lock_sha256'] = adopted_lock
        locks[track] = lock_predictions(options, track, gate, runtime, train, days, roster, build_fit=builders[track])

    # Complete every promoted prediction lock before the first B label read.
    ready = {track: before_scoring(options.out / track, track, keys_for(days, roster, track), digest)
             for track, digest in locks.items()}
    event(options.out, 'B', status='all_predictions_verified', tracks=list(ready))
    for track, predictions in ready.items():
        path = options.out / track / 'score.json'
        if path.is_file():
            result = read_json(path)
            if result['prediction_lock_sha256'] != locks[track]:
                raise ValueError(f'{track}: existing B score has a different prediction lock.')
        else:
            gate = promoted[track]
            result = paired_scores(options, track, gate, days, roster, predictions, locks[track])
            result['evaluation_scope'] = 'follow-up on previously used warm B dates'
            if track == 'T1' and 'routes' in gate:
                result['station_decisions'] = b_station_decisions(result, gate['routes'])
            write_json(path, result)
        values = result['ensemble']
        print(f'[B] {track} 三种子等权分差 {values["delta"][PRIMARY_SCORE]:+.6f}，报告 {path}', flush=True)


def run(args):
    paths = ('context_run', 'tree_run', 'short_tree', 'adopted_t2', 't2_policy')
    args.runtime = {'data_root': str(args.data_root.resolve()), 'split_sha256': sha256(args.split),
                    'device': args.device, 'threads': args.threads, 'fractions': FRACTIONS,
                    't2_policy_sha256': sha256(args.t2_policy),
                    'sources': {name: str(getattr(args, name).resolve()) for name in paths},
                    'code_sha256': {**code_version(), **{
                        name: sha256(SCRIPTS / name) for name in (
                            Path(__file__).name, 'advance_batch_b.py', 'batch_energy_short.py',
                            'screen_same_issue_context.py', 'run_t1_wind_profile.py')}}}
    freeze_json(args.out / 'setup.json', args.runtime)
    train, days = date_roles(args.split, 'A')
    if (len(train), len(days)) != (586, 42):
        raise ValueError('Expected the registered 586/42 A split.')
    roster = load_roster(args.data_root)
    origin = frozen_pooled_origin(args.tree_run)
    show_progress(args.out, '三赛道融合验证', 0, 4, 'T1 核对三个训练块与暖季 A')
    g1 = t1_screen(args, roster, origin)
    show_progress(args.out, '三赛道融合验证', 1, 4, 'T2 对照现行融合')
    g2 = t2_screen(args, days, roster)
    gates = {'T1': g1, 'T2': g2}
    freeze_json(args.out / 'b_gate.json', gates)
    for track, gate in gates.items():
        print(f'[{track} A] {gate["candidate"]} 分差 {gate["a_delta"]:+.6f}，{gate["status"]}', flush=True)
    show_progress(args.out, '三赛道融合验证', 2, 4, 'T3 两个内部窗口和固定 50/10')
    t3_screen(args)
    show_progress(args.out, '三赛道融合验证', 3, 4, '按冻结的 A 结果自动决定 B')
    run_b(args, gates, roster, origin['a']['training']['baseline'])
    show_progress(args.out, '三赛道融合验证', 4, 4, '完成')
    print(f'报告目录 {args.out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    runs = Path('/workspace/qhl/runs')
    parser.add_argument('--out', type=Path, default=runs / 'forecast_blends_gpu_v1')
    parser.add_argument('--context-run', type=Path, default=runs / 'same_issue_followup_v1')
    parser.add_argument('--tree-run', type=Path, default=runs / 't1_champion_st_tree_gpu_v1')
    parser.add_argument('--short-tree', type=Path, default=runs / 'three_track_gpu_batch_v1' / 'T3')
    parser.add_argument('--adopted-t2', type=Path, default=runs / 't2_mlp_comparison_v1')
    parser.add_argument('--t2-policy', type=Path, default=POLICY)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--local-wind-cache', type=Path, default=runs / 'companion_t1_local_wind')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--mlp-python', type=Path)
    parser.add_argument('--mlp-run', type=Path, default=runs / 'warm_shared_power_v1')
    args = parser.parse_args()
    args.backend = 'xgboost'
    args.mlp_weather_python = None
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except (Exception, KeyboardInterrupt) as error:
            event(args.out, 'batch', status='failed', detail=str(error))
            raise
        event(args.out, 'batch', status='complete')


if __name__ == '__main__':
    main()
