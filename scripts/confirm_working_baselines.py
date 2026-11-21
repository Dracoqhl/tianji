#!/usr/bin/env python3
"""Materialize the accepted T1/T2 recipes and replicate the T3 transfer mixture."""

import argparse
import fcntl
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

from advance_batch_b import before_scoring, keys_for, read_frozen_forecast
from batch_energy_short import fit_short_forecasts, roster_for, save_csv, short_source, t3_windows
from compare_t2_mlp import select_mlp_runtime
from compare_warm_t1 import PRIMARY_SCORE, b_statistics, compare_pair
from experiment_companion_t1 import assert_matched, event, identical_seed_groups, read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, date_roles, forecast_keys, sha256, write_json
from run_forecast_blends import FRACTIONS, T1_ARM, T2_ARM, blend, freeze_json, freeze_window
from run_same_issue_followup import call
from same_issue_weather import archive_stamp
from score_forecasts import MAX_PREDICTION, POINT_KEYS, score_track
from screen_companion_t1_stations import show_progress, station_scores
from screen_short_transfer import RIDGE, fit_models, pretrain_dates
from tree_model import check_backend
from warm_shared_power import labels, load_roster


SCRIPT = Path(__file__).resolve()
POLICY = SCRIPT.parents[1] / 'configs/current_working_baselines_v2.json'
WINDOW = 'fixed_50_10'
T3_ARM = 'tree_transfer_75_25'
TRANSFER = 'frozen_context_ridge'


def keep_sites(base, candidate, sites):
    if not base[POINT_KEYS].equals(candidate[POINT_KEYS]):
        raise ValueError('The saved T1 predictions have different ordered keys.')
    return base.assign(prediction=base.prediction.where(~base.station_id.isin(sites), candidate.prediction))


def prior_a(directory, keys, track, arm):
    manifest = directory / 'predictions_frozen.json'
    freeze = read_json(manifest)
    report = read_json(directory / 'score.json')
    digest = sha256(manifest)
    if report['prediction_lock_sha256'] != digest:
        raise ValueError(f'{track} A score is for an earlier lock.')
    forecasts = [read_frozen_forecast(directory / f'{name}.csv', freeze['hashes'][name], keys, track)
                 for name in ('baseline', arm)]
    return forecasts, digest, report


def average_seeds(keys, predictions, track):
    value = 'energy' if track == 'T2' else 'prediction'
    ensemble = {}
    for arm in ('baseline', 'candidate'):
        average = np.mean([pair[arm][value].to_numpy() for pair in predictions.values()], axis=0)
        ensemble[arm] = keys.assign(**{value: np.clip(average, 0, MAX_PREDICTION[track])})
    return ensemble


def freeze_warm(args, policy, track, roster):
    train, a_days = date_roles(args.split, 'A')
    b_train, b_days = date_roles(args.split, 'B')
    if len(a_days) != 42 or len(b_days) != 92 or len(b_train) != 628 or b_train != sorted(train + a_days):
        raise ValueError('Expected the registered warm A split and 628-day B fit.')
    source = args.blends / track / 'A'
    b_source = args.blends / 'B' / track
    directory = args.out / track
    a_keys = keys_for(a_days, roster, track)
    b_keys = keys_for(b_days, roster, track)
    arms, a_lock, a_report = prior_a(source, a_keys, track, T1_ARM if track == 'T1' else T2_ARM)
    b_lock = sha256(b_source / 'predictions_frozen.json')
    b_forecasts = before_scoring(b_source, track, b_keys, b_lock)
    b_report = read_json(b_source / 'score.json')
    if (b_report['prediction_lock_sha256'] != b_lock
            or b_report['gate']['a_score_sha256'] != sha256(source / 'score.json')):
        raise ValueError(f'{track} B results do not match the warm A report or locked forecasts.')
    plan = read_json(b_source / 'plan.json')
    assert_matched(plan['runtime'], {'split_sha256': sha256(args.split),
                                   'data_root': str(args.data_root.resolve())}, ('split_sha256', 'data_root'))
    if (plan['train_days'], plan['prediction_days'], plan['seeds']) != (628, 92, list(B_SEEDS)):
        raise ValueError(f'{track} B fit or seeds do not match the registered split.')

    rule = policy[track]
    if track == 'T1':
        if plan['runtime']['fractions'][T1_ARM] != FRACTIONS[T1_ARM] or rule['station_mix'] != T1_ARM:
            raise ValueError('The T1 forecast lock uses a different station mixture.')
        sites = rule['stations']
        decisions = b_report['station_decisions']
        if sorted(sites) != sorted(site for site, row in decisions.items() if row['accepted']):
            raise ValueError('The T1 recipe must include exactly the accepted stations.')
        for site in sites:
            row = decisions[site]
            gain = row['statistics']['ensemble_delta']
            if row['component'] != rule['station_mix'] or gain < 0.05 or not b_statistics(row['seed_deltas'], gain)['accepted']:
                raise ValueError(f'{site} failed the registered T1 acceptance rule.')
        arms[1] = keep_sites(*arms, sites)
        for forecasts in b_forecasts.values():
            forecasts['candidate'] = keep_sites(forecasts['baseline'], forecasts['candidate'], sites)
        b_delta = sum(decisions[site]['statistics']['ensemble_delta'] for site in sites) / len(roster)
        b_score = b_report['ensemble']['scores']['baseline'][PRIMARY_SCORE] + b_delta
        b_evidence = 'Reconstructed from the two accepted station scores and the 12-station equal aggregation.'
    else:
        if (b_report['gate']['candidate'] != rule['name'] or rule['name'] != T2_ARM
                or plan['runtime']['fractions'][T2_ARM] != FRACTIONS[T2_ARM]
                or rule['components'] != {'mlp_integral': 0.675, 'relative_mse': 0.225, 'pooled_t1_tree_integral': 0.10}):
            raise ValueError('The T2 B source does not contain the promoted three-way blend.')
        b_score = b_report['ensemble']['scores']['candidate'][PRIMARY_SCORE]
        b_delta = b_report['ensemble']['delta'][PRIMARY_SCORE]
        if not b_statistics([b_report['pairs'][str(seed)]['delta'][PRIMARY_SCORE] for seed in B_SEEDS], b_delta)['accepted']:
            raise ValueError('The T2 score no longer satisfies the paired acceptance check.')
        b_evidence = 'Exact candidate score from the matched frozen B forecasts.'

    provenance = {'source_a_lock': a_lock, 'source_b_lock': b_lock, 'recipe': rule}
    saved_a, a_output = freeze_window(directory / 'A', track, a_keys, dict(zip(('baseline', 'candidate'), arms)), provenance)
    if track == 'T1':
        scored_a = compare_pair(labels(args.data_root, a_days, roster), saved_a['baseline'],
                                saved_a['candidate'], roster, development=True)
        a_score = scored_a['scores']['candidate'][PRIMARY_SCORE]
        a_delta = scored_a['delta'][PRIMARY_SCORE]
    else:
        a_score = a_report['scores'][T2_ARM][PRIMARY_SCORE]
        a_delta = a_score - a_report['scores']['baseline'][PRIMARY_SCORE]
    locks = {}
    for seed, forecasts in b_forecasts.items():
        _, digest = freeze_window(directory / 'B' / str(seed), track, b_keys, forecasts, {**provenance, 'seed': seed})
        locks[str(seed)] = digest
    ensemble = average_seeds(b_keys, b_forecasts, track)
    _, ensemble_lock = freeze_window(directory / 'B' / 'ensemble', track, b_keys, ensemble, {'seed_locks': locks})
    manifest = {'recipe': rule, 'a_lock': a_output, 'a_score': a_score, 'a_delta': a_delta, 'b_seed_locks': locks,
                'b_ensemble_lock': ensemble_lock, 'b_ensemble': b_score, 'b_delta': b_delta, 'b_score_provenance': b_evidence}
    freeze_json(directory / 'baseline.json', manifest)
    print(f'[{track}] A {a_score:.6f}，B {b_score:.6f}，B 差 {b_delta:+.6f}', flush=True)
    return manifest


def target_window(args):
    _, train, days = next(window for window in t3_windows() if window[0] == WINDOW)
    regular, short = load_roster(args.data_root), roster_for(args.data_root, 'T3')
    allowed, _ = date_roles(args.split, 'A')
    source = pretrain_dates(allowed, days)
    if (len(train), len(days)) != (50, 10) or not source or max(train) >= min(days):
        raise ValueError('T3 requires the original 50/10 chronological window.')
    return train, days, regular, short, source


def transfer_worker(args):
    train, days, regular, short, source = target_window(args)
    args.out.mkdir(parents=True, exist_ok=True)
    forecasts, fit = fit_models(args, source, regular, short, train, days, args.out,
                                seed=args.transfer_worker, include_scratch=False)
    path = args.out / f'{TRANSFER}.csv'
    keys = forecast_keys(days, short)
    digest = save_csv(path, keys.assign(prediction=forecasts[TRANSFER].ravel()))
    read_frozen_forecast(path, digest, keys, 'T3')
    write_json(args.out / 'receipt.json', {'seed': args.transfer_worker, 'fit': fit, 'prediction_sha256': digest})


def t3_reference(args, train, days, regular, short, source, keys):
    prior = read_json(args.short_tree / 'results.json')
    window = prior['windows'][WINDOW]
    base = window['predictions_sha256']['short_only']
    old_transfer = args.transfer_run / WINDOW
    marker = old_transfer / 'predictions_frozen.json'
    transfer = read_json(marker)
    expected = {
        'seed': A_SEED, 'reference_sha256': base, 'device': args.device, 'epoch_budget': 16, 'ridge': RIDGE,
        'fit_dates': list(map(str, train)), 'score_dates': list(map(str, days)), 'source_dates': list(map(str, source)),
        'target_sites': short, 'source_sites': regular,
        'cache': archive_stamp(args.workdir, sorted(set(source + train + days))),
        'station_coordinates_sha256': sha256(args.data_root / 'static/power_stations.csv'),
    }
    assert_matched(transfer['plan'], expected, expected)
    tree_settings = {
        'data_root': str(args.data_root.resolve()), 'split_sha256': sha256(args.split), 'device': args.device,
        'threads': args.threads, 'seed': A_SEED, 'backend_version': check_backend('xgboost', args.device),
        'feature_spec_sha256': sha256(args.workdir / 'stats/feature_spec_train.json'),
        'companion_config_sha256': sha256(args.config or args.companion_code / 'configs/default.json'),
    }
    assert_matched(prior['settings'], tree_settings, tree_settings)
    reference = read_frozen_forecast(args.short_tree / WINDOW / 'short_only.csv', base, keys, 'T3')
    neural = read_frozen_forecast(old_transfer / f'{TRANSFER}.csv', transfer['forecasts'][TRANSFER], keys, 'T3')
    if window['window'] != expected['score_dates'] or window['train_days'] != len(train):
        raise ValueError('The original T3 short-only tree used a different window.')
    return reference, neural, {'tree_sha256': base, 'transfer_lock_sha256': sha256(marker),
                               'tree_settings': tree_settings, 'transfer_plan': expected}


def fit_t3_seed(args, target, seed, train, days, regular, short, keys):
    stage = Path(mkdtemp(dir=target.parent, prefix=f'.{seed}.attempt-'))
    roster, tree = short_source(args, [*regular, *short])
    *features, params = tree
    model_source = (*features, dict(params, random_state=seed))
    _, _, tree_receipt = fit_short_forecasts(args, stage, train, days, roster, short, model_source, modes=('short_only',))
    path = stage / 'short_only.csv'
    digest = tree_receipt['predictions_sha256']['short_only']
    reference = read_frozen_forecast(path, digest, keys, 'T3')
    interpreter = select_mlp_runtime(args, require_weather=False)[0]
    transfer = stage / 'transfer'
    call(interpreter, SCRIPT, '--transfer-worker', seed, '--out', transfer, '--data-root', args.data_root,
         '--device', args.device, '--workdir', args.workdir, '--split', args.split)
    receipt = read_json(transfer / 'receipt.json')
    if receipt['seed'] != seed:
        raise ValueError('The transfer receipt has the wrong seed.')
    neural = read_frozen_forecast(transfer / f'{TRANSFER}.csv', receipt['prediction_sha256'], keys, 'T3')
    return stage, reference, neural, {'tree': {'seed': seed, **tree_receipt}, 'transfer': receipt}


def score_t3(truth, roster, forecasts):
    scores = {arm: dict(score_track('T3', truth, frame, roster, allow_unscored_stations=True),
                        by_station=station_scores(truth, frame)) for arm, frame in forecasts.items()}
    return {'scores': scores, 'delta': scores['candidate'][PRIMARY_SCORE] - scores['baseline'][PRIMARY_SCORE],
            'station_deltas': {site: row['score'] - scores['baseline']['by_station'][site]['score']
                               for site, row in scores['candidate']['by_station'].items()}}


def confirm_t3(args, recipe):
    train, days, regular, short, source = target_window(args)
    keys = forecast_keys(days, short)
    reference, neural, provenance = t3_reference(args, train, days, regular, short, source, keys)
    directory = args.out / 'T3'
    directory.mkdir(exist_ok=True)
    plan = {'recipe': recipe, 'seeds': list(B_SEEDS), 'fraction': FRACTIONS[T3_ARM], 'source': provenance,
            'code_sha256': {name: sha256(SCRIPT.with_name(name)) for name in (
                SCRIPT.name, 'batch_energy_short.py', 'screen_short_transfer.py', 'warm_shared_power.py',
                'tree_model.py', 'same_issue_weather.py', 'score_forecasts.py')}}
    plan_digest = freeze_json(directory / 'plan.json', plan)
    locks, predictions = {}, {}
    for step, seed in enumerate(B_SEEDS, 1):
        target = directory / str(seed)
        if not target.is_dir():
            action = '复用已存预测' if seed == A_SEED else '开始拟合树和迁移模型'
            show_progress(directory, 'T3 配对确认', step - 1, len(B_SEEDS), f'seed={seed} {action}')
            if seed == A_SEED:
                stage = Path(mkdtemp(dir=directory, prefix=f'.{seed}.attempt-'))
                tree, transfer, fit = reference, neural, {'reuse': provenance}
            else:
                stage, tree, transfer, fit = fit_t3_seed(args, target, seed, train, days, regular, short, keys)
            frames = {'baseline': tree, 'candidate': blend(tree, transfer, FRACTIONS[T3_ARM], 'T3')}
            freeze_window(stage, 'T3', keys, frames, {'plan': plan_digest, 'seed': seed, 'fit': fit})
            stage.rename(target)
        marker = target / 'predictions_frozen.json'
        locked = read_json(marker)
        assert_matched(locked['sources'], {'plan': plan_digest, 'seed': seed}, ('plan', 'seed'))
        predictions[seed] = {arm: read_frozen_forecast(target / f'{arm}.csv', locked['hashes'][arm], keys, 'T3')
                             for arm in ('baseline', 'candidate')}
        locks[str(seed)] = sha256(marker)
        show_progress(directory, 'T3 配对确认', step, len(B_SEEDS), f'seed={seed} 预测核验')
    digest = freeze_json(directory / 'predictions_frozen.json', {'plan': plan_digest, 'seed_locks': locks})
    ensemble = average_seeds(keys, predictions, 'T3')
    ensemble, ensemble_lock = freeze_window(directory / 'ensemble', 'T3', keys, ensemble, {'seed_locks': locks})
    event(directory, 'T3', status='all_predictions_verified', seeds=list(B_SEEDS))

    truth = labels(args.data_root, days, short)
    scores = {str(seed): score_t3(truth, short, frames) for seed, frames in predictions.items()}
    old_score = read_json(args.blends / 'T3' / WINDOW / 'score.json')
    original = scores[str(A_SEED)]['scores']
    for arm, original_arm in (('baseline', 'baseline'), ('candidate', T3_ARM)):
        if abs(original[arm][PRIMARY_SCORE] - old_score['scores'][original_arm][PRIMARY_SCORE]) > 1e-7:
            raise ValueError('Rescoring seed 20261020 changed the previously reported T3 result.')
    overall = score_t3(truth, short, ensemble)
    acceptance = b_statistics([score['delta'] for score in scores.values()], overall['delta'])
    duplicated = {arm: identical_seed_groups({seed: pair[arm].prediction.to_numpy() for seed, pair in predictions.items()})
                  for arm in ('baseline', 'candidate')}
    report = {'window': WINDOW, 'evaluation_scope': 'matched-seed repeat of the same previously scored window',
              'prediction_lock_sha256': digest, 'ensemble_lock_sha256': ensemble_lock,
              'pairs': scores, 'ensemble': overall, 'acceptance': acceptance,
              'identical_seed_groups': duplicated, 'active_recipe': recipe['candidate'] if acceptance['accepted'] else recipe['baseline']}
    freeze_json(directory / 'score.json', report)
    print(f'[T3 {WINDOW}] 每种子分差 ' + ', '.join(f'{seed} {row["delta"]:+.6f}' for seed, row in scores.items()), flush=True)
    baseline, candidate = (overall['scores'][arm][PRIMARY_SCORE] for arm in ('baseline', 'candidate'))
    interval = acceptance['paired_t_95_interval']
    print(f'[T3 {WINDOW}] 集成 基线 {baseline:.6f}，候选 {candidate:.6f}，分差 {overall["delta"]:+.6f}，'
          f'配对 95% 区间 [{interval[0]:+.6f}, {interval[1]:+.6f}]', flush=True)
    print(f'[T3 {WINDOW}] 当前实验基线 {report["active_recipe"]}', flush=True)
    return {'name': report['active_recipe'], 'ensemble_delta': overall['delta'], 'acceptance': acceptance,
            'report_sha256': sha256(directory / 'score.json')}


def run(args):
    policy = read_json(args.policy)
    if policy['split_sha256'] != sha256(args.split):
        raise ValueError('The working-baseline contract uses a different warm split.')
    warm_tracks = () if args.t3_only else ('T1', 'T2')
    freeze_json(args.out / 'setup.json', {'policy_sha256': sha256(args.policy),
                                         'data_root': str(args.data_root.resolve()),
                                         'blend_source': str(args.blends.resolve()), 'code_sha256': sha256(SCRIPT),
                                         'tracks': [*warm_tracks, 'T3'],
                                         'device': args.device, 'threads': args.threads})
    steps = len(warm_tracks) + 1
    summary = {}
    if warm_tracks:
        roster = load_roster(args.data_root)
        for index, track in enumerate(warm_tracks):
            show_progress(args.out, '基线确认', index, steps, f'{track} 核验并固化预测')
            summary[track] = freeze_warm(args, policy, track, roster)
            write_json(args.out / 'summary.json', summary)
    show_progress(args.out, '基线确认', steps - 1, steps, 'T3 补跑种子并核验三组配对')
    summary['T3'] = confirm_t3(args, policy['T3'])
    write_json(args.out / 'summary.json', summary)
    show_progress(args.out, '基线确认', steps, steps, '全部完成')
    print(f'报告目录 {args.out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    runs = Path('/workspace/qhl/runs')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--t3-only', action='store_true', help='Run only the fixed-window T3 confirmation.')
    parser.add_argument('--blends', type=Path, default=runs / 'forecast_blends_gpu_v1')
    parser.add_argument('--short-tree', type=Path, default=runs / 'three_track_gpu_batch_v1' / 'T3')
    parser.add_argument('--transfer-run', type=Path, default=runs / 'same_issue_followup_v1' / 'T3')
    parser.add_argument('--mlp-run', type=Path, default=runs / 'warm_shared_power_v1')
    parser.add_argument('--mlp-python', type=Path)
    parser.add_argument('--policy', type=Path, default=POLICY)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--transfer-worker', type=int, choices=B_SEEDS)
    args = parser.parse_args()
    if args.out is None:
        if args.transfer_worker is not None:
            parser.error('--transfer-worker requires --out.')
        args.out = runs / ('t3_transfer_confirmation_gpu_v1' if args.t3_only else 'confirmed_baselines_gpu_v2')
    args.mlp_weather_python = None
    if args.transfer_worker is not None:
        transfer_worker(args)
        return
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
