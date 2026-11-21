#!/usr/bin/env python3
"""Screen T1 components per station on training-role time blocks, then score a locked A composite."""

import argparse
import csv
from datetime import date
import fcntl
from math import isclose
from pathlib import Path

from companion_t1_features import COMPONENTS
from companion_spatial_features import SPATIAL_COMPONENTS
from compare_warm_t1 import MINIMUM_GAIN, PRIMARY_SCORE, compare_pair, daily_scores
from experiment_companion_t1 import RECIPE_FIELDS, assert_matched, event, fit_or_reuse, load_run, phase_versions
from experiment_companion_t1 import read_json, setup
from run_companion_warm_t1 import A_SEED, SPLIT_PATH, date_roles, forecast_keys, sha256, training_block_dates, write_json
from run_companion_warm_t1 import block_early_stop_days
from score_forecasts import DAY_KEYS, POINT_KEYS, score_track
from warm_shared_power import labels, load_roster


BLOCKS = tuple((date.fromisoformat(first), date.fromisoformat(last)) for first, last in (
    ('2024-08-01', '2024-08-31'),
    ('2025-08-01', '2025-08-31'),
    ('2025-09-01', '2025-09-30'),
))
FREEZE = 'station_freeze.json'
SCREEN = 'station_screen.json'
A_REPORT = 'a_station_routes.json'
CPU_A_REFERENCE = 85.80038190973126
STATION_RECIPE_FIELDS = tuple(key for key in RECIPE_FIELDS if key != 'early_stopping_days_inside_train')
ST_TREE = 'st_tree_halves'
CANDIDATE_SETS = {'features': COMPONENTS, 'st-tree': (ST_TREE,), 'spatial': SPATIAL_COMPONENTS}


def fit_options(name):
    if name == 'baseline':
        return [], 'full'
    if name == ST_TREE:
        return [], 'halves'
    return [name], 'full'


def show_progress(out, stage, done, total, label):
    width = 10
    filled = width * done // total
    bar = '█' * filled + '░' * (width - filled)
    print(f'[{bar}] {stage} {done}/{total} {label}', flush=True)
    write_json(out / 'progress.json', {'stage': stage, 'completed': done, 'total': total, 'last_step': label})


def study_setup(args):
    result = dict(setup(args), station_screen_sha256=sha256(Path(__file__)),
                train_blocks=[[first.isoformat(), last.isoformat()] for first, last in BLOCKS],
                candidate_set=args.candidate_set, candidates=list(CANDIDATE_SETS[args.candidate_set]),
                station_gate={'minimum_gain_each_block': MINIMUM_GAIN, 'require_all_blocks': True})
    if getattr(args, 'baseline_run', None):
        origin = frozen_pooled_origin(args.baseline_run)
        result['baseline_origin'] = {key: origin[key] for key in ('freeze_sha256', 'a_sha256')}
    return result


def frozen_pooled_origin(path):
    freeze = read_json(path / FREEZE)
    a = read_json(path / A_REPORT)
    if (freeze['screen_sha256'] != sha256(path / SCREEN) or a['freeze_sha256'] != sha256(path / FREEZE)):
        raise ValueError('The frozen source of pooled predictions changed.')
    screen = read_json(path / SCREEN)
    return {'freeze_sha256': sha256(path / FREEZE), 'a_sha256': sha256(path / A_REPORT), 'a': a,
            'blocks': {item['date_range'][0]: item for item in screen['blocks']}}


def frozen_pooled_forecast(path, origin, name, train, held_out, keys, *, device):
    if name == 'A':
        expected = origin['a']['training']['baseline']
        directory, scope = path / 'a', None
    else:
        block = origin['blocks'][name]
        tags = [str(held_out[0]), str(held_out[-1])]
        if block['date_range'] != tags:
            raise ValueError('Forecast dates differ from the recorded baseline block.')
        expected = block['training']['baseline']
        directory = path / 'training_blocks' / name
        scope = {'train_block': tags, 'train_days': len(train), 'forecast_days': len(held_out),
                 'early_stopping_days_inside_train': block_early_stop_days(train)}
    receipt, forecast, digest = load_run(directory / 'baseline', keys, 'A', A_SEED, [], backend='xgboost',
                                         device=device, block_scope=scope)
    if receipt != expected:
        raise ValueError('The pooled forecast receipt differs from the frozen source.')
    return receipt, forecast, digest


def reference_fit(args, settings, name, directory, train, held_out, keys, versions, *, train_block=None):
    if not args.baseline_run:
        return fit_or_reuse(args, directory, keys, 'A', A_SEED, [], versions, train_block=train_block)
    origin = frozen_pooled_origin(args.baseline_run)
    if {key: origin[key] for key in ('freeze_sha256', 'a_sha256')} != settings['baseline_origin']:
        raise ValueError('The pooled baseline origin differs from the study setup.')
    receipt, forecast, digest = frozen_pooled_forecast(
        args.baseline_run, origin, name, train, held_out, keys, device=args.device,
    )
    assert_matched(settings, receipt, ('data_root', 'config_sha256', 'backend', 'device', 'backend_version', 'threads'))
    if receipt['companion_cache_metadata_sha256'] != versions:
        raise ValueError('The pooled baseline weather cache differs from the candidate cache.')
    print(f'Reusing the frozen pooled baseline for {name}', flush=True)
    return receipt, forecast, digest


def station_scores(truth, forecast):
    daily = daily_scores(truth, forecast)
    by_station = daily.groupby(level='station_id').agg(['mean', 'size'])
    return {row.Index: {'score': float(100 * row.mean), 'scored_days': int(row.size)}
            for row in by_station.itertuples()}


def block_comparisons(truth, forecasts, roster, candidates=COMPONENTS):
    diagnostics = {name: score_track('T1', truth, forecast, roster, allow_unscored_stations=True)
                   for name, forecast in forecasts.items()}
    scores = {name: station_scores(truth, forecast) for name, forecast in forecasts.items()}
    baseline = scores['baseline']
    rows = {}
    for name in candidates:
        candidate = scores[name]
        rows[name] = {}
        for station in roster:
            left, right = baseline.get(station), candidate.get(station)
            if (left is None) != (right is None) or (left and left['scored_days'] != right['scored_days']):
                raise ValueError(f'{name}/{station} changed the scoring population.')
            rows[name][station] = (
                {'baseline': left['score'], 'candidate': right['score'],
                 'delta': right['score'] - left['score'], 'scored_days': left['scored_days']}
                if left else {'baseline': None, 'candidate': None, 'delta': None, 'scored_days': 0}
            )
    return {'scores': diagnostics, 'station_pairs': rows}


def station_routes(roster, block_reports, candidates=COMPONENTS):
    """Use each time block once in selection, including negative and unavailable results."""
    routes, comparisons = {}, {}
    for station in roster:
        rows = {}
        for component in candidates:
            pair_rows = [report['station_pairs'][component][station] for report in block_reports]
            deltas = [row['delta'] for row in pair_rows]
            finite = [delta for delta in deltas if delta is not None]
            complete = len(finite) == len(BLOCKS)
            rows[component] = {
                'block_deltas': deltas, 'scored_days_by_block': [row['scored_days'] for row in pair_rows],
                'mean_delta': sum(finite) / len(finite) if finite else None,
                'eligible': complete and all(delta >= MINIMUM_GAIN or
                                             isclose(delta, MINIMUM_GAIN, rel_tol=0, abs_tol=1e-12)
                                             for delta in finite),
            }
        eligible = [component for component in candidates if rows[component]['eligible']]
        if eligible:
            routes[station] = max(eligible, key=lambda component: rows[component]['mean_delta'])
        comparisons[station] = rows
    return routes, comparisons


def verified_freeze(args, settings):
    path = args.out / FREEZE
    if not path.is_file():
        raise ValueError('Complete the training-block screen and freeze the station map before scoring A.')
    frozen = read_json(path)
    screen_path = args.out / SCREEN
    if frozen['settings'] != settings or frozen['screen_sha256'] != sha256(screen_path):
        raise ValueError('The station screen or run settings changed after freezing.')
    saved = read_json(screen_path)
    if any(frozen[name] != saved[name] for name in ('routes', 'recipe')):
        raise ValueError('The station routes or fit recipe changed after freezing.')
    return frozen


def write_station_table(path, comparisons, routes):
    block_names = [first.strftime('%Y-%m') for first, _ in BLOCKS]
    fields = ['station_id', 'component', 'mean_delta', 'passes', 'selected']
    fields += [f'{kind}_{block}' for block in block_names for kind in ('delta', 'scored_days')]
    pending = path.with_suffix('.tmp')
    with pending.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for station, options in comparisons.items():
            for component, outcome in options.items():
                row = {'station_id': station, 'component': component, 'mean_delta': outcome['mean_delta'],
                       'passes': outcome['eligible'], 'selected': routes.get(station) == component}
                for block, delta, days in zip(block_names, outcome['block_deltas'], outcome['scored_days_by_block']):
                    row.update({f'delta_{block}': delta, f'scored_days_{block}': days})
                writer.writerow(row)
    pending.replace(path)


def noon_change_error(truth, forecasts):
    """Measure a split-boundary jump only when both actual points are scored-valid."""
    at_noon = truth.loc[truth['point_index'].isin((48, 49)) & truth['power_limit'].eq(1), [*POINT_KEYS, 'power']]
    actual = at_noon.pivot(index=DAY_KEYS, columns='point_index', values='power').reindex(columns=[48, 49])
    change = (actual[49] - actual[48]).dropna()
    result = {}
    for name, forecast in forecasts.items():
        boundary = forecast.loc[forecast['point_index'].isin((48, 49))]
        modeled = boundary.pivot(index=DAY_KEYS, columns='point_index', values='prediction')
        error = ((modeled[49] - modeled[48]).reindex(change.index) - change).abs()
        per_station = error.groupby(level='station_id').agg(['mean', 'size'])
        result[name] = {'paired_station_days': len(error),
                        'change_mae_mw': float(error.mean()) if len(error) else None,
                        'per_station': {row.Index: {'change_mae_mw': float(row.mean), 'days': int(row.size)}
                                        for row in per_station.itertuples()}}
    return result


def screen(args, settings):
    if (args.out / FREEZE).exists():
        frozen = verified_freeze(args, settings)
        print(f'Reusing frozen station routes {frozen["routes"]}', flush=True)
        return
    roster = load_roster(args.data_root)
    reports = []
    recipe = None
    candidates = settings['candidates']
    completed = 0
    total = len(BLOCKS) * (1 + len(candidates)) + 1
    show_progress(args.out, '训练内配对', completed, total, '开始')
    for first, last in BLOCKS:
        train, held_out = training_block_dates(args.split, first, last)
        path = args.out / 'training_blocks' / first.isoformat()
        versions = phase_versions(args, train, held_out)
        keys = forecast_keys(held_out, roster)
        forecasts, receipts = {}, {}
        for name in ('baseline', *candidates):
            print(f'[训练内 {first}–{last}] {name}，{len(train)} 个拟合日', flush=True)
            components, partition = fit_options(name)
            if name == 'baseline':
                receipt, forecast, _ = reference_fit(args, settings, first.isoformat(), path / name,
                                                     train, held_out, keys, versions, train_block=(first, last))
            else:
                receipt, forecast, _ = fit_or_reuse(
                    args, path / name, keys, 'A', A_SEED, components, versions, train_block=(first, last),
                    point_partition=partition,
                )
            if receipts:
                assert_matched(receipts['baseline'], receipt)
            forecasts[name], receipts[name] = forecast, receipt
            completed += 1
            show_progress(args.out, '训练内配对', completed, total, f'{first} {name} 预测已保存')
        # Every model is fit and every prediction is written before scoring this training block.
        truth = labels(args.data_root, held_out, roster)
        baseline = receipts['baseline']
        if recipe is None:
            recipe = {key: baseline[key] for key in STATION_RECIPE_FIELDS}
        assert_matched(recipe, baseline, STATION_RECIPE_FIELDS)
        report = dict(block_comparisons(truth, forecasts, roster, candidates),
                      date_range=[first.isoformat(), last.isoformat()], training=receipts)
        if ST_TREE in candidates:
            report['point_48_to_49'] = noon_change_error(truth, forecasts)
        write_json(path / 'comparison.json', report)
        reports.append(report)
    routes, comparisons = station_routes(roster, reports, candidates)
    study = {'settings': settings, 'blocks': reports, 'stations': comparisons, 'routes': routes, 'recipe': recipe}
    write_json(args.out / SCREEN, study)
    write_station_table(args.out / 'station_summary.csv', comparisons, routes)
    write_json(args.out / FREEZE, {'settings': settings, 'routes': routes, 'recipe': recipe,
                                    'screen_sha256': sha256(args.out / SCREEN)})
    print(f'[训练内筛选] 冻结场站与组件 {routes}', flush=True)
    show_progress(args.out, '训练内配对', total, total, '逐站计分与映射冻结完成')


def combine_forecasts(baseline, candidates, routes):
    combined = baseline.copy()
    if any(not forecast[POINT_KEYS].equals(baseline[POINT_KEYS]) for forecast in candidates.values()):
        raise ValueError('Station routes require candidate keys aligned to the baseline.')
    for station, name in routes.items():
        mask = combined['station_id'].eq(station)
        if not mask.any():
            raise ValueError(f'Route references an unknown station {station}.')
        combined.loc[mask, 'prediction'] = candidates[name].loc[mask, 'prediction'].to_numpy()
    return combined


def confirm_a(args, settings):
    frozen = verified_freeze(args, settings)
    freeze_digest = sha256(args.out / FREEZE)
    report_path = args.out / A_REPORT
    if report_path.exists():
        report = read_json(report_path)
        if report['freeze_sha256'] != freeze_digest:
            raise ValueError('The existing A result used a different station freeze.')
        print(f'Reusing locked A result in {report_path}', flush=True)
        return
    routes = frozen['routes']
    train, held_out = date_roles(args.split, 'A')
    roster = load_roster(args.data_root)
    keys = forecast_keys(held_out, roster)
    versions = phase_versions(args, train, held_out)
    directory = args.out / 'a'
    used = [name for name in settings['candidates'] if name in routes.values()]
    total = 2 + len(used)
    show_progress(args.out, 'A 集复核', 0, total, '开始')
    print('[A] Fit the matched GPU baseline', flush=True)
    baseline_receipt, baseline, _ = reference_fit(args, settings, 'A', directory / 'baseline',
                                                  train, held_out, keys, versions)
    assert_matched(frozen['recipe'], baseline_receipt, STATION_RECIPE_FIELDS)
    receipts, candidates = {'baseline': baseline_receipt}, {}
    show_progress(args.out, 'A 集复核', 1, total, '基线预测已保存')
    for index, name in enumerate(used, 2):
        print(f'[A] Refit for frozen station route {name}', flush=True)
        components, partition = fit_options(name)
        receipt, forecast, _ = fit_or_reuse(args, directory / name, keys, 'A', A_SEED, components, versions,
                                            point_partition=partition)
        assert_matched(baseline_receipt, receipt)
        receipts[name], candidates[name] = receipt, forecast
        show_progress(args.out, 'A 集复核', index, total, f'{name} 预测已保存')
    composite = combine_forecasts(baseline, candidates, routes)
    forecast_path = directory / 'station_composite_T1_development.csv'
    pending = forecast_path.with_suffix('.tmp')
    composite.to_csv(pending, index=False)
    pending.replace(forecast_path)
    forecast_digest = sha256(forecast_path)

    truth = labels(args.data_root, held_out, roster)
    report = compare_pair(truth, baseline, composite, roster, development=True)
    cohort = report['scores']['baseline']
    if len(keys) != 48384 or cohort['scored_stations'] != 12 or cohort['scored_station_days'] != 399:
        raise ValueError('The locked A scoring cohort differs from the registered warm A cohort.')
    report.update(routes=routes, freeze_sha256=freeze_digest, prediction_sha256=forecast_digest, training=receipts,
                  historical_lightgbm_reference=CPU_A_REFERENCE)
    if ST_TREE in used:
        report['point_48_to_49'] = noon_change_error(truth, {'baseline': baseline, 'candidate': composite})
    write_json(report_path, report)
    print(f'[A] {args.backend}/{args.device} 基线 {cohort[PRIMARY_SCORE]:.6f}，'
          f'逐站组合 {report["scores"]["candidate"][PRIMARY_SCORE]:.6f}，'
          f'总分差 {report["delta"][PRIMARY_SCORE]:+.6f}', flush=True)
    for station, name in routes.items():
        print(f'[A] {station} {name} {report["per_station_readme"][station]["delta"]:+.6f}', flush=True)
    show_progress(args.out, 'A 集复核', total, total, '计分完成')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('screen', 'confirm-a'))
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--out', type=Path, default=Path('/workspace/qhl/runs/t1_station_screen_gpu_v1'))
    parser.add_argument('--candidate-set', choices=CANDIDATE_SETS, default='features')
    parser.add_argument('--baseline-run', type=Path, help='Reuse a verified prior pooled baseline from the same date blocks.')
    parser.add_argument('--local-wind-cache', type=Path, default=Path('/workspace/qhl/runs/companion_t1_local_wind'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error('Use 1 to 4 CPU threads.')
    args.backend = 'xgboost'
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        event(args.out, args.stage, status='started', backend=args.backend, device=args.device)
        try:
            settings = study_setup(args)
            registered = args.out / 'setup.json'
            if registered.is_file() and read_json(registered) != settings:
                raise ValueError('Experiment settings changed. Use a fresh run directory.')
            if not registered.is_file():
                write_json(registered, settings)
            {'screen': screen, 'confirm-a': confirm_a}[args.stage](args, settings)
        except (Exception, KeyboardInterrupt) as error:
            event(args.out, args.stage, status='failed', error=str(error))
            raise
        event(args.out, args.stage, status='complete')


if __name__ == '__main__':
    main()
