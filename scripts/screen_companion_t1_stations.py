#!/usr/bin/env python3
"""Screen T1 components per station on training-role time blocks, then score a locked A composite."""

import argparse
import csv
from datetime import date
import fcntl
from math import isclose
from pathlib import Path

from companion_t1_features import COMPONENTS
from compare_warm_t1 import MINIMUM_GAIN, PRIMARY_SCORE, compare_pair, daily_scores
from experiment_companion_t1 import RECIPE_FIELDS, assert_matched, event, fit_or_reuse, phase_versions, read_json, setup
from run_companion_warm_t1 import A_SEED, SPLIT_PATH, date_roles, forecast_keys, sha256, training_block_dates, write_json
from score_forecasts import POINT_KEYS, score_track
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


def study_setup(args):
    return dict(setup(args), station_screen_sha256=sha256(Path(__file__)),
                train_blocks=[[first.isoformat(), last.isoformat()] for first, last in BLOCKS],
                station_gate={'minimum_gain_each_block': MINIMUM_GAIN, 'require_all_blocks': True})


def station_scores(truth, forecast):
    daily = daily_scores(truth, forecast)
    by_station = daily.groupby(level='station_id').agg(['mean', 'size'])
    return {row.Index: {'score': float(100 * row.mean), 'scored_days': int(row.size)}
            for row in by_station.itertuples()}


def block_comparisons(truth, forecasts, roster):
    diagnostics = {name: score_track('T1', truth, forecast, roster, allow_unscored_stations=True)
                   for name, forecast in forecasts.items()}
    scores = {name: station_scores(truth, forecast) for name, forecast in forecasts.items()}
    baseline = scores['baseline']
    rows = {}
    for name in COMPONENTS:
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


def station_routes(roster, block_reports):
    """Use each time block once in selection, including negative and unavailable results."""
    routes, comparisons = {}, {}
    for station in roster:
        rows = {}
        for component in COMPONENTS:
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
        eligible = [component for component in COMPONENTS if rows[component]['eligible']]
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


def screen(args, settings):
    if (args.out / FREEZE).exists():
        frozen = verified_freeze(args, settings)
        print(f'Reusing frozen station routes {frozen["routes"]}', flush=True)
        return
    roster = load_roster(args.data_root)
    reports = []
    recipe = None
    for first, last in BLOCKS:
        train, held_out = training_block_dates(args.split, first, last)
        path = args.out / 'training_blocks' / first.isoformat()
        versions = phase_versions(args, train, held_out)
        keys = forecast_keys(held_out, roster)
        forecasts, receipts = {}, {}
        for name, components in [('baseline', []), *((name, [name]) for name in COMPONENTS)]:
            print(f'[训练内 {first}–{last}] {name}，{len(train)} 个拟合日', flush=True)
            receipt, forecast, _ = fit_or_reuse(
                args, path / name, keys, 'A', A_SEED, components, versions, train_block=(first, last),
            )
            if receipts:
                assert_matched(receipts['baseline'], receipt)
            forecasts[name], receipts[name] = forecast, receipt
        # Every model is fit and every prediction is written before scoring this training block.
        truth = labels(args.data_root, held_out, roster)
        baseline = receipts['baseline']
        if recipe is None:
            recipe = {key: baseline[key] for key in STATION_RECIPE_FIELDS}
        assert_matched(recipe, baseline, STATION_RECIPE_FIELDS)
        report = dict(block_comparisons(truth, forecasts, roster),
                      date_range=[first.isoformat(), last.isoformat()], training=receipts)
        write_json(path / 'comparison.json', report)
        reports.append(report)
    routes, comparisons = station_routes(roster, reports)
    study = {'settings': settings, 'blocks': reports, 'stations': comparisons, 'routes': routes, 'recipe': recipe}
    write_json(args.out / SCREEN, study)
    write_station_table(args.out / 'station_summary.csv', comparisons, routes)
    write_json(args.out / FREEZE, {'settings': settings, 'routes': routes, 'recipe': recipe,
                                    'screen_sha256': sha256(args.out / SCREEN)})
    print(f'[训练内筛选] 冻结场站与组件 {routes}', flush=True)


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
    print('[A] Fit the matched GPU baseline', flush=True)
    baseline_receipt, baseline, _ = fit_or_reuse(args, directory / 'baseline', keys, 'A', A_SEED, [], versions)
    assert_matched(frozen['recipe'], baseline_receipt, STATION_RECIPE_FIELDS)
    receipts, candidates = {'baseline': baseline_receipt}, {}
    for name in COMPONENTS:
        if name in routes.values():
            print(f'[A] Refit for frozen station route {name}', flush=True)
            receipt, forecast, _ = fit_or_reuse(args, directory / name, keys, 'A', A_SEED, [name], versions)
            assert_matched(baseline_receipt, receipt)
            receipts[name], candidates[name] = receipt, forecast
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
    write_json(report_path, report)
    print(f'[A] {args.backend}/{args.device} 基线 {cohort[PRIMARY_SCORE]:.6f}，'
          f'逐站组合 {report["scores"]["candidate"][PRIMARY_SCORE]:.6f}，'
          f'总分差 {report["delta"][PRIMARY_SCORE]:+.6f}', flush=True)
    for station, name in routes.items():
        print(f'[A] {station} {name} {report["per_station_readme"][station]["delta"]:+.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('screen', 'confirm-a'))
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--out', type=Path, default=Path('/workspace/qhl/runs/t1_station_screen_gpu_v1'))
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
