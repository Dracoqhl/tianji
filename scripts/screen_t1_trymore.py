#!/usr/bin/env python3
"""Assess a fixed trymore overlap correction using the frozen HIK forecasts."""

import argparse
import fcntl
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from compare_warm_t1 import PRIMARY_SCORE, compare_pair
from experiment_companion_t1 import assert_matched, event, fit_or_reuse, load_run, phase_versions, read_json
from run_companion_warm_t1 import A_SEED, SPLIT_PATH, block_early_stop_days, date_roles, forecast_keys
from run_companion_warm_t1 import sha256, training_block_dates, write_json
from score_forecasts import DAY_KEYS, MAX_PREDICTION, POINT_KEYS
from screen_companion_t1_stations import A_REPORT, BLOCKS, FREEZE, SCREEN, STATION_RECIPE_FIELDS, ST_TREE
from screen_companion_t1_stations import block_comparisons, combine_forecasts, noon_change_error
from screen_companion_t1_stations import show_progress, station_routes, study_setup, verified_freeze, write_station_table
from warm_shared_power import labels, load_roster


CANDIDATE = 'trymore_overlap'
SHRINKAGE = 0.5


def overlap_correction(baseline, short):
    """Shift only the tail, using each station-day's two prediction prefixes."""
    if not short[POINT_KEYS].equals(baseline[POINT_KEYS]):
        raise ValueError('Overlap inputs must have identical aligned forecast keys.')
    points = baseline['point_index']
    if (baseline[POINT_KEYS].isna().any().any() or baseline.duplicated(POINT_KEYS).any()
            or not (points.between(1, 96) & points.mod(1).eq(0)).all()
            or not baseline.groupby(DAY_KEYS)['point_index'].size().eq(96).all()):
        raise ValueError('Overlap correction requires a complete and unique 96-point grid per station-day.')
    prefix = points.le(48)
    base_values = baseline['prediction'].to_numpy(dtype=float)
    short_prefix = short.loc[prefix, 'prediction'].to_numpy(dtype=float)
    if not np.isfinite(base_values).all() or not np.isfinite(short_prefix).all():
        raise ValueError('Overlap inputs require finite prediction values.')
    offset = (baseline.loc[prefix, DAY_KEYS].assign(offset_mw=short_prefix - base_values[prefix])
              .groupby(DAY_KEYS, sort=False)['offset_mw'].mean())
    tail_keys = pd.MultiIndex.from_frame(baseline.loc[~prefix, DAY_KEYS])
    adjustment = offset.reindex(tail_keys).to_numpy(dtype=float)
    corrected = baseline.copy()
    corrected['prediction'] = base_values
    corrected.loc[~prefix, 'prediction'] = np.clip(
        base_values[~prefix] + SHRINKAGE * adjustment, 0.0, MAX_PREDICTION['T1'],
    )
    return corrected, offset


def transfer_diagnostic(truth, baseline, offset):
    """Compare the forecast offset with the realized late-half mean residual."""
    tail = truth.loc[truth['point_index'].gt(48) & truth['power_limit'].eq(1)
                     & np.isfinite(truth['power']), [*POINT_KEYS, 'power']]
    rows = tail.merge(baseline, on=POINT_KEYS, validate='one_to_one')
    rows['residual'] = rows['power'] - rows['prediction']
    residual = rows.groupby(DAY_KEYS)['residual'].mean().to_frame()
    pairs = residual.join(offset, how='inner')
    estimate = pairs['offset_mw'].to_numpy()
    actual = pairs['residual'].to_numpy()
    has_variation = len(pairs) > 1 and np.ptp(estimate) > 0 and np.ptp(actual) > 0
    return {'station_day_pairs': len(pairs),
            'mean_abs_overlap_offset_mw': float(np.abs(estimate).mean()) if len(pairs) else None,
            'offset_vs_late_residual_correlation': float(np.corrcoef(estimate, actual)[0, 1]) if has_variation else None}


def save_forecast(path, forecast):
    pending = path.with_suffix('.tmp')
    forecast.to_csv(pending, index=False)
    pending.replace(path)
    return sha256(path)


def hik_source(args):
    hik_setup = study_setup(args)
    source_freeze = args.hik_run / FREEZE
    source_screen = args.hik_run / SCREEN
    source_a = args.hik_run / A_REPORT
    verified_freeze(SimpleNamespace(out=args.hik_run), hik_setup)
    if not source_a.is_file():
        raise ValueError('Complete HIK A confirmation before running the next candidate.')
    if read_json(source_a)['freeze_sha256'] != sha256(source_freeze):
        raise ValueError('The HIK A report does not match its frozen station map.')
    return {'settings': hik_setup, 'source_dir': str(args.hik_run.resolve()),
            'source_screen_sha256': sha256(source_screen), 'source_a_sha256': sha256(source_a),
            'trymore_code_sha256': sha256(Path(__file__)), 'shrinkage': SHRINKAGE}


def read_hik_model(args, source, keys, name, expected_receipt, *, scope=None):
    partition = 'full' if name == 'baseline' else 'halves'
    receipt, forecast, digest = load_run(source / name, keys, 'A', A_SEED, [], backend=args.backend,
                                         device=args.device, block_scope=scope, point_partition=partition)
    if receipt != expected_receipt:
        raise ValueError(f'{source / name}: receipt does not match the HIK result used as source.')
    return receipt, forecast, digest


def screen(args, settings):
    if (args.out / FREEZE).exists():
        frozen = verified_freeze(args, settings)
        print(f'Reusing trymore freeze {frozen["routes"]}', flush=True)
        return
    roster = load_roster(args.data_root)
    origins = read_json(args.hik_run / SCREEN)['blocks']
    if len(origins) != len(BLOCKS):
        raise ValueError('Expected all three registered HIK fold reports.')
    reports = []
    total = len(BLOCKS) + 1
    show_progress(args.out, 'trymore 训练内计分', 0, total, '读取 HIK 预测')
    for index, ((first, last), origin) in enumerate(zip(BLOCKS, origins), 1):
        train_days, held_out = training_block_dates(args.split, first, last)
        tags = [first.isoformat(), last.isoformat()]
        if origin['date_range'] != tags:
            raise ValueError('HIK time blocks do not match the registered order.')
        keys = forecast_keys(held_out, roster)
        scope = {'train_block': tags, 'train_days': len(train_days), 'forecast_days': len(held_out),
                 'early_stopping_days_inside_train': block_early_stop_days(train_days)}
        source = args.hik_run / 'training_blocks' / tags[0]
        receipts = origin['training']
        control, baseline, control_digest = read_hik_model(args, source, keys, 'baseline', receipts['baseline'],
                                                           scope=scope)
        split, short, split_digest = read_hik_model(args, source, keys, ST_TREE, receipts[ST_TREE], scope=scope)
        assert_matched(control, split)
        corrected, offset = overlap_correction(baseline, short)
        output = args.out / 'training_blocks' / tags[0]
        output.mkdir(parents=True, exist_ok=True)
        forecast_digest = save_forecast(output / 'trymore_T1.csv', corrected)

        truth = labels(args.data_root, held_out, roster)
        forecasts = {'baseline': baseline, CANDIDATE: corrected}
        report = dict(block_comparisons(truth, forecasts, roster, (CANDIDATE,)),
                      date_range=tags, forecast_sha256=forecast_digest,
                      source_receipt_sha256={'baseline': control_digest, ST_TREE: split_digest},
                      point_48_to_49=noon_change_error(truth, forecasts),
                      overlap_transfer=transfer_diagnostic(truth, baseline, offset))
        reports.append(report)
        write_json(output / 'comparison.json', report)
        show_progress(args.out, 'trymore 训练内计分', index, total, f'{tags[0]} 分差已保存')
    routes, comparisons = station_routes(roster, reports, (CANDIDATE,))
    recipe = read_json(args.hik_run / FREEZE)['recipe']
    write_json(args.out / SCREEN, {'settings': settings, 'routes': routes, 'recipe': recipe,
                                    'blocks': reports, 'stations': comparisons})
    write_station_table(args.out / 'station_summary.csv', comparisons, routes)
    write_json(args.out / FREEZE, {'settings': settings, 'routes': routes, 'recipe': recipe,
                                    'screen_sha256': sha256(args.out / SCREEN)})
    show_progress(args.out, 'trymore 训练内计分', total, total, f'冻结场站 {list(routes)}')


def confirm_a(args, settings):
    frozen = verified_freeze(args, settings)
    digest = sha256(args.out / FREEZE)
    receipt_path = args.out / A_REPORT
    if receipt_path.exists():
        if read_json(receipt_path)['freeze_sha256'] != digest:
            raise ValueError('The existing A result used a different trymore freeze.')
        print(f'Reusing trymore A status in {receipt_path}', flush=True)
        return
    routes = frozen['routes']
    if not routes:
        base_score = read_json(args.hik_run / A_REPORT)['scores']['baseline'][PRIMARY_SCORE]
        write_json(receipt_path, {'status': 'no_station_passed', 'routes': {}, 'freeze_sha256': digest,
                                  'hik_baseline_a_score': base_score})
        show_progress(args.out, 'trymore A 集复核', 1, 1, '本轮维持全天基线')
        return

    train_days, held_out = date_roles(args.split, 'A')
    roster = load_roster(args.data_root)
    keys = forecast_keys(held_out, roster)
    previous = read_json(args.hik_run / A_REPORT)['training']
    source = args.hik_run / 'a'
    base_receipt, baseline, _ = read_hik_model(args, source, keys, 'baseline', previous['baseline'])
    assert_matched(frozen['recipe'], base_receipt, STATION_RECIPE_FIELDS)
    show_progress(args.out, 'trymore A 集复核', 1, 3, '基线预测已核对')
    if ST_TREE in previous:
        short_receipt, short, _ = read_hik_model(args, source, keys, ST_TREE, previous[ST_TREE])
        short_source = 'hik_a'
    else:
        versions = phase_versions(args, train_days, held_out)
        short_receipt, short, _ = fit_or_reuse(args, args.out / 'a' / ST_TREE, keys, 'A', A_SEED, [], versions,
                                              point_partition='halves')
        short_source = 'one_new_matched_refit'
    assert_matched(base_receipt, short_receipt)
    corrected, offset = overlap_correction(baseline, short)
    composite = combine_forecasts(baseline, {CANDIDATE: corrected}, routes)
    out = args.out / 'a'
    out.mkdir(parents=True, exist_ok=True)
    forecast_digest = save_forecast(out / 'trymore_station_composite_T1.csv', composite)
    show_progress(args.out, 'trymore A 集复核', 2, 3, '组合预测已保存')
    truth = labels(args.data_root, held_out, roster)
    report = compare_pair(truth, baseline, composite, roster, development=True)
    cohort = report['scores']['baseline']
    if len(keys) != 48384 or cohort['scored_stations'] != 12 or cohort['scored_station_days'] != 399:
        raise ValueError('The A scoring cohort differs from the warm validation contract.')
    report.update(status='scored', routes=routes, freeze_sha256=digest, forecast_sha256=forecast_digest,
                  short_source=short_source, training={'baseline': base_receipt, ST_TREE: short_receipt},
                  point_48_to_49=noon_change_error(truth, {'baseline': baseline, 'candidate': composite}),
                  overlap_transfer=transfer_diagnostic(truth, baseline, offset))
    write_json(receipt_path, report)
    print(f'[A] 全天基线 {cohort[PRIMARY_SCORE]:.6f}，'
          f'逐站组合 {report["scores"]["candidate"][PRIMARY_SCORE]:.6f}', flush=True)
    for station in routes:
        print(f'[A] {station} {CANDIDATE} {report["per_station_readme"][station]["delta"]:+.6f}', flush=True)
    show_progress(args.out, 'trymore A 集复核', 3, 3, f'整体分差 {report["delta"][PRIMARY_SCORE]:+.6f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('screen', 'confirm-a'))
    parser.add_argument('--hik-run', type=Path, default=Path('/workspace/qhl/runs/t1_champion_st_tree_gpu_v1'))
    parser.add_argument('--out', type=Path, default=Path('/workspace/qhl/runs/t1_champion_trymore_gpu_v1'))
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--local-wind-cache', type=Path, default=Path('/workspace/qhl/runs/companion_t1_local_wind'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4 or args.out.resolve() == args.hik_run.resolve():
        parser.error('Use up to 4 threads and a separate output directory for trymore.')
    args.backend, args.candidate_set = 'xgboost', 'st-tree'
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        event(args.out, args.stage, status='started')
        try:
            settings = hik_source(args)
            registered = args.out / 'setup.json'
            if registered.is_file() and read_json(registered) != settings:
                raise ValueError('The registered trymore inputs changed. Use a fresh output directory.')
            if not registered.is_file():
                write_json(registered, settings)
            {'screen': screen, 'confirm-a': confirm_a}[args.stage](args, settings)
        except (Exception, KeyboardInterrupt) as error:
            event(args.out, args.stage, status='failed', error=str(error))
            raise
        event(args.out, args.stage, status='complete')


if __name__ == '__main__':
    main()
