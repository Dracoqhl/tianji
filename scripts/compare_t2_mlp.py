#!/usr/bin/env python3
"""Compare T2 direct and integrated forecasts with the original warm-season MLP."""

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
from tempfile import mkdtemp

import pandas as pd

from advance_batch_b import before_scoring, choose_gate, keys_for, lock_predictions, make_t2, paired_scores
from advance_batch_b import pooled_curve, read_frozen_forecast
from batch_energy_short import energy_arm_report, save_csv, weather_day_targets
from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import RECIPE_FIELDS, assert_matched, code_version, event, phase_versions, read_json
from independent_t2 import load_daily_targets
from run_companion_warm_t1 import B_SEEDS, SPLIT_PATH, cache_paths, cache_version, date_roles, sha256, write_json
from score_forecasts import DAY_KEYS, MAX_PREDICTION
from screen_companion_t1_stations import frozen_pooled_origin, show_progress
from warm_shared_power import load_roster


MLP = 'mlp_integral'
TREE_ARMS = ('relative_mse', 't1_integral')
BLENDS = {'mlp_relative_75_25': 'relative_mse', 'mlp_t1_75_25': 't1_integral'}
A_ARMS = (MLP, *TREE_ARMS, *BLENDS)
WORKER = Path(__file__).with_name('warm_shared_power.py')


def mix(mlp, tree):
    if not mlp[DAY_KEYS].equals(tree[DAY_KEYS]):
        raise ValueError('Blend components have different station-day grids.')
    return mlp.assign(energy=0.75 * mlp.energy + 0.25 * tree.energy)


def read_a_components(args, days, roster):
    keys = keys_for(days, roster, 'T2')
    tree_report = args.batch / 'T2' / 'results.json'
    tree = read_json(tree_report)
    a = tree['windows']['A']
    mlp_report = args.mlp_run / 'results.json'
    mlp = read_json(mlp_report)
    data_root = str(args.data_root.resolve())
    split = sha256(args.split)
    expected_dates = [day.isoformat() for day in days]
    if (a['window'] != expected_dates or a['train_days'] != 586
            or mlp['development_days'] != len(days) or mlp.get('train_days', 586) != 586
            or mlp['data_root'] != data_root or str(Path(tree['settings']['data_root']).resolve()) != data_root
            or mlp['split_sha256'] != split or tree['settings']['split_sha256'] != split
            or mlp.get('epochs', 16) != 16 or mlp.get('batch_station_days', 64) != 64):
        raise ValueError('The saved MLP and tree results must use the registered warm A comparison.')
    mlp_path = args.mlp_run / 'T2_development.csv'
    hashes = {MLP: sha256(mlp_path), **{name: a['predictions_sha256'][name] for name in TREE_ARMS}}
    single = {MLP: read_frozen_forecast(mlp_path, hashes[MLP], keys, 'T2')}
    for name in TREE_ARMS:
        single[name] = read_frozen_forecast(args.batch / 'T2' / 'A' / f'{name}.csv', hashes[name], keys, 'T2')
    origins = {'mlp_report_sha256': sha256(mlp_report), 'tree_report_sha256': sha256(tree_report),
               'forecast_sha256': hashes}
    return single, origins, {MLP: mlp['scores']['T2'], **{name: a['scores'][name] for name in TREE_ARMS}}


def run_a(args, days, roster):
    directory = args.out / 'A'
    directory.mkdir(parents=True, exist_ok=True)
    single, origins, old_scores = read_a_components(args, days, roster)
    forecasts = dict(single, **{arm: mix(single[MLP], single[source]) for arm, source in BLENDS.items()})
    paths = {arm: directory / f'{arm}.csv' for arm in forecasts}
    manifest_path = directory / 'predictions_frozen.json'
    if manifest_path.is_file():
        frozen = read_json(manifest_path)
        if frozen['origins'] != origins or set(frozen['hashes']) != set(A_ARMS):
            raise ValueError('The frozen A comparison differs from its source runs.')
    else:
        frozen = {'origins': origins, 'hashes': {arm: save_csv(paths[arm], frame) for arm, frame in forecasts.items()}}
        write_json(manifest_path, frozen)
    keys = keys_for(days, roster, 'T2')
    forecasts = {arm: read_frozen_forecast(paths[arm], frozen['hashes'][arm], keys, 'T2') for arm in A_ARMS}
    digest = sha256(manifest_path)
    report_path = directory / 'score.json'
    if report_path.is_file():
        report = read_json(report_path)
        if report['prediction_lock_sha256'] != digest:
            raise ValueError('A scores refer to a different prediction lock.')
        return report
    truth, eligible = load_daily_targets(weather_day_targets(args.data_root, days), roster)
    scores = energy_arm_report(truth, eligible, forecasts, roster)
    for name, expected in old_scores.items():
        if (abs(scores[name][PRIMARY_SCORE] - expected[PRIMARY_SCORE]) > 1e-7
                or scores[name]['scored_station_days'] != expected['scored_station_days']):
            raise ValueError(f'{name}: same-window rescore changed its reported A score or cohort.')
    gate = choose_gate(scores, MLP, A_ARMS)
    station_delta = {
        arm: {s: row['score'] - scores[MLP]['by_station'][s]['score'] for s, row in scores[arm]['by_station'].items()}
        for arm in A_ARMS if arm != MLP
    }
    blocks = {}
    for tag in dict.fromkeys(day.strftime('%Y-%m') for day in days):
        held = truth.date.str.startswith(tag)
        rows = eligible.date.str.startswith(tag)
        monthly = {arm: frame.loc[frame.date.str.startswith(tag)] for arm, frame in forecasts.items()}
        month_scores = energy_arm_report(truth.loc[held], eligible.loc[rows], monthly, roster)
        blocks[tag] = {arm: {'score': item[PRIMARY_SCORE], 'delta': item[PRIMARY_SCORE] - month_scores[MLP][PRIMARY_SCORE],
                            'station_days': item['scored_station_days']} for arm, item in month_scores.items()}
    result = {'prediction_lock_sha256': digest, 'window': [str(day) for day in days], 'scores': scores,
              'gate': gate, 'blocks': blocks, 'per_station_delta_vs_mlp': station_delta}
    write_json(report_path, result)
    return result


def select_mlp_python(args):
    candidates = [args.mlp_python] if args.mlp_python else [
        Path(sys.executable),
        Path('/workspace/qhl/.venv/bin/python'),
        Path('/workspace/qhl/venv/bin/python'),
        Path('/workspace/.venv/bin/python'),
        Path('/workspace/tianji-env-check/.venv/bin/python'),
        *sorted(Path('/workspace').glob('*/.venv/bin/python')),
        *sorted(Path('/workspace').glob('*/*/.venv/bin/python')),
        Path('/usr/local/bin/python3'),
        Path('/usr/bin/python3'),
    ]
    probe = ('import netCDF4, torch, pandas; '
             f'assert {args.device!r} == "cpu" or torch.cuda.is_available()')
    for python in dict.fromkeys(candidates):
        if python.is_file():
            try:
                result = subprocess.run([str(python), '-c', probe], capture_output=True, timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0:
                return python
    raise RuntimeError('No MLP Python with PyTorch, netCDF4 and the requested device. Set T2_MLP_PYTHON to the original MLP interpreter.')


def mlp_forecast(args, keys, seed, directory, interpreter):
    folder = directory / MLP
    if not folder.is_dir():
        pending = Path(mkdtemp(prefix='.mlp-attempt-', dir=directory))
        command = [str(interpreter), '-u', str(WORKER), '--predict-b', '--seed', str(seed),
                   '--data-root', str(args.data_root), '--split', str(args.split), '--out', str(pending),
                   '--cache-root', str(args.mlp_cache), '--workers', str(args.mlp_workers), '--device', args.device]
        with (pending / 'train.log').open('w') as logfile, subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        ) as process:
            for line in process.stdout:
                logfile.write(line)
                logfile.flush()
                print(line, end='', flush=True)
            if process.wait():
                raise RuntimeError(f'MLP B fit failed. See {pending / "train.log"}.')
        digest = sha256(pending / 'T2_sealed.csv')
        read_frozen_forecast(pending / 'T2_sealed.csv', digest, keys, 'T2')
        write_json(pending / 'frozen.json', {'prediction_sha256': digest,
                                             'receipt_sha256': sha256(pending / 'training_receipt.json')})
        pending.rename(folder)
    hashes = read_json(folder / 'frozen.json')
    receipt_path = folder / 'training_receipt.json'
    receipt = read_json(receipt_path)
    if (sha256(receipt_path) != hashes['receipt_sha256'] or receipt['seed'] != seed
            or receipt['train_days'] != 628 or receipt['development_days'] != 92
            or receipt['split_sha256'] != sha256(args.split)
            or Path(receipt['data_root']).resolve() != args.data_root.resolve() or receipt['device_requested'] != args.device
            or receipt['epochs'] != 16 or receipt['batch_station_days'] != 64):
        raise ValueError('Reused MLP forecast does not match the registered B fit.')
    return read_frozen_forecast(folder / 'T2_sealed.csv', hashes['prediction_sha256'], keys, 'T2'), receipt


def existing_relative_b(args, train, days, keys, runtime):
    directory = args.batch / 'B' / 'T2'
    manifest = directory / 'predictions_frozen.json'
    if not manifest.is_file():
        return None
    plan = read_json(directory / 'plan.json')
    if plan['gate']['candidate'] != 'relative_mse':
        return None
    assert_matched(plan['runtime'], runtime, ('data_root', 'split_sha256', 'device', 'threads'))
    if (plan['train_days'] != len(train) or plan['prediction_days'] != len(days) or plan['seeds'] != list(B_SEEDS)
            or plan['fit_cache_version'] != cache_version(cache_paths(args.workdir, train))
            or plan['forecast_cache_version'] != cache_version(cache_paths(args.workdir, days))
            or plan['gate']['a_report_sha256'] != sha256(args.batch / 'T2/results.json')):
        raise ValueError('Existing relative MSE forecasts use a different comparison source or date cache.')
    digest = sha256(manifest)
    saved = before_scoring(directory, 'T2', keys, digest)
    return saved, digest


def build_b_fit(args, train, days, roster, gate):
    keys = keys_for(days, roster, 'T2')
    interpreter = select_mlp_python(args)
    print(f'[T2 B] PyTorch 环境 {interpreter}', flush=True)
    component = BLENDS.get(gate['candidate'], gate['candidate'])
    reuse_relative = t2 = t1_recipe = None
    if component == 'relative_mse':
        reuse_relative = existing_relative_b(args, train, days, keys, args.runtime)
        if reuse_relative is None:
            t2 = make_t2(args, train, days, roster, {'baseline': component, 'candidate': component})
    else:
        t1_recipe = frozen_pooled_origin(args.hik_run)['a']['training']['baseline']
        curve_keys = keys_for(days, roster, 'T1')
        versions = phase_versions(args, train, days)

    def fit(seed, directory):
        mlp, mlp_receipt = mlp_forecast(args, keys, seed, directory, interpreter)
        if component == 't1_integral':
            info, curve = pooled_curve(args, seed, curve_keys, versions)
            assert_matched(t1_recipe, info, RECIPE_FIELDS)
            tree = curve.groupby(DAY_KEYS, as_index=False).prediction.sum()
            tree['energy'] = (tree.pop('prediction') / 4).clip(0, MAX_PREDICTION['T2'])
            tree = keys.merge(tree, on=DAY_KEYS, validate='one_to_one')
        elif reuse_relative is not None:
            saved, lock_digest = reuse_relative
            tree, info = saved[seed]['candidate'], {'original_prediction_lock_sha256': lock_digest}
        else:
            other, info = t2(seed, directory)
            tree = keys.merge(other['candidate'], on=DAY_KEYS, validate='one_to_one')
        forecast = mix(mlp, tree) if gate['candidate'] in BLENDS else tree
        return {'baseline': mlp, 'candidate': forecast}, {MLP: mlp_receipt, component: info}

    return fit


def execute(args):
    train, a_days = date_roles(args.split, 'A')
    b_train, b_days = date_roles(args.split, 'B')
    roster = load_roster(args.data_root)
    show_progress(args.out, 'T2 对照', 0, 3, '核验同窗预测')
    a = run_a(args, a_days, roster)
    for arm in A_ARMS:
        score = a['scores'][arm][PRIMARY_SCORE]
        delta = score - a['scores'][MLP][PRIMARY_SCORE]
        print(f'[T2 A] {arm} {score:.6f}，对 MLP {delta:+.6f}', flush=True)
    gate = {**a['gate'], 'a_score_sha256': sha256(args.out / 'A/score.json')}
    print(f'[T2 A] 报告 {args.out / "A/score.json"}', flush=True)
    show_progress(args.out, 'T2 对照', 1, 3, 'A 集五组配对完成')
    if gate['status'] != 'promoted':
        show_progress(args.out, 'T2 对照', 3, 3, 'A 集对 MLP 无正增益')
        return
    if b_train != sorted(train + a_days) or len(b_days) != 92:
        raise ValueError('B requires the frozen 628 fit days and 92 forecast days.')
    args.out = args.out / 'B'
    args.out.mkdir(exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    extra_code = (Path(__file__).name, 'advance_batch_b.py', 'batch_energy_short.py', 'screen_companion_t1_stations.py')
    args.runtime = {'data_root': str(args.data_root.resolve()), 'split_sha256': sha256(args.split),
                    'device': args.device, 'threads': args.threads, 'mlp_epochs': 16, 'mlp_batch': 64,
                    'blend_trees_fraction': 0.25, 'mlp_workers': args.mlp_workers,
                    'code_sha256': {**code_version(), **{name: sha256(script_dir / name) for name in extra_code}}}
    print(f'[T2 B] 冻结 {gate["candidate"]}，586 + 42 = 628 天重训，92 天计分', flush=True)
    manifest = lock_predictions(args, 'T2', gate, args.runtime, b_train, b_days, roster, build_fit=build_b_fit)
    keys = keys_for(b_days, roster, 'T2')
    directory = args.out / 'T2'
    predictions = before_scoring(directory, 'T2', keys, manifest)
    path = directory / 'score.json'
    event(args.out, 'B', status='all_predictions_verified')
    if path.is_file():
        result = read_json(path)
        if result['prediction_lock_sha256'] != manifest:
            raise ValueError('B score refers to a different forecast lock.')
    else:
        result = paired_scores(args, 'T2', gate, b_days, roster, predictions, manifest)
        result['evaluation_scope'] = 'follow-up on the previously used warm B dates'
        write_json(path, result)
    final = result['ensemble']
    delta = final['delta'][PRIMARY_SCORE]
    print(f'[T2 B] MLP {final["scores"]["baseline"][PRIMARY_SCORE]:.6f}，'
          f'{gate["candidate"]} {final["scores"]["candidate"][PRIMARY_SCORE]:.6f}，差 {delta:+.6f}', flush=True)
    for seed, pair in result['pairs'].items():
        print(f'[T2 B] seed={seed} 配对分差 {pair["delta"][PRIMARY_SCORE]:+.6f}', flush=True)
    print(f'[T2 B] 重复预测种子 {result["identical_seed_groups"]}，报告 {path}', flush=True)
    show_progress(args.out.parent, 'T2 对照', 3, 3, 'B 集配对完成')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/workspace/qhl/runs/t2_mlp_comparison_v1'))
    parser.add_argument('--batch', type=Path, default=Path('/workspace/qhl/runs/three_track_gpu_batch_v1'))
    parser.add_argument('--mlp-run', type=Path, default=Path('/workspace/qhl/runs/warm_shared_power_v1'))
    parser.add_argument('--hik-run', type=Path, default=Path('/workspace/qhl/runs/t1_champion_st_tree_gpu_v1'))
    parser.add_argument('--mlp-cache', type=Path, default=Path('/workspace/qhl/runs/warm_power_feature_cache'))
    parser.add_argument('--mlp-python', type=Path)
    parser.add_argument('--mlp-workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--local-wind-cache', type=Path, default=Path('/workspace/qhl/runs/companion_t1_local_wind'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.backend = 'xgboost'
    if not args.mlp_cache.is_dir() and (args.mlp_run / 'features').is_dir():
        args.mlp_cache = args.mlp_run / 'features'
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args)
        except (Exception, KeyboardInterrupt) as error:
            event(out, 'T2 MLP comparison', status='failed', detail=str(error))
            raise
        event(out, 'T2 MLP comparison', status='complete')


if __name__ == '__main__':
    main()
