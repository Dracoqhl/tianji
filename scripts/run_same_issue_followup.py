#!/usr/bin/env python3
"""Run the three-track follow-up with the adopted T2 mixture and sealed T1 promotion."""

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from advance_batch_b import before_scoring, choose_gate, keys_for, lock_predictions, paired_scores, pooled_curve
from advance_batch_b import read_frozen_forecast
from compare_t2_mlp import select_mlp_runtime
from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import RECIPE_FIELDS, assert_matched, code_version, event, phase_versions, read_json
from run_companion_warm_t1 import SPLIT_PATH, date_roles, sha256, write_json
from screen_companion_t1_stations import frozen_pooled_origin, show_progress
from warm_shared_power import load_roster


SCRIPT_DIR = Path(__file__).resolve().parent
POLICY = SCRIPT_DIR.parent / 'configs/t2_current_baseline_v1.json'
CONTEXT_RUNNER = SCRIPT_DIR / 'screen_same_issue_context.py'


def call(interpreter, entry, *options):
    subprocess.run([str(interpreter), '-u', str(entry), *map(str, options)], check=True)


def check_t2(args):
    call(sys.executable, SCRIPT_DIR / 'diagnose_warm_power_errors.py',
         '--data-root', args.data_root, '--split', args.split, '--t2-policy', args.t2_policy,
         '--t2-adopted-run', args.t2_adopted_run,
         '--forecast', f'mlp={args.mlp_run / "T1_development.csv"}',
         '--forecast', f'pooled_xgboost={args.t1_run / "a/baseline/companion_T1_development.csv"}',
         '--out', args.out / 'T2/diagnostics.json')


def t1_b(args, interpreter):
    score_path = args.out / 'T1/A/score.json'
    a = read_json(score_path)
    gate = choose_gate(a['scores'], 'pooled_xgboost', ('pooled_xgboost', 'full_context'))
    gate['a_score_sha256'] = sha256(score_path)
    gate_path = args.out / 'T1/b_gate.json'
    if gate_path.is_file() and read_json(gate_path) != gate:
        raise ValueError('The T1 A result changed after the B decision.')
    if not gate_path.is_file():
        write_json(gate_path, gate)
    print(f'[T1 B] A 差 {gate["a_delta"]:+.6f}，{gate["status"]}', flush=True)
    if gate['status'] != 'promoted':
        return
    train, days = date_roles(args.split, 'B')
    if (len(train), len(days)) != (628, 92):
        raise ValueError('Expected 628/92 dates for the sealed comparison.')
    roster = load_roster(args.data_root)
    reference = frozen_pooled_origin(args.t1_run)['a']['training']['baseline']
    runtime = {'code_sha256': {**code_version(), **{name: sha256(SCRIPT_DIR / name) for name in (
        'run_same_issue_followup.py', 'advance_batch_b.py', 'screen_same_issue_context.py', 'same_issue_weather.py')}},
               'device': args.device, 'threads': args.threads, 'data_root': str(args.data_root.resolve()),
               'split_sha256': sha256(args.split), 'a_pooled_model': reference}
    b_args = SimpleNamespace(**{**vars(args), 'out': args.out / 'B'})
    b_args.out.mkdir(parents=True, exist_ok=True)

    def build_fit(options, train, days, roster, gate):
        keys = keys_for(days, roster, 'T1')
        versions = phase_versions(options, train, days)

        def fit(seed, directory):
            path = directory / 'context'
            call(interpreter, CONTEXT_RUNNER, '--predict-b', seed, '--data-root', options.data_root,
                 '--workdir', options.workdir, '--split', options.split, '--device', options.device, '--out', path)
            source = path / 'full_context'
            model = read_json(source / 'frozen.json')
            candidate = read_frozen_forecast(source / 'T1_sealed.csv', model['forecast_sha256'], keys, 'T1')
            receipt, baseline = pooled_curve(options, seed, keys, versions)
            assert_matched(reference, receipt, RECIPE_FIELDS)
            return {'baseline': baseline, 'candidate': candidate}, {'pooled_xgboost': receipt, 'full_context': model}

        return fit

    digest = lock_predictions(b_args, 'T1', gate, runtime, train, days, roster, build_fit=build_fit)
    directory = b_args.out / 'T1'
    verified = before_scoring(directory, 'T1', keys_for(days, roster, 'T1'), digest)
    path = directory / 'score.json'
    if path.is_file():
        result = read_json(path)
        if result['prediction_lock_sha256'] != digest:
            raise ValueError('T1 B report refers to different forecast files.')
    else:
        result = paired_scores(b_args, 'T1', gate, days, roster, verified, digest)
        write_json(path, result)
    print(f'[T1 B] 集成差 {result["ensemble"]["delta"][PRIMARY_SCORE]:+.6f}，'
          f'按登记规则采用 {result["acceptance"]["accepted"]}，报告 {path}', flush=True)


def execute(args):
    outcomes = {}

    def step(name, operation):
        try:
            operation()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            outcomes[name] = {'status': 'failed', 'error': str(error)}
            print(f'[{name}] {error}', file=sys.stderr, flush=True)
        else:
            outcomes[name] = {'status': 'completed'}
        write_json(args.out / 'status.json', outcomes)
        event(args.out, name, **outcomes[name])

    show_progress(args.out, '三赛道验证', 0, 4, 'T2 检查固定预测和现行基线')
    step('T2_A_diagnostics', lambda: check_t2(args))
    show_progress(args.out, '三赛道验证', 1, 4, '识别 PyTorch CUDA 环境')
    interpreter = select_mlp_runtime(args, require_weather=False)[0]
    common = ('--data-root', args.data_root, '--workdir', args.workdir, '--split', args.split, '--device', args.device)
    step('T1_screen_and_A', lambda: call(interpreter, CONTEXT_RUNNER, '--window', 'all', '--tree-run', args.t1_run,
                                        '--out', args.out / 'T1', *common))
    show_progress(args.out, '三赛道验证', 2, 4, 'T3 短历史迁移')
    step('T3_transfer', lambda: call(interpreter, SCRIPT_DIR / 'screen_short_transfer.py',
                                    '--tree-run', args.t3_run, '--out', args.out / 'T3', *common))
    show_progress(args.out, '三赛道验证', 3, 4, 'T1 按 A 结果自动决定 B')
    if outcomes['T1_screen_and_A']['status'] == 'completed':
        step('T1_B', lambda: t1_b(args, interpreter))
    else:
        outcomes['T1_B'] = {'status': 'awaiting_a'}
        write_json(args.out / 'status.json', outcomes)
    show_progress(args.out, '三赛道验证', 4, 4, '阶段状态已保存')
    print(f'报告目录 {args.out}', flush=True)
    return any(value['status'] == 'failed' for value in outcomes.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/workspace/qhl/runs/same_issue_followup_v1'))
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--mlp-python', type=Path)
    parser.add_argument('--local-wind-cache', type=Path, default=Path('/workspace/qhl/runs/companion_t1_local_wind'))
    parser.add_argument('--t1-run', type=Path, default=Path('/workspace/qhl/runs/t1_champion_st_tree_gpu_v1'))
    parser.add_argument('--mlp-run', type=Path, default=Path('/workspace/qhl/runs/warm_shared_power_v1'))
    parser.add_argument('--t3-run', type=Path, default=Path('/workspace/qhl/runs/three_track_gpu_batch_v1/T3'))
    parser.add_argument('--t2-policy', type=Path, default=POLICY)
    parser.add_argument('--t2-adopted-run', type=Path, default=Path('/workspace/qhl/runs/t2_mlp_comparison_v1'))
    args = parser.parse_args()
    args.backend = 'xgboost'
    args.mlp_weather_python = None
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(int(execute(args)))


if __name__ == '__main__':
    main()
