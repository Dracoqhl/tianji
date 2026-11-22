#!/usr/bin/env python3
"""Compare T1 power-only and two auxiliary wind masks on the chronological split."""

import argparse
import fcntl
from functools import partial
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from advance_batch_b import read_frozen_forecast
from batch_energy_short import save_csv
from compare_t2_mlp import select_mlp_runtime
from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import event, read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, chronological_dates, forecast_keys, sha256, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from run_same_issue_followup import call
from run_t1_t3_batch import paired_report, timing_setup, timed, worker_options
from run_wind_attribution import wind_labels
from same_issue_weather import archived_issue_weather
from score_forecasts import MAX_PREDICTION, POINT_KEYS
from screen_companion_t1_stations import block_comparisons, show_progress
from screen_same_issue_context import CONTEXTS, make_context_network
from warm_shared_power import fit_power, labels, load_roster, normalize


SCRIPT = Path(__file__).resolve()
RUNS = Path('/workspace/qhl/runs')
REFERENCE = 'three_source_conditioned_blend'
MODES = ('wind_at_valid_power', 'wind_at_available_points')
WIND_WEIGHT = 0.10
EPOCHS = 16
BATCH = 64


def settings(args, train, days, roster, seed):
    return {**timing_setup(args, train, days, roster), 'seed': seed, 'epochs': EPOCHS,
            'network': list(CONTEXTS['full_context']), 'batch': BATCH}


def scaled_wind(truth, valid, roster):
    wind = truth.wind.to_numpy(dtype=float).reshape(-1, len(roster), 96)
    usable_days = valid.any(axis=1).reshape(-1, len(roster))
    seen = np.isfinite(wind) & (wind >= 0) & usable_days[..., None]
    counts = seen.sum(axis=(0, 2))
    if not counts.any():
        raise ValueError('The training fold has no auxiliary wind observations.')
    center = np.where(seen, wind, 0).sum(axis=(0, 2)) / counts.clip(min=1)
    variance = np.where(seen, (wind - center[None, :, None])**2, 0).sum(axis=(0, 2)) / counts.clip(min=1)
    scale = np.sqrt(variance).clip(min=1e-6)
    normalized = (wind - center[None, :, None]) / scale[None, :, None]
    return np.where(seen, normalized, np.nan).reshape(valid.shape), center, scale, counts


def inference(model, weather, device, scopes):
    import torch

    model.eval()
    power, wind = [], []
    with torch.inference_mode():
        for part in np.array_split(weather, max(1, (len(weather) + BATCH - 1) // BATCH)):
            encoded = model.encode(torch.as_tensor(part, device=device))
            power.append(torch.sigmoid(model.output(encoded).squeeze(-1)).cpu().numpy())
            wind.append(model.wind_head(encoded).squeeze(-1).cpu().numpy())
    center, scale = scopes
    power = np.concatenate(power).astype(float) * MAX_PREDICTION['T1']
    wind = np.concatenate(wind).reshape(-1, len(center), 96)
    wind = wind * scale[None, :, None] + center[None, :, None]
    return np.clip(power.ravel(), 0, MAX_PREDICTION['T1']), np.maximum(wind.ravel(), 0)


def fit_networks(args):
    import torch

    train, days = chronological_dates(args.split, args.phase)
    roster = load_roster(args.data_root)
    keys = forecast_keys(days, roster)
    plan = args.out / 'settings.json'
    freeze_json(plan, settings(args, train, days, roster, args.seed))
    pending = [name for name in args.modes if not (args.out / name / 'predictions_frozen.json').is_file()]
    if not pending:
        return
    raw = timed(args.out, 'weather_read',
                lambda: archived_issue_weather(args.workdir, args.data_root, [*train, *days], roster, print)[0])
    cut = len(train) * len(roster)
    x, test, _, _ = timed(args.out, 'normalization', lambda: normalize(raw[:cut], raw[cut:]))
    del raw
    truth = wind_labels(args.data_root, train, roster)
    power = truth.power.to_numpy(dtype=float).reshape(cut, 96)
    valid = truth.power_limit.eq(1).to_numpy().reshape(power.shape) & np.isfinite(power)
    wind, center, scale, counts = scaled_wind(truth, valid, roster)
    maker = partial(make_context_network, dilations=CONTEXTS['full_context'])
    for name in pending:
        print(f'[T1 {args.phase}] {args.seed} 开始 {name}', flush=True)
        target = np.where(valid, wind, np.nan) if name == MODES[0] else wind
        model, predict, history, device_name, _ = timed(
            args.out, name + '_fit',
            lambda: fit_power(x, power, valid, test, EPOCHS, BATCH, args.device, print,
                              seed=args.seed, network_factory=maker, wind_target=target, wind_weight=WIND_WEIGHT))
        values, speed = timed(args.out, name + '_predict', lambda: inference(model, test, args.device, (center, scale)))
        if not np.isfinite(speed).all():
            raise ValueError('Wind head returned nonfinite forecasts.')
        attempt = Path(mkdtemp(dir=args.out, prefix=f'.{name}-attempt-'))
        torch.save(model.state_dict(), attempt / 'network.pt')
        wind_hash = save_csv(attempt / 'wind.csv', keys.assign(wind=speed))
        freeze_window(attempt, 'T1', keys, {'network': keys.assign(prediction=values)},
                      {'settings_sha256': sha256(plan), 'wind_sha256': wind_hash, 'mode': name,
                       'wind_weight': WIND_WEIGHT, 'device': device_name, 'power_trajectory_rmse_mw': history,
                       'wind_target_rows': int(np.isfinite(target).sum()),
                       'wind_scaling': {site: {'center': c, 'scale': s, 'rows': int(n)}
                                        for site, c, s, n in zip(roster, center, scale, counts)}})
        attempt.rename(args.out / name)
        del model, predict


def fetch_power(directory, label, keys):
    marker = directory / 'predictions_frozen.json'
    record = read_json(marker)
    return read_frozen_forecast(directory / f'{label}.csv', record['hashes'][label], keys, 'T1')


def prior_baseline(args, phase, keys, seed):
    root = args.wind_run / 'T1'
    a = read_json(root / 'A_score.json')['gate']
    if a['candidate'] != REFERENCE or not a['promoted']:
        raise ValueError('The prior T1 A gate is different from the wind-fusion baseline.')
    if phase == 'A':
        target = root / 'A'
        marker = sha256(target / 'predictions_frozen.json')
        if a['forecast_lock'] != marker:
            raise ValueError('The prior wind A predictions changed.')
        return fetch_power(target, REFERENCE, keys), marker
    target = root / 'B'
    lock = target / 'predictions_frozen.json'
    score = read_json(target / 'b_score.json')
    seed_dir = target / str(seed)
    receipt = seed_dir / 'predictions_frozen.json'
    if (score['prediction_lock_sha256'] != sha256(lock) or not score['acceptance']['accepted']
            or read_json(lock)[str(seed)] != sha256(receipt)
            or read_json(receipt)['sources']['gate'] != a):
        raise ValueError('The prior wind B forecasts no longer match their accepted lock.')
    return fetch_power(seed_dir, 'candidate', keys), sha256(receipt)


def control_forecast(args, comparable, keys):
    source = args.power_run / 'T1' / 'A' / str(A_SEED) / 'networks'
    directory = source / 'daily_rmse'
    marker = directory / 'predictions_frozen.json'
    if read_json(source / 'settings.json') != comparable:
        raise ValueError('The frozen pure-power control has different inputs or training dates.')
    receipt = read_json(marker)
    if receipt['sources']['settings_sha256'] != sha256(source / 'settings.json'):
        raise ValueError('The frozen pure-power forecast settings changed.')
    return fetch_power(directory, 'network', keys), sha256(marker)


def fit_candidates(args, phase, seed, modes, keys, interpreter):
    output = args.out / phase / str(seed) / 'networks'
    output.mkdir(parents=True, exist_ok=True)
    call(interpreter, SCRIPT, '--worker', '--phase', phase, '--seed', seed, '--out', output,
         '--modes', *modes, *worker_options(args))
    frames, manifests = {}, {}
    for mode in modes:
        path = output / mode
        lock = path / 'predictions_frozen.json'
        if read_json(lock)['sources']['settings_sha256'] != sha256(output / 'settings.json'):
            raise ValueError('The network inputs do not match its predictions.')
        frames[mode] = fetch_power(path, 'network', keys)
        manifests[mode] = sha256(lock)
    return frames, manifests, output


def wind_scores(truth, output):
    observed = truth.wind.to_numpy(dtype=float)
    aligned = {}
    common = np.isfinite(observed) & (observed >= 0)
    for mode in MODES:
        path = output / mode / 'wind.csv'
        receipt = read_json(path.parent / 'predictions_frozen.json')
        if sha256(path) != receipt['sources']['wind_sha256']:
            raise ValueError('Wind forecast hash changed.')
        raw = pd.read_csv(path, dtype={'date': str, 'station_id': str})
        aligned_wind = truth[POINT_KEYS].merge(raw, validate='one_to_one', on=POINT_KEYS, how='left')
        values = aligned_wind.wind.to_numpy()
        aligned[mode] = values
        common &= np.isfinite(values)
    if not common.any():
        return {'points': 0, 'mae': {}}
    return {'points': int(common.sum()), 'mae': {mode: float(np.abs(values[common] - observed[common]).mean())
                                               for mode, values in aligned.items()}}


def execute(args, interpreter):
    setup = {'device': args.device, 'split': sha256(args.split), 'wind_reference': str(args.wind_run.resolve()),
             'power_control': str(args.power_run.resolve()), 'epochs': EPOCHS, 'wind_weight': WIND_WEIGHT,
             'code': {name: sha256(SCRIPT.with_name(name)) for name in (SCRIPT.name, 'warm_shared_power.py',
                      'run_t1_t3_batch.py', 'same_issue_weather.py', 'run_wind_attribution.py', 'score_forecasts.py')}}
    freeze_json(args.out / 'setup.json', setup)
    roster = load_roster(args.data_root)
    train, days = chronological_dates(args.split, 'A')
    keys = forecast_keys(days, roster)
    show_progress(args.out, 'T1 风速辅助', 0, 3, '时序 A 对照与两组辅助监督')
    baseline, old = prior_baseline(args, 'A', keys, A_SEED)
    control, control_lock = control_forecast(args, settings(args, train, days, roster, A_SEED), keys)
    options, locks, network_dir = fit_candidates(args, 'A', A_SEED, MODES, keys, interpreter)
    models = {'power_only': control, **options}
    arms = {'baseline': baseline, **models, **{name + '_blend': blend(baseline, value, .20, 'T1')
                                              for name, value in models.items()}}
    directory = args.out / 'A'
    arms, lock = freeze_window(directory / 'comparison', 'T1', keys, arms,
                               {'prior': old, 'control': control_lock, 'aux': locks})
    truth = wind_labels(args.data_root, days, roster)
    result = block_comparisons(truth, arms, roster, tuple(name for name in arms if name != 'baseline'))
    scores = {name: score[PRIMARY_SCORE] for name, score in result['scores'].items()}
    candidate = max(MODES, key=lambda name: scores[name + '_blend'])
    selected = scores[candidate + '_blend']
    gate = {'mode': candidate, 'promoted': selected > max(scores['baseline'], scores['power_only_blend']),
            'delta_vs_baseline': selected - scores['baseline'], 'delta_vs_control': selected - scores['power_only_blend']}
    diagnostics = wind_scores(truth, network_dir)
    freeze_json(directory / 'score.json', {'gate': gate, 'predictions': lock, 'result': result, 'wind': diagnostics})
    summary = {'A': {'scores': scores, 'gate': gate, 'wind_mae_raw_label_units': diagnostics}}
    write_json(args.out / 'summary.json', summary)
    print(f'[T1 A] {scores}，准入 {gate}', flush=True)
    show_progress(args.out, 'T1 风速辅助', 1, 3, '按时序 A 结果决定 B')
    if gate['promoted']:
        _, days = chronological_dates(args.split, 'B')
        keys = forecast_keys(days, roster)
        directory = args.out / 'B'
        directory.mkdir(exist_ok=True)
        pairs, manifests = {}, {}
        freeze_json(directory / 'gate.json', {**gate, 'A_sha256': sha256(args.out / 'A' / 'score.json')})
        for i, seed in enumerate(B_SEEDS, 1):
            show_progress(directory, 'T1 B 预测', i - 1, len(B_SEEDS), f'seed={seed}')
            baseline, prior_lock = prior_baseline(args, 'B', keys, seed)
            variants, hashes, _ = fit_candidates(args, 'B', seed, (candidate,), keys, interpreter)
            pair = {'baseline': baseline, 'candidate': blend(baseline, variants[candidate], .20, 'T1')}
            pairs[seed], manifests[str(seed)] = freeze_window(directory / str(seed) / 'pair', 'T1', keys, pair,
                                                              {'baseline': prior_lock, 'network': hashes[candidate]})
        digest = freeze_json(directory / 'predictions_frozen.json', manifests)
        show_progress(directory, 'T1 B 预测', 3, 3, '三种子预测已锁定')
        event(directory, 'predictions_frozen', status='complete', sha256=digest)
        report = paired_report(directory, 'T1', labels(args.data_root, days, roster), roster, keys, pairs, digest)
        summary['B'] = {'acceptance': report['acceptance'],
                        'seed_deltas': {seed: pair['delta'] for seed, pair in report['pairs'].items()},
                        'ensemble': {name: row[PRIMARY_SCORE] for name, row in report['ensemble']['scores'].items()}}
    else:
        print('[T1 B] A 未达到辅助监督准入', flush=True)
    show_progress(args.out, 'T1 风速辅助', 2, 3, '汇总配对报告')
    write_json(args.out / 'summary.json', summary)
    show_progress(args.out, 'T1 风速辅助', 3, 3, '完成')
    print(f'摘要 {args.out / "summary.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RUNS / 't1_wind_aux_gpu_v1')
    parser.add_argument('--wind-run', type=Path, default=RUNS / 'wind_attribution_gpu_v1')
    parser.add_argument('--power-run', type=Path, default=RUNS / 't1_t3_fast_gpu_v1')
    parser.add_argument('--mlp-run', type=Path, default=RUNS / 'warm_shared_power_v1')
    parser.add_argument('--mlp-python', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--phase', choices=('A','B'), default='A')
    parser.add_argument('--seed', type=int, choices=B_SEEDS, default=A_SEED)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    args.mlp_weather_python = None
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        fit_networks(args)
        return
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args, select_mlp_runtime(args, require_weather=False)[0])
        except (Exception, KeyboardInterrupt) as error:
            event(args.out, 'run', status='failed', detail=str(error))
            raise
        event(args.out, 'run', status='complete')


if __name__ == '__main__':
    main()
