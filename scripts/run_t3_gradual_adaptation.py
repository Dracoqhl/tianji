#!/usr/bin/env python3
"""Fine-tune existing T3 source checkpoints without changing the weather schema."""
import argparse
import fcntl
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic

import numpy as np

from advance_batch_b import read_frozen_forecast
from batch_energy_short import roster_for, t3_windows
from compare_t2_mlp import select_mlp_runtime
from compare_warm_t1 import PRIMARY_SCORE
from confirm_working_baselines import score_t3
from experiment_companion_t1 import assert_matched, read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, date_roles, sha256, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from run_steamlink_followup import confirmed_t3
from run_t1_t3_batch import call, paired_report, station_day_weights
from same_issue_weather import archive_stamp, archived_issue_weather
from score_forecasts import MAX_PREDICTION
from screen_companion_t1_stations import show_progress
from screen_same_issue_context import CONTEXTS, make_context_network
from screen_short_transfer import pretrain_dates, targets
from warm_shared_power import labels, load_roster

ENTRY = Path(__file__).resolve()
RUNS = Path('/workspace/qhl/runs')
FIXED = 'fixed_50_10'
HEAD = 'head_only'
GRADUAL = 'gradual_last_block'
MODES = (HEAD, GRADUAL)
EPOCHS = 6
WARMUP = 3
BATCH = 64
HEAD_LR = 1e-3
BLOCK_LR = HEAD_LR / 10
TOLERANCE = 1e-8


def checkpoint_contract(args, window, train, days, roster, seed):
    folder = args.transfer_run / window
    marker = folder / 'predictions_frozen.json'
    record = read_json(marker)
    plan = record['plan']
    allowed, _ = date_roles(args.split, 'A')
    donor_dates = pretrain_dates(allowed, days)
    if not donor_dates or max(donor_dates) >= min(days):
        raise ValueError('Invalid transfer source cutoff.')
    expected = {'device': args.device, 'seed': A_SEED, 'fit_dates': list(map(str, train)),
                'score_dates': list(map(str, days)), 'source_dates': list(map(str, donor_dates)),
                'source_sites': load_roster(args.data_root), 'target_sites': roster,
                'cache': archive_stamp(args.workdir, sorted(set(donor_dates + train + days))),
                'station_coordinates_sha256': sha256(args.data_root / 'static/power_stations.csv')}
    assert_matched(plan, expected, expected)
    keys, tree, transfer, baseline, forecast_lock = confirmed_t3(args, window, train, days, roster, seed)
    fit = record['fit']
    if seed != A_SEED:
        seed_dir = args.t3_confirm / 'T3' / str(seed)
        folder = seed_dir / 'transfer'
        receipt = read_json(folder / 'receipt.json')
        seed_lock = read_json(seed_dir / 'predictions_frozen.json')
        if window != FIXED or receipt['seed'] != seed or receipt != seed_lock['sources']['fit']['transfer']:
            raise ValueError('T3 checkpoint receipt does not match the fixed-window locked predictions.')
        fit = receipt['fit']
    path = folder / 'frozen_encoder.pt'
    receipt = {'source_lock': sha256(marker), 'paired_reference_lock': forecast_lock,
               'checkpoint_sha256': sha256(path), 'forecast_dates': list(map(str, days)),
               'train_dates': list(map(str, train)), 'seed': seed, 'device': args.device}
    return keys, tree, transfer, baseline, fit, path, receipt


def ridge_parameters(fit, roster, width):
    weights, intercepts = [], []
    for site in roster:
        head = fit['heads'][site]
        if not head['points']:
            weights.append(np.zeros(width))
            intercepts.append(0.)
            continue
        params = np.asarray(head['coefficients'], dtype=float)
        center = np.asarray(head['feature_center'], dtype=float)
        scale = np.asarray(head['feature_scale'], dtype=float)
        slopes = params[:-1] / scale
        if slopes.shape != (width,) or center.shape != (width,):
            raise ValueError(f'{site}: ridge head and encoder widths differ.')
        weights.append(slopes)
        intercepts.append(params[-1] - center @ slopes)
    cap = MAX_PREDICTION['T3']
    result = np.asarray(weights, dtype=np.float32) / cap, np.asarray(intercepts, dtype=np.float32) / cap
    if not all(np.isfinite(field).all() for field in result):
        raise ValueError('Nonfinite station head initialization.')
    return result


def adapt_model(checkpoint, fit, train_weather, test_weather, actual, valid, roster, args, mode, expected):
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    network = make_context_network(train_weather.shape[-1], CONTEXTS['full_context']).to(device)
    network.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device), strict=True)
    network.requires_grad_(False)
    network.eval()
    block = network.convolutions[-1]
    slopes, offsets = ridge_parameters(fit, roster, network.output.in_features)
    slopes = torch.nn.Parameter(torch.as_tensor(slopes, device=device))
    offsets = torch.nn.Parameter(torch.as_tensor(offsets, device=device))
    optimizer = torch.optim.AdamW([
        {'params': [slopes, offsets], 'lr': HEAD_LR, 'weight_decay': 0.},
        {'params': list(block.parameters()), 'lr': BLOCK_LR, 'weight_decay': 0.},
    ])
    sites = len(roster)
    cap = MAX_PREDICTION['T3']

    def power(weather, station):
        encoded = network.encode(weather)
        raw = torch.sigmoid(network.output(encoded).squeeze(-1))
        correction = (encoded * slopes[station, None]).sum(dim=-1) + offsets[station, None]
        return raw + correction

    test = torch.as_tensor(test_weather, device=device)
    test_station = torch.arange(len(test), device=device) % sites

    def predict():
        with torch.inference_mode():
            values = [power(test[start:start+BATCH], test_station[start:start+BATCH]).cpu().numpy()
                      for start in range(0, len(test), BATCH)]
        return np.clip(np.concatenate(values).astype(float) * cap, 0, cap)

    restored = predict()
    difference = float(np.abs(restored.ravel() - expected).max())
    if difference > 1e-4:
        raise ValueError(f'The initialized ridge forecast differs from the locked one by {difference:.6g} MW.')
    before = [p.detach().clone() for p in network.parameters()]
    x = torch.as_tensor(train_weather, device=device)
    y = torch.as_tensor(np.where(valid, actual / cap, 0), device=device, dtype=torch.float32)
    mask = torch.as_tensor(valid, device=device)
    station = torch.arange(len(x), device=device) % sites
    weights = station_day_weights(valid, sites)
    keep = torch.as_tensor(np.flatnonzero(weights > 0), device=device)
    weights = torch.as_tensor(weights, device=device, dtype=torch.float32)
    trace = []
    for epoch in range(1, EPOCHS+1):
        unfrozen = mode == GRADUAL and epoch > WARMUP
        block.requires_grad_(unfrozen)
        order = keep[torch.randperm(len(keep), device=device)]
        total = 0.
        for batch in order.split(BATCH):
            error = torch.where(mask[batch], power(x[batch], station[batch]) - y[batch], 0.)
            rmse = torch.sqrt(error.square().sum(dim=1) / mask[batch].sum(dim=1) + 1e-8)
            loss = (rmse * weights[batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float((rmse.detach() * weights[batch]).sum())
        trace.append({'epoch': epoch, 'last_block_unfrozen': unfrozen, 'train_rmse_mw': total / len(keep) * cap})
        print(f'[T3 {mode}] epoch {epoch}/{EPOCHS}，主干末块{"解冻" if unfrozen else "冻结"}，'
              f'训练站日 RMSE {trace[-1]["train_rmse_mw"]:.4f} MW', flush=True)
    block_ids = {id(value) for value in block.parameters()}
    changes = [(id(p) in block_ids, not torch.equal(old, p.detach())) for old, p in zip(before, network.parameters())]
    if any(changed and (not is_block or mode == HEAD) for is_block, changed in changes):
        raise ValueError('A frozen source parameter changed during T3 adaptation.')
    result = predict()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite T3 adaptation forecast.')
    return result, {'ridge_replay_max_abs_mw': difference, 'epochs': trace,
                    'last_block_changed': any(changed for is_block, changed in changes if is_block),
                    'head_lr': HEAD_LR, 'block_lr': BLOCK_LR}


def run_worker(args):
    roster = roster_for(args.data_root, 'T3')
    _, train, days = next(item for item in t3_windows() if item[0] == args.window)
    keys, _, original, _, fit, checkpoint, source = checkpoint_contract(
        args, args.window, train, days, roster, args.seed)
    weather, _, _ = archived_issue_weather(args.workdir, args.data_root, [*train, *days], roster, print)
    norm = fit['source_normalization']
    mean, scale = np.asarray(norm['mean'], dtype=float), np.asarray(norm['scale'], dtype=float)
    present = np.isfinite(weather)
    weather = np.concatenate([np.where(present, (weather - mean) / scale, 0), ~present], axis=-1).astype(np.float32)
    cut = len(train) * len(roster)
    actual, valid = targets(args.data_root, train, roster)
    expected = original.prediction.to_numpy()
    forecasts, training = {}, {}
    for mode in args.modes:
        result, training[mode] = adapt_model(checkpoint, fit, weather[:cut], weather[cut:], actual, valid, roster,
                                             args, mode, expected)
        forecasts[mode] = keys.assign(prediction=result.ravel())
    freeze_window(args.out, 'T3', keys, forecasts, {'inputs': source, 'training': training})


def predictions(args, window, train, days, roster, seed, modes, interpreter):
    keys, tree, original, baseline, _, _, expected = checkpoint_contract(args, window, train, days, roster, seed)
    target = args.out / window / str(seed)
    target.mkdir(parents=True, exist_ok=True)
    trained = target / 'neural'
    marker = trained / 'predictions_frozen.json'
    if not marker.is_file():
        folder = Path(mkdtemp(dir=target, prefix='.adapt-attempt-'))
        options = ['--worker', '--window', window, '--seed', seed, '--out', folder, '--modes', *modes]
        for key in ('data_root', 'workdir', 'transfer_run', 'short_tree', 't3_confirm', 'split', 'device', 'threads'):
            options.extend((f'--{key.replace("_", "-")}', getattr(args, key)))
        call(interpreter, ENTRY, *options)
        folder.rename(trained)
    record = read_json(marker)
    if record['sources']['inputs'] != expected or set(record['hashes']) != set(modes):
        raise ValueError('Cached T3 adaptation provenance or modes changed.')
    neural = {name: read_frozen_forecast(trained / f'{name}.csv', digest, keys, 'T3')
              for name, digest in record['hashes'].items()}
    arms = {'baseline': baseline, 'frozen_ridge': original, 'short_tree': tree}
    for mode, forecast in neural.items():
        arms[mode] = blend(tree, forecast, .25, 'T3')
        arms[mode + '_single'] = forecast
    frames, lock = freeze_window(target / 'forecasts', 'T3', keys, arms,
                                 {'checkpoint': expected, 'neural_lock': sha256(marker)})
    return keys, frames, lock


def execute(args):
    start = monotonic()
    roster = roster_for(args.data_root, 'T3')
    windows = list(t3_windows())
    args.mlp_weather_python = None
    interpreter = select_mlp_runtime(args, require_weather=False)[0]
    config = {'script': sha256(ENTRY), 'device': args.device, 'threads': args.threads,
              'split': sha256(args.split), 'seed': A_SEED, 'b_seeds': list(B_SEEDS),
              'head_lr': HEAD_LR, 'block_lr': BLOCK_LR, 'epochs': EPOCHS, 'warmup': WARMUP, 'batch': BATCH,
              'sites': roster, 'paths': {key: str(getattr(args, key).resolve()) for key in
                 ('data_root', 'workdir', 'short_tree', 'transfer_run', 't3_confirm')}}
    freeze_json(args.out / 'plan.json', config)
    early = {}
    for index, (window, train, days) in enumerate(windows[:2]):
        directory = args.out / window
        directory.mkdir(exist_ok=True)
        show_progress(args.out, 'T3 早窗', index, 2, window)
        _, arms, lock = predictions(args, window, train, days, roster, A_SEED, MODES, interpreter)
        truth = labels(args.data_root, days, roster)
        report = score_t3(truth, roster, {**arms, 'candidate': arms[GRADUAL]})
        scores = {name: row[PRIMARY_SCORE] for name, row in report['scores'].items() if name != 'candidate'}
        deltas = {mode: scores[mode] - scores['baseline'] for mode in MODES}
        freeze_json(directory / 'score.json', {'scores': report['scores'], 'deltas': deltas, 'lock': lock})
        print(f'[T3 {window}] {scores}，配对差 {deltas}', flush=True)
        early[window] = {'scores': scores, 'deltas': deltas, 'lock': lock}
    means = {mode: float(np.mean([result['deltas'][mode] for result in early.values()])) for mode in MODES}
    chosen = GRADUAL if means[GRADUAL] > means[HEAD] + TOLERANCE else HEAD
    gate = {'selected': chosen, 'mean_early_delta': means, 'promoted': means[chosen] > TOLERANCE}
    freeze_json(args.out / 'a_score.json', {'windows': early, 'gate': gate})
    show_progress(args.out, 'T3 早窗', 2, 2, f'准入 {gate}')
    summary = {'early': early, 'gate': gate}
    write_json(args.out / 'summary.json', summary)
    if gate['promoted']:
        window, train, days = windows[-1]
        folder = args.out / window
        pairs, locks = {}, {}
        for index, seed in enumerate(B_SEEDS):
            show_progress(args.out, 'T3 50/10', index, len(B_SEEDS), f'seed={seed}')
            keys, frames, digest = predictions(args, window, train, days, roster, seed, (chosen,), interpreter)
            pairs[seed], locks[str(seed)] = freeze_window(folder / str(seed) / 'pair', 'T3', keys,
                {'baseline': frames['baseline'], 'candidate': frames[chosen]}, {'adapted': digest, 'gate': gate})
        digest = freeze_json(folder / 'predictions_frozen.json', locks)
        report = paired_report(folder, 'T3', labels(args.data_root, days, roster), roster, keys, pairs, digest)
        summary['B'] = {'acceptance': report['acceptance'], 'scores': {
            arm: value[PRIMARY_SCORE] for arm, value in report['ensemble']['scores'].items()},
            'seed_deltas': {key: value['delta'] for key, value in report['pairs'].items()},
            'station_deltas': report['ensemble']['station_deltas'], 'identical_seed_groups': report['identical_seed_groups']}
        show_progress(args.out, 'T3 50/10', len(B_SEEDS), len(B_SEEDS), '计分完成')
    summary['seconds'] = monotonic() - start
    write_json(args.out / 'summary.json', summary)
    print(f'结果摘要 {args.out / "summary.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RUNS / 't3_gradual_adaptation_gpu_v1')
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--short-tree', type=Path, default=RUNS / 'three_track_gpu_batch_v1/T3')
    parser.add_argument('--transfer-run', type=Path, default=RUNS / 'same_issue_followup_v1/T3')
    parser.add_argument('--t3-confirm', type=Path, default=RUNS / 't3_transfer_confirmation_gpu_v1')
    parser.add_argument('--mlp-run', type=Path, default=RUNS / 'warm_shared_power_v1')
    parser.add_argument('--mlp-python', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--window', choices=[window[0] for window in t3_windows()])
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--seed', type=int, choices=B_SEEDS, default=A_SEED)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if args.window is None:
            parser.error('--worker requires --window')
        run_worker(args)
    else:
        with (args.out / '.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(args)


if __name__ == '__main__':
    main()
