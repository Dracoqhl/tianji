#!/usr/bin/env python3
"""Compare frozen regular-site representation transfer for the fixed T3 windows."""

import argparse
import fcntl
from functools import partial
from pathlib import Path

import numpy as np

from advance_batch_b import read_frozen_forecast
from batch_energy_short import roster_for, save_csv, t3_windows
from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import read_json
from run_companion_warm_t1 import A_SEED, SPLIT_PATH, date_roles, forecast_keys, sha256, write_json
from same_issue_weather import archive_stamp, archived_issue_weather
from score_forecasts import MAX_PREDICTION, score_track
from screen_companion_t1_stations import show_progress, station_scores
from screen_same_issue_context import CONTEXTS, make_context_network
from warm_shared_power import fit_power, labels, load_roster, normalize


NETWORK = partial(make_context_network, dilations=CONTEXTS['full_context'])
RIDGE = 0.05
MODES = ('scratch_context', 'frozen_context_ridge')


def pretrain_dates(allowed, forecast_days):
    return [day for day in allowed if day < min(forecast_days)]


def targets(root, dates, roster):
    truth = labels(root, dates, roster)
    values = truth.power.to_numpy(dtype=float).reshape(-1, 96)
    valid = truth.power_limit.eq(1).to_numpy().reshape(values.shape) & np.isfinite(values)
    return values, valid


def encode(model, weather, device):
    import torch

    model.eval()
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(weather), 64):
            matrix = torch.as_tensor(weather[start:start+64], device=device)
            embeddings.append(model.encode(matrix).cpu().numpy())
    return np.concatenate(embeddings)


def ridge_residual(values, weights, embedding, forecast_embedding):
    weights = weights / weights.sum()
    center = weights @ embedding
    scale = np.sqrt(weights @ ((embedding - center)**2)).clip(1e-3)
    matrix = np.column_stack([(embedding - center) / scale, np.ones(len(embedding))])
    forecast = (forecast_embedding - center) / scale
    penalty = np.eye(matrix.shape[1]) * RIDGE
    penalty[-1, -1] = 0
    beta = np.linalg.solve(matrix.T @ (matrix * weights[:, None]) + penalty, matrix.T @ (weights * values))
    return forecast @ beta[:-1] + beta[-1], beta, center, scale


def adjust_short(encoder, predict, train, forecast, power, valid, roster, device):
    embeddings = encode(encoder, np.concatenate((train, forecast)), device)
    raw = predict()
    final = raw[len(train):].copy()
    fit_encoded, forecast_encoded = embeddings[:len(train)], embeddings[len(train):]
    heads = {}
    for index, station in enumerate(roster):
        station_train = slice(index, None, len(roster))
        mask = valid[station_train]
        counts = mask.sum(axis=1, keepdims=True)
        weights = np.divide(mask, counts, out=np.zeros(mask.shape), where=counts > 0)
        if not mask.any():
            heads[station] = {'points': 0}
            continue
        residual = (power[station_train] - raw[:len(train)][station_train])[mask]
        change, beta, center, scale = ridge_residual(residual, weights[mask], fit_encoded[station_train][mask],
                                                      forecast_encoded[station_train])
        final[station_train] += change
        heads[station] = {'points': int(mask.sum()), 'coefficients': beta.tolist(),
                          'feature_center': center.tolist(), 'feature_scale': scale.tolist()}
    return np.clip(final, 0, MAX_PREDICTION['T3']), heads


def fit_models(args, source, regular, short, train, days, directory):
    import torch

    def weather(dates, sites):
        return archived_issue_weather(args.workdir, args.data_root, dates, sites, print)[0]

    short_weather = weather([*train, *days], short)
    cut = len(train) * len(short)
    actual, valid = targets(args.data_root, train, short)
    train_short, eval_short, _, _ = normalize(short_weather[:cut], short_weather[cut:])
    print('[T3] 短历史站从头拟合', flush=True)
    scratch, predict_scratch, scratch_loss, _, _ = fit_power(
        train_short, actual, valid, eval_short, 16, 64, args.device, print, seed=A_SEED, network_factory=NETWORK,
    )
    scratch_prediction = predict_scratch()
    torch.save(scratch.state_dict(), directory / 'scratch.pt')
    del scratch, predict_scratch, train_short, eval_short
    source_weather, target_weather, mean, scale = normalize(weather(source, regular), short_weather)
    source_power, source_valid = targets(args.data_root, source, regular)
    print(f'[T3] 常规站预训练 {len(source)} 天，随后冻结主干', flush=True)
    encoder, predict_pretrained, source_loss, _, _ = fit_power(
        source_weather, source_power, source_valid, target_weather, 16, 64, args.device, print,
        seed=A_SEED, network_factory=NETWORK,
    )
    transferred, heads = adjust_short(encoder, predict_pretrained, target_weather[:cut], target_weather[cut:],
                                      actual, valid, short, args.device)
    torch.save(encoder.state_dict(), directory / 'frozen_encoder.pt')
    return dict(zip(MODES, (scratch_prediction, transferred))), {
        'source_train_daily_rmse_mw': source_loss, 'scratch_train_daily_rmse_mw': scratch_loss,
        'source_normalization': {'mean': mean.tolist(), 'scale': scale.tolist()}, 'heads': heads,
    }


def run_window(args, name, train, days, allowed, regular, short, previous):
    source = pretrain_dates(allowed, days)
    if not source or set(train) & set(days) or max(train) >= min(days):
        raise ValueError('Invalid T3 source or target chronology.')
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    keys = forecast_keys(days, short)
    reference = previous['windows'][name]
    if (reference['window'] != [str(day) for day in days] or reference['train_days'] != len(train)
            or previous['settings']['split_sha256'] != sha256(args.split)
            or Path(previous['settings']['data_root']).resolve() != args.data_root.resolve()):
        raise ValueError('The original T3 control differs from the registered target window.')
    digest = reference['predictions_sha256']['short_only']
    predictions = {'short_only_xgboost': read_frozen_forecast(
        args.tree_run / name / 'short_only.csv', digest, keys, 'T3',
    )}
    plan = {'device': args.device, 'seed': A_SEED, 'epoch_budget': 16, 'ridge': RIDGE, 'fit_dates': [str(day) for day in train],
            'score_dates': [str(day) for day in days], 'source_dates': [str(day) for day in source],
            'source_sites': regular, 'target_sites': short,
            'cache': archive_stamp(args.workdir, sorted(set(source + train + days))),
            'station_coordinates_sha256': sha256(args.data_root / 'static/power_stations.csv'),
            'code_sha256': {file: sha256(Path(__file__).with_name(file)) for file in (
                Path(__file__).name, 'warm_shared_power.py', 'screen_same_issue_context.py', 'same_issue_weather.py')},
            'reference_sha256': digest}
    freeze = out / 'predictions_frozen.json'
    if freeze.is_file():
        record = read_json(freeze)
        if record['plan'] != plan:
            raise ValueError('T3 forecasts were generated with a different plan.')
    else:
        forecast, fit = fit_models(args, source, regular, short, train, days, out)
        hashes = {arm: save_csv(out / f'{arm}.csv', keys.assign(prediction=values.ravel()))
                  for arm, values in forecast.items()}
        record = {'plan': plan, 'forecasts': hashes, 'fit': fit}
        write_json(freeze, record)
    for arm in MODES:
        predictions[arm] = read_frozen_forecast(out / f'{arm}.csv', record['forecasts'][arm], keys, 'T3')
    path = out / 'score.json'
    lock_sha256 = sha256(freeze)
    if path.is_file():
        result = read_json(path)
        if result['prediction_lock_sha256'] != lock_sha256:
            raise ValueError('T3 report does not match its frozen forecasts.')
        return result
    truth = labels(args.data_root, days, short)
    scores = {name: dict(score_track('T3', truth, pred, short, allow_unscored_stations=True),
                         by_station=station_scores(truth, pred)) for name, pred in predictions.items()}
    result = {'prediction_lock_sha256': lock_sha256, 'source_train_days': len(source),
              'target_train_days': len(train), 'score_days': len(days), 'scores': scores}
    write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--tree-run', type=Path, default=Path('/workspace/qhl/runs/three_track_gpu_batch_v1/T3'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    regular, short = load_roster(args.data_root), roster_for(args.data_root, 'T3')
    allowed, _ = date_roles(args.split, 'A')
    previous = read_json(args.tree_run / 'results.json')
    windows = list(t3_windows())
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {}
        for step, (name, train, days) in enumerate(windows, 1):
            show_progress(args.out, 'T3 短历史迁移', step - 1, len(windows), name)
            results[name] = run_window(args, name, train, days, allowed, regular, short, previous)
            values = results[name]['scores']
            print(f'[T3 {name}] ' + '  '.join(f'{arm}={score[PRIMARY_SCORE]:.6f}' for arm, score in values.items()), flush=True)
        show_progress(args.out, 'T3 短历史迁移', len(windows), len(windows), '完成')
        write_json(args.out / 'results.json', results)


if __name__ == '__main__':
    main()
