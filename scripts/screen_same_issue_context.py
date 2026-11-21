#!/usr/bin/env python3
"""Paired T1 screen for extra context within a single 96-point forecast issue."""

import argparse
import fcntl
from functools import partial
from pathlib import Path
from tempfile import mkdtemp

from advance_batch_b import read_frozen_forecast
from batch_energy_short import t2_windows
from compare_warm_t1 import PRIMARY_SCORE, daily_scores
from experiment_companion_t1 import read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, date_roles, forecast_keys, sha256, write_json
from same_issue_weather import archive_stamp, warm_archive_weather
from score_forecasts import score_track
from screen_companion_t1_stations import frozen_pooled_forecast, frozen_pooled_origin, show_progress
from warm_shared_power import fit_forecasts, labels, load_roster


CONTEXTS = {'local_context': (1, 1, 1, 1, 1, 1, 1), 'full_context': (1, 2, 4, 8, 16, 32, 64)}


def make_context_network(feature_count, dilations):
    import torch.nn as nn
    import torch.nn.functional as F

    class CurveModel(nn.Module):
        def __init__(self):
            super().__init__()
            width = 32
            self.project = nn.Sequential(nn.Linear(feature_count, width), nn.LayerNorm(width), nn.SiLU())
            self.convolutions = nn.ModuleList([
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation) for dilation in dilations
            ])
            self.output = nn.Linear(width, 1)

        def encode(self, weather):
            features = self.project(weather).transpose(1, 2)
            for convolution in self.convolutions:
                features = features + 0.5 * F.silu(convolution(features))
            return features.transpose(1, 2)

        def forward(self, weather):
            return self.output(self.encode(weather))

    return CurveModel()


def run_forecast(args, name, train, days, keys, directory, *, seed=A_SEED, phase='A'):
    directory = directory / name
    tag = 'sealed' if phase == 'B' else 'development'
    csv_name = f'T1_{tag}.csv'
    settings = {'data_root': str(args.data_root.resolve()), 'device': args.device, 'seed': seed,
                'epochs': 16, 'batch_station_days': 64, 'dilations': list(CONTEXTS[name]),
                'fit_dates': [str(day) for day in train], 'forecast_dates': [str(day) for day in days],
                'archive': archive_stamp(args.workdir, [*train, *days]),
                'station_coordinates_sha256': sha256(args.data_root / 'static/power_stations.csv'),
                'code_sha256': {file: sha256(Path(__file__).with_name(file))
                                for file in (Path(__file__).name, 'warm_shared_power.py', 'same_issue_weather.py')}}
    if not directory.exists():
        pending = Path(mkdtemp(dir=directory.parent, prefix=f'.{name}.attempt-'))
        fit_forecasts(args.data_root, pending, train, days, args.workers, 16, 64, args.device,
                      seed=seed, forecast_tag=tag, weather_loader=partial(warm_archive_weather, args.workdir),
                      network_factory=partial(make_context_network, dilations=CONTEXTS[name]))
        digest = sha256(pending / csv_name)
        read_frozen_forecast(pending / csv_name, digest, keys, 'T1')
        write_json(pending / 'frozen.json', {'settings': settings, 'forecast_sha256': digest})
        pending.rename(directory)
    record = read_json(directory / 'frozen.json')
    if record['settings'] != settings:
        raise ValueError(f'{name} was frozen with different training dates or settings.')
    return read_frozen_forecast(directory / csv_name, record['forecast_sha256'], keys, 'T1'), record['forecast_sha256']


def evaluate_window(args, name, train, days, roster, origin):
    target = args.out / name
    target.mkdir(parents=True, exist_ok=True)
    keys = forecast_keys(days, roster)
    predictions, hashes = {}, {}
    for model in CONTEXTS:
        predictions[model], hashes[model] = run_forecast(args, model, train, days, keys, target)
    if origin:
        receipt, predictions['pooled_xgboost'], _ = frozen_pooled_forecast(
            args.tree_run, origin, name, train, days, keys, device=args.device,
        )
        if Path(receipt['data_root']).resolve() != args.data_root.resolve():
            raise ValueError('The frozen tree controls use a different data root.')
        hashes['pooled_xgboost'] = receipt['prediction_sha256']
    report = target / 'score.json'
    if report.is_file():
        result = read_json(report)
        if result['prediction_sha256'] != hashes:
            raise ValueError('Forecast hashes differ from the scored comparison.')
        return result
    truth = labels(args.data_root, days, roster)
    scores = {model: score_track('T1', truth, pred, roster, allow_unscored_stations=True)
              for model, pred in predictions.items()}

    daily = {model: daily_scores(truth, pred) for model, pred in predictions.items()}
    per_station = {model: (100 * values.groupby(level='station_id').mean()).to_dict()
                   for model, values in daily.items()}
    comparisons = {}
    for reference in ('local_context', 'pooled_xgboost'):
        if reference in per_station:
            comparisons[reference] = {site: score - per_station[reference][site]
                                      for site, score in per_station['full_context'].items()}
    monthly = {}
    dates = truth.date.str[:7].drop_duplicates().tolist()
    if len(dates) > 1:
        for month in dates:
            monthly[month] = {
                model: float(100 * frame.loc[frame.index.get_level_values('date').str.startswith(month)]
                             .groupby(level='station_id').mean().mean()) for model, frame in daily.items()
            }

    result = {'prediction_sha256': hashes, 'train_days': len(train), 'evaluation_days': len(days),
              'scores': scores, 'per_station_scores': per_station, 'full_context_station_deltas': comparisons,
              'monthly_readme': monthly}
    write_json(report, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--tree-run', type=Path)
    parser.add_argument('--window', choices=('training', 'A', 'all'), default='training')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--predict-b', type=int, choices=B_SEEDS, metavar='SEED')
    args = parser.parse_args()
    roster = load_roster(args.data_root)
    origin = frozen_pooled_origin(args.tree_run) if args.tree_run else None
    windows = [(name, train, held) for name, train, held in t2_windows(args.split)
               if args.window == 'all' or (name == 'A') == (args.window == 'A')]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.predict_b is not None:
            train, days = date_roles(args.split, 'B')
            run_forecast(args, 'full_context', train, days, forecast_keys(days, roster), args.out,
                         seed=args.predict_b, phase='B')
            return
        reports = {}
        for step, (name, train, held) in enumerate(windows, 1):
            show_progress(args.out, '同日起报轨迹', step - 1, len(windows), name)
            reports[name] = evaluate_window(args, name, train, held, roster, origin)
            values = reports[name]['scores']
            print(f'[{name}] ' + '  '.join(f'{arm}={score[PRIMARY_SCORE]:.6f}' for arm, score in values.items()))
        show_progress(args.out, '同日起报轨迹', len(windows), len(windows), '完成')
        write_json(args.out / f'{args.window}_reports.json', reports)


if __name__ == '__main__':
    main()
