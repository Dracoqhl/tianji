#!/usr/bin/env python3
"""Compare weather-to-wind error mechanisms and run a two-site cold-start replay."""

import argparse
import fcntl
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic

import numpy as np
import pandas as pd

from advance_batch_b import read_frozen_forecast
from batch_energy_short import fit_short_forecasts, roster_for, short_source
from compare_t2_mlp import select_mlp_runtime
from companion_power_curve import monotone
from compare_warm_t1 import PRIMARY_SCORE
from confirm_working_baselines import keep_sites
from experiment_companion_t1 import event, read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, SPLIT_PATH, chronological_dates, forecast_keys
from run_companion_warm_t1 import block_early_stop_days, sha256, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from run_t1_t3_batch import call, paired_report, point_diagnostics, shared_head, timed, worker_options
from same_issue_weather import archive_stamp, archived_issue_weather
from score_forecasts import DAY_KEYS, MAX_PREDICTION, MIN_VALID_POINTS, POINT_KEYS
from screen_companion_t1_stations import block_comparisons, show_progress, station_scores
from screen_short_transfer import fit_models
from warm_shared_power import labels, load_roster


SCRIPT = Path(__file__).resolve()
RUNS = Path('/workspace/qhl/runs')
SOURCES = ('tianji', 'ec', 'aifs')
STATION = 'KNTBH4'
RIDGE_ALPHA = 0.01


def wind_labels(root, dates, roster):
    fields = [*POINT_KEYS, 'wind', 'power', 'power_limit']
    frames = [pd.read_csv(root / 'train' / f'{day:%Y%m%d}_0000' / 'power_target.csv', usecols=fields,
                          dtype={'date': str, 'station_id': str}) for day in dates]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[raw.station_id.isin(roster), fields]
    keys = forecast_keys(dates, roster)
    if len(raw) != len(keys):
        raise ValueError('Wrong number of wind-label rows.')
    aligned = keys.merge(raw, on=POINT_KEYS, validate='one_to_one', indicator=True)
    if not aligned['_merge'].eq('both').all() or len(aligned) != len(keys):
        raise ValueError('Wrong dates or keys in wind labels.')
    return aligned.drop(columns='_merge')


def wind_features(data, columns, source, conditioned):
    u, v = (data[..., columns.index(f'{source}_{field}')] for field in ('u100', 'v100'))
    speed = np.hypot(u, v)
    if not conditioned:
        return speed[..., None]
    lead = np.broadcast_to(np.linspace(-1, 1, 96), speed.shape)
    return np.stack((speed, u, v, lead, lead * speed), axis=-1)


def map_wind(train, target, forecast, conditioned):
    mask = np.isfinite(target) & (target >= 0) & np.isfinite(train).all(axis=-1)
    values = target[mask]
    inputs = train[mask].astype(float)
    if len(values) < 2:
        raise ValueError('Insufficient training wind observations for a site.')
    center, scale = inputs.mean(axis=0), inputs.std(axis=0).clip(1e-6)
    x = (inputs - center) / scale
    y = values - values.mean()
    ridge = RIDGE_ALPHA if conditioned else 1e-8
    coef = np.linalg.solve(x.T @ x / len(x) + ridge * np.eye(x.shape[1]), x.T @ y / len(x))
    result = ((forecast - center) / scale) @ coef + values.mean()
    metadata = {'valid_wind_rows': len(values), 'coefficients': coef.tolist(), 'center': center.tolist(),
                'scale': scale.tolist(), 'intercept': float(values.mean()), 'ridge': ridge}
    return np.where(np.isfinite(result), np.maximum(result, 0), np.nan), metadata


def observed_curve(train_wind, train_power, valid):
    mask = valid & np.isfinite(train_wind) & (train_wind >= 0) & np.isfinite(train_power)
    if not mask.any():
        raise ValueError('Missing joint wind and valid power training points.')
    support, bins = np.unique(train_wind[mask], return_inverse=True)
    counts = np.bincount(bins)
    power = np.bincount(bins, weights=train_power[mask]) / counts
    values = monotone(power, counts)
    return support, values, float(np.median(train_wind[mask]))


def wind_arms(args, train, days, roster, selected=None):
    weather, columns, _ = archived_issue_weather(args.workdir, args.data_root, [*train, *days], roster, print)
    weather = weather.reshape(len(train) + len(days), len(roster), 96, len(columns))
    truth = wind_labels(args.data_root, train, roster)
    usable = truth.power_limit.eq(1) & np.isfinite(truth.power)
    adjacent = usable & usable.shift(fill_value=False) & truth.point_index.gt(1)
    ramp_threshold = float(truth.power.diff().abs().loc[adjacent].quantile(.90))
    shape = (len(train), len(roster), 96)
    wind = truth.wind.to_numpy(dtype=float).reshape(shape)
    power = truth.power.to_numpy(dtype=float).reshape(shape)
    valid = truth.power_limit.eq(1).to_numpy().reshape(shape)
    curves = [observed_curve(wind[:, site], power[:, site], valid[:, site]) for site in range(len(roster))]
    winds, fits = {}, {}
    for condition, suffix in ((False, 'linear'), (True, 'conditioned')):
        if selected and not selected.endswith(suffix):
            continue
        streams = []
        for source in SOURCES:
            name = f'{source}_{suffix}'
            if selected and selected not in (name, f'three_source_{suffix}'):
                continue
            features = wind_features(weather, columns, source, condition)
            estimates, meta = [], {}
            for index, site in enumerate(roster):
                estimated, fit = map_wind(features[:len(train), index], wind[:, index],
                                         features[len(train):, index], condition)
                estimates.append(estimated)
                meta[site] = fit
            winds[name] = np.stack(estimates, axis=1)
            fits[name] = meta
            streams.append(winds[name])
        if len(streams) == len(SOURCES):
            stack = np.stack(streams)
            counts = np.isfinite(stack).sum(axis=0)
            winds[f'three_source_{suffix}'] = np.divide(np.nansum(stack, axis=0), counts,
                                                       out=np.full(stack.shape[1:], np.nan), where=counts > 0)
    def power_for(wind):
        values = []
        for i, (support, curve, fallback) in enumerate(curves):
            field = wind[:, i]
            values.append(np.interp(np.where(np.isfinite(field), field, fallback), support, curve))
        return np.clip(np.stack(values, axis=1).ravel(), 0, MAX_PREDICTION['T1'])
    keys = forecast_keys(days, roster)
    fits['power_curves'] = {site: {'wind': support.tolist(), 'power': curve.tolist(), 'fallback': fallback}
                            for site, (support, curve, fallback) in zip(roster, curves)}
    return {name: keys.assign(prediction=power_for(wind)) for name, wind in winds.items()}, winds, power_for, fits, ramp_threshold


def read_arm(directory, name, keys, track, lock='predictions_frozen.json'):
    record = read_json(directory / lock)
    return read_frozen_forecast(directory / f'{name}.csv', record['hashes'][name], keys, track)


def baseline_a(args, keys):
    directory = args.prior / 'T1' / 'A_forecasts'
    lock = sha256(directory / 'predictions_frozen.json')
    if read_json(args.prior / 'T1/a_score.json')['forecasts_sha256'] != lock:
        raise ValueError('The frozen T1 A predictions do not match the source report.')
    tree = read_arm(directory, 'baseline', keys, 'T1')
    candidate = read_arm(directory, 'daily_rmse_blend', keys, 'T1')
    return keep_sites(tree, candidate, [STATION]), lock


def baseline_b(args, keys, seed):
    directory = args.prior / 'T1' / 'B'
    path = directory / 'predictions_frozen.json'
    lock = sha256(path)
    scores = read_json(directory / 'b_score.json')
    if scores['prediction_lock_sha256'] != lock or scores['accepted_routes'] != {STATION: 'daily_rmse_blend'}:
        raise ValueError('T1 B does not match the accepted KNTBH4 baseline.')
    pair = directory / str(seed) / 'pair'
    if read_json(path)[str(seed)] != sha256(pair / 'predictions_frozen.json'):
        raise ValueError('T1 B seed lock changed.')
    return keep_sites(read_arm(pair, 'baseline', keys, 'T1'), read_arm(pair, 'candidate', keys, 'T1'), [STATION]), lock


def wind_diagnostics(observed, arms, winds, power_for, roster, ramp_threshold):
    wind = observed.wind.to_numpy(dtype=float)
    power = observed.power.to_numpy(dtype=float)
    shape = (len(observed) // (len(roster) * 96), len(roster), 96)
    common = np.isfinite(wind) & (wind >= 0) & np.isfinite(power) & observed.power_limit.eq(1).to_numpy()
    for field in winds.values():
        common &= np.isfinite(field.ravel())
    keys = observed[POINT_KEYS]
    station_day = common.reshape(-1, 96).sum(axis=1) >= MIN_VALID_POINTS
    common &= np.repeat(station_day, 96)
    if not common.any():
        raise ValueError('No eligible shared wind/power comparison cohort.')
    measured = power_for(wind.reshape(shape))
    result = {'matched_station_days': int(station_day.sum()), 'matched_points': int(common.sum()), 'models': {}}
    error_vectors = {}
    for name, field in winds.items():
        error = field.ravel() - wind
        error_vectors[name] = error[common]
        weather_error = arms[name].prediction.to_numpy() - measured
        response_error = measured - power
        rows = {}
        for first in (1, 25, 49, 73):
            use = common & keys.point_index.between(first, first + 23).to_numpy()
            if use.any():
                rows[f'{first}-{first+23}'] = {'mae': float(np.abs(error[use]).mean()), 'points': int(use.sum())}
        by_station = {}
        for site in roster:
            use = common & keys.station_id.eq(site).to_numpy()
            if use.any():
                by_station[site] = {'points': int(use.sum()), 'bias': float(error[use].mean()),
                                    'mae': float(np.abs(error[use]).mean())}
        result['models'][name] = {
            'wind_label_scale_mae': float(np.abs(error[common]).mean()), 'lead_bins': rows, 'station': by_station,
            'power_mse_mw2': {'weather_component': float(np.square(weather_error[common]).mean()),
                              'response_component': float(np.square(response_error[common]).mean()),
                              'cross_term': float(2 * np.mean(weather_error[common] * response_error[common]))},
        }
    # Correlate all sources on the same retained observations.
    correlation = pd.DataFrame(error_vectors).corr()
    result['wind_error_correlation'] = correlation.astype(object).where(correlation.notna(), None).to_dict()
    restricted = observed.assign(power_limit=observed.power_limit.where(common, 0))
    result['oracle_scores'] = block_comparisons(
        restricted, {'baseline': arms['baseline'], 'observed_wind': keys.assign(prediction=measured)},
        roster, ('observed_wind',))
    result['power_shape'] = point_diagnostics(observed, arms, ramp_threshold)
    result['ramp_threshold_mw'] = ramp_threshold if np.isfinite(ramp_threshold) else None
    return result


def t1(args):
    train, days = chronological_dates(args.split, 'A')
    roster = load_roster(args.data_root)
    keys = forecast_keys(days, roster)
    output = args.out / 'T1'
    output.mkdir(exist_ok=True)
    base, source = baseline_a(args, keys)
    arms, wind, power_for, fits, threshold = timed(output, 'wind_models_A', lambda: wind_arms(args, train, days, roster))
    blends = {name + '_blend': blend(base, estimate, 0.20, 'T1') for name, estimate in arms.items()}
    arms = {'baseline': base, **arms, **blends}
    arms, locked = freeze_window(output / 'A', 'T1', keys, arms, {'source': source, 'fits': fits})
    truth = wind_labels(args.data_root, days, roster)
    report = block_comparisons(truth, arms, roster, tuple(name for name in arms if name != 'baseline'))
    scores = report['scores']
    deltas = {name: scores[name][PRIMARY_SCORE] - scores['baseline'][PRIMARY_SCORE] for name in blends}
    best = max(deltas, key=deltas.get)
    gate = {'candidate': best, 'promoted': deltas[best] > 0, 'delta': deltas[best], 'forecast_lock': locked}
    freeze_json(output / 'A_score.json', {'score': report, 'gate': gate, 'deltas': deltas})
    diagnostics = wind_diagnostics(truth, arms, wind, power_for, roster, threshold)
    write_json(output / 'wind_diagnostics.json', diagnostics)
    print(f'[T1 A] 8 路 80/20 配对分差 {deltas}，准入 {gate}', flush=True)
    summary = {'A_gain': deltas, 'A_selected': best, 'B_started': gate['promoted'],
               'matched_wind_station_days': diagnostics['matched_station_days'],
               'oracle_shared_cohort': {arm: row[PRIMARY_SCORE] for arm, row in diagnostics['oracle_scores']['scores'].items()},
               'wind_power_mse': {name: row['power_mse_mw2'] for name, row in diagnostics['models'].items()},
               'wind_mae': {name: row['wind_label_scale_mae'] for name, row in diagnostics['models'].items()}}
    if gate['promoted']:
        train, days = chronological_dates(args.split, 'B')
        b_keys = forecast_keys(days, roster)
        directory = output / 'B'
        directory.mkdir(exist_ok=True)
        name = best.removesuffix('_blend')
        b_arms, _, _, b_fits, _ = timed(directory, 'wind_models_B', lambda: wind_arms(args, train, days, roster, selected=name))
        fit_hash = freeze_json(directory / 'wind_model.json', b_fits)
        pairs, locks = {}, {}
        for seed in B_SEEDS:
            reference, stamp = baseline_b(args, b_keys, seed)
            frames = {'baseline': reference, 'candidate': blend(reference, b_arms[name], .20, 'T1')}
            pairs[seed], locks[str(seed)] = freeze_window(directory / str(seed), 'T1', b_keys, frames,
                                                        {'source': stamp, 'fit_sha256': fit_hash, 'gate': gate})
        digest = freeze_json(directory / 'predictions_frozen.json', locks)
        report = paired_report(directory, 'T1', labels(args.data_root, days, roster), roster, b_keys, pairs, digest)
        summary['B'] = {'ensemble_delta': report['ensemble']['delta'], 'acceptance': report['acceptance'],
                        'seed_deltas': {key: row['delta'] for key, row in report['pairs'].items()}}
    return summary


def pseudo_dates(args):
    first, a_days = chronological_dates(args.split, 'A')
    train, days = a_days[-60:-10], a_days[-10:]
    return [*first, *a_days[:-10]], train, days


def pseudo_sites(args, train, regular):
    observed = labels(args.data_root, train, regular)
    available = observed.power_limit.eq(1) & np.isfinite(observed.power)
    coverage = available.groupby([observed.station_id, observed.date]).sum().ge(MIN_VALID_POINTS)
    counts = coverage.groupby(level='station_id').sum()
    internal_stop = str(train[-block_early_stop_days(train)])
    usable = observed.loc[available]
    eligible = set(usable.loc[usable.date.lt(internal_stop)].station_id) & set(usable.loc[usable.date.ge(internal_stop)].station_id)
    ordered = sorted(eligible, key=lambda site: (int(counts[site]), site))
    if len(ordered) < 3:
        raise ValueError('Cold-start replay requires two eligible target stations and surviving donors.')
    return sorted((ordered[0], ordered[-1])), {site: int(counts[site]) for site in ordered}


def pseudo_network(args):
    donors, train, days = pseudo_dates(args)
    plan = read_json(args.out / 'plan.json')
    selected, source = plan['sites'], plan['donors']
    if (set(selected) & set(source) or plan['score_dates'] != list(map(str, days))
            or plan['target_train_dates'] != list(map(str, train)) or plan['donor_dates'] != list(map(str, donors))):
        raise ValueError('Pseudo target is in the donor cohort or dates differ.')
    keys = forecast_keys(days, selected)
    target = args.out / 'neural'
    if (target / 'predictions_frozen.json').is_file():
        return
    staged = Path(mkdtemp(dir=args.out, prefix='.neural-attempt-'))
    forecasts, metadata = fit_models(args, donors, source, selected, train, days, staged, seed=A_SEED, include_scratch=False)
    original = keys.assign(prediction=forecasts['frozen_context_ridge'].ravel())
    replay, alternatives, ridge = shared_head(args, staged / 'frozen_encoder.pt', metadata, keys, selected, train, days)
    if not np.allclose(original.prediction, replay.prediction, rtol=0, atol=5e-5):
        raise ValueError('The pseudo-target head differs after replaying its checkpoint.')
    freeze_window(staged, 'T3', keys, {'station_ridge': original, **alternatives},
                  {'plan_sha256': sha256(args.out / 'plan.json'), 'fit': metadata, 'ridge': ridge})
    staged.rename(target)


def cold_start(args):
    donors, train, days = pseudo_dates(args)
    regular, short = load_roster(args.data_root), roster_for(args.data_root, 'T3')
    sites, coverage = pseudo_sites(args, train, regular)
    donors_sites = [site for site in regular if site not in sites]
    directory = args.out / 'T3_pseudo'
    directory.mkdir(exist_ok=True)
    marker = directory / 'plan.json'
    config = args.config or args.companion_code / 'configs/default.json'
    setup = {'sites': sites, 'donors': donors_sites, 'coverage': coverage, 'seed': A_SEED, 'device': args.device,
             'target_train_dates': list(map(str, train)), 'score_dates': list(map(str, days)),
             'donor_dates': list(map(str, donors)), 'archive': archive_stamp(args.workdir, [*donors, *days]),
             'tree_config': sha256(config)}
    digest = freeze_json(marker, setup)
    keys = forecast_keys(days, sites)
    tree = directory / 'tree'
    if not (tree / 'fit.json').is_file():
        whole_roster, (indices, spec, names, build, params) = short_source(args, regular + short)
        indices = [indices[whole_roster.index(site)] for site in sites]
        # Use the cache's station ordering for the feature builder; the forecast reader aligns back to site order.
        ordered = [whole_roster[index] for index in sorted(indices)]
        staged = Path(mkdtemp(dir=directory, prefix='.tree-attempt-'))
        _, _, fitted = fit_short_forecasts(args, staged, train, days, ordered, ordered,
                                            (sorted(indices), spec, names, build, params), modes=('short_only',))
        write_json(staged / 'fit.json', {'plan_sha256': digest, 'fit': fitted})
        staged.rename(tree)
    stored = read_json(tree / 'fit.json')
    if stored['plan_sha256'] != digest:
        raise ValueError('Pseudo tree plan changed.')
    baseline = read_frozen_forecast(tree / 'short_only.csv', stored['fit']['predictions_sha256']['short_only'], keys, 'T3')
    neural = directory / 'neural'
    if not (neural / 'predictions_frozen.json').is_file():
        interpreter = select_mlp_runtime(args, require_weather=False)[0]
        call(interpreter, SCRIPT, '--pseudo-worker', '--out', directory, *worker_options(args))
    predictions = read_json(neural / 'predictions_frozen.json')
    if predictions['sources']['plan_sha256'] != digest:
        raise ValueError('Pseudo network plan changed.')
    arms = {'short_only': baseline}
    for name, value in predictions['hashes'].items():
        forecast = read_frozen_forecast(neural / f'{name}.csv', value, keys, 'T3')
        arms[name] = blend(baseline, forecast, .25, 'T3')
    arms, lock = freeze_window(directory / 'forecasts', 'T3', keys, arms,
                               {'tree': sha256(tree/'fit.json'), 'neural': sha256(neural/'predictions_frozen.json')})
    truth = labels(args.data_root, days, sites)
    scores = {name: station_scores(truth, forecast) for name, forecast in arms.items()}
    deltas = {name: {site: row['score'] - scores['short_only'][site]['score'] for site, row in rows.items()}
              for name, rows in scores.items() if name != 'short_only'}
    result = {'sites': sites, 'site_training_coverage': {site: coverage[site] for site in sites}, 'scores': scores,
              'station_deltas': deltas, 'forecast_lock': lock, 'scope': 'diagnostic_pseudo_targets'}
    freeze_json(directory / 'score.json', result)
    print(f'[T3 模拟短历史] 两站 {sites}，分差 {deltas}', flush=True)
    return result


def execute(args):
    start = monotonic()
    paths = ('data_root', 'workdir', 'companion_code', 'prior')
    sources = {name: str(getattr(args, name).resolve()) for name in paths}
    code = (SCRIPT.name, 'screen_short_transfer.py', 'run_t1_t3_batch.py', 'run_companion_warm_t1.py',
            'warm_shared_power.py', 'same_issue_weather.py', 'companion_power_curve.py', 'score_forecasts.py')
    freeze_json(args.out / 'setup.json', {'source_paths': sources, 'device': args.device, 'threads': args.threads,
                                        'split_sha256': sha256(args.split),
                                        'code_sha256': {name: sha256(SCRIPT.with_name(name)) for name in code}})
    summary = {}
    for index, (track, operation) in enumerate((('T1_wind_attribution', t1), ('T3_pseudo_cold_start', cold_start))):
        show_progress(args.out, '风功率诊断与迁移', index, 2, track)
        try:
            summary[track] = {'status': 'complete', 'result': operation(args)}
        except Exception as error:
            event(args.out, track, status='failed', detail=str(error))
            summary[track] = {'status': 'failed', 'detail': str(error)}
            print(f'[{track}] {error}', flush=True)
        write_json(args.out / 'summary.json', summary)
    event(args.out, 'finished', seconds=monotonic() - start)
    show_progress(args.out, '风功率诊断与迁移', 2, 2, '结果与进度已写入文件')
    print(f'摘要 {args.out / "summary.json"}', flush=True)
    return int(any(item['status'] != 'complete' for item in summary.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RUNS / 'wind_attribution_gpu_v1')
    parser.add_argument('--prior', type=Path, default=RUNS / 't1_t3_fast_gpu_v1')
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--mlp-run', type=Path, default=RUNS / 'warm_shared_power_v1')
    parser.add_argument('--mlp-python', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', choices=range(1, 5), type=int, default=4)
    parser.add_argument('--pseudo-worker', action='store_true')
    args = parser.parse_args()
    args.mlp_weather_python = None
    args.out.mkdir(parents=True, exist_ok=True)
    if args.pseudo_worker:
        pseudo_network(args)
    else:
        with (args.out / '.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(execute(args))


if __name__ == '__main__':
    main()
