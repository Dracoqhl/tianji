#!/usr/bin/env python3
"""Independent T2 and T3 GPU comparisons on their registered validation windows."""

import argparse
from datetime import date, timedelta
import fcntl
import hashlib
import json
from pathlib import Path
from time import monotonic
import warnings

import numpy as np
import pandas as pd

from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import event, read_json
from independent_t2 import diagnostics, fit_weights, load_daily_targets
from run_companion_warm_t1 import A_SEED, SPLIT_PATH, block_early_stop_days, cache_paths, cache_version
from run_companion_warm_t1 import check_training_sample, date_roles, forecast_keys, sha256, training_block_dates
from run_companion_warm_t1 import weather_rows, write_json
from score_forecasts import CAPACITY_MW, DAY_KEYS, MAX_PREDICTION, STATION_COUNTS, score_track
from screen_companion_t1_stations import BLOCKS, frozen_pooled_forecast, frozen_pooled_origin, show_progress, station_scores
from smoke_tree_backend import fit_station_factors
from tree_model import best_iteration, check_backend, fitted_device, predict_tree, save_tree_model, train_tree
from warm_shared_power import labels, load_roster


ENERGY_MODELS = ('log_mse', 'relative_mse')
SHORT_MODELS = ('short_only', 'pooled', 'pooled_equal_group')
T3_START = date(2026, 1, 29)
T3_WINDOWS = (
    ('inner_feb', date(2026, 2, 18), date(2026, 2, 24)),
    ('inner_mar', date(2026, 3, 5), date(2026, 3, 11)),
    ('fixed_50_10', date(2026, 3, 20), date(2026, 3, 29)),
)
SOURCES = ('tianji', 'ec', 'aifs')


def t3_windows():
    for name, first, last in T3_WINDOWS:
        training = [T3_START + timedelta(days=i) for i in range((first - T3_START).days)]
        forecast = [first + timedelta(days=i) for i in range((last - first).days + 1)]
        yield name, training, forecast


def t2_windows(split):
    for first, last in BLOCKS:
        training, forecast = training_block_dates(split, first, last)
        yield first.isoformat(), training, forecast
    training, forecast = date_roles(split, 'A')
    yield 'A', training, forecast


def weather_day_targets(root, days):
    return [(day.isoformat(), root / 'train' / f'{day:%Y%m%d}_0000') for day in days]


def roster_for(root, track):
    path = root / 'submission' / f'{track}_dataset' / 'submission_template.csv'
    values = sorted(pd.read_csv(path, dtype={'station_id': str}, usecols=['station_id']).station_id.dropna().unique())
    if len(values) != STATION_COUNTS[track]:
        raise ValueError(f'{track} template has an unexpected station roster.')
    return values


def save_csv(path, frame):
    temp = path.with_suffix('.tmp')
    frame.to_csv(temp, index=False)
    temp.replace(path)
    return sha256(path)


def daily_cache_inputs(paths, roster, spec):
    """Read the cached exogenous arrays only, using the same ex-ante 96 points."""
    names = ['station_idx']
    fields = ('speed', 'speed_cube', 'wpd')
    statistics = ('mean', 'sd', 'q10', 'q50', 'q90')
    names.extend(f'{source}_{field}_{stat}' for source in SOURCES for field in fields for stat in statistics)
    names.extend(name for group in ('static_names', 'sat_names', 'obs_names') for name in spec[group])
    lookup = {name: i for i, name in enumerate(spec['nwp_names'])}
    frames = []
    for i, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as archive:
            file_roster = archive['station_ids'].tolist()
            indices = [file_roster.index(station) for station in roster]
            if archive['date_code'].tolist() != [path.stem] or len(file_roster) != len(set(file_roster)):
                raise ValueError(f'{path}: unexpected date or station keys.')
            nwp = archive['nwp_point'][indices].astype(np.float64)
            if nwp.shape != (len(roster), 96, len(lookup)):
                raise ValueError(f'{path}: unexpected companion NWP schema.')
            statistics_values = [np.arange(len(roster))[:, None]]
            for source in SOURCES:
                speed = np.hypot(nwp[..., lookup[f'{source}_u100']], nwp[..., lookup[f'{source}_v100']])
                wpd = nwp[..., lookup[f'{source}_wpd']]
                for field in (speed, speed**3, wpd):
                    field = np.where(np.isfinite(field), field, np.nan)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        quantiles = np.nanquantile(field, [0.1, 0.5, 0.9], axis=1).T
                        statistics_values.append(np.column_stack(
                            (np.nanmean(field, axis=1), np.nanstd(field, axis=1), quantiles),
                        ))
            statistics_values.extend(archive[group][indices] for group in ('static', 'sat', 'obs'))
            data = np.concatenate(statistics_values, axis=1).astype(np.float32)
        if data.shape != (len(roster), len(names)) or np.isinf(data).any():
            raise ValueError(f'{path}: invalid daily features.')
        date_value = date.fromisoformat(f'{path.stem[:4]}-{path.stem[4:6]}-{path.stem[6:8]}').isoformat()
        frame = pd.DataFrame(data, columns=names).assign(date=date_value, station_id=roster)
        frames.append(frame)
        if i == 1 or i % 50 == 0 or i == len(paths):
            print(f'[T2] 日特征 {i}/{len(paths)}', flush=True)
    return pd.concat(frames, ignore_index=True), names


def energy_features(args, roster, days, spec_path):
    paths = cache_paths(args.workdir, days)
    original = cache_version(paths)
    code_hash = sha256(Path(__file__))
    spec_hash = sha256(spec_path)
    fingerprint = hashlib.sha256(json.dumps([original, code_hash, spec_hash, roster]).encode()).hexdigest()
    directory = args.out / 'energy_features'
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f'{fingerprint}.csv'
    meta_path = directory / f'{fingerprint}.json'
    if meta_path.is_file():
        meta = read_json(meta_path)
        if sha256(csv_path) != meta['sha256']:
            raise ValueError('The saved daily feature cache hash changed.')
        frame = pd.read_csv(csv_path, dtype={'station_id': str, 'date': str}, float_precision='round_trip')
        if list(frame) != [*meta['columns'], 'date', 'station_id']:
            raise ValueError('The saved daily feature column schema changed.')
        return frame, meta
    frame, names = daily_cache_inputs(paths, roster, read_json(spec_path))
    if cache_version(paths) != original:
        raise ValueError('The companion weather cache changed during daily feature extraction.')
    meta = {'version': fingerprint, 'weather_cache_version': original, 'feature_spec_sha256': spec_hash,
            'sha256': save_csv(csv_path, frame), 'columns': names}
    write_json(meta_path, meta)
    return frame, meta


def raw_energy(model, rows, features, mode):
    raw = predict_tree(model, rows[features].to_numpy(dtype=np.float32), backend='xgboost').astype(float)
    with np.errstate(over='raise', invalid='raise'):
        values = np.exp(raw) if mode == 'log_mse' else np.maximum(raw, 0.0)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite T2 energy.')
    return values


def fit_energy(args, features, training, allowed_days, forecast, mode, directory):
    calibration_days = max(1, len(allowed_days) // 5)
    before_cal = allowed_days[:-calibration_days]
    cal_start = allowed_days[-calibration_days].isoformat()
    stop_start = before_cal[-block_early_stop_days(before_cal)].isoformat()
    earlier = training.loc[training.date.lt(stop_start)]
    stop = training.loc[training.date.ge(stop_start) & training.date.lt(cal_start)]
    calibration = training.loc[training.date.ge(cal_start)]
    pre_cal = training.loc[training.date.lt(cal_start)]
    if any(frame.empty for frame in (earlier, stop, calibration, pre_cal, training)):
        raise ValueError('A T2 fitting, early stopping or calibration partition is empty.')
    params = {'objective':'regression','n_estimators':500,'num_leaves':15,'min_child_samples':40,
              'learning_rate':0.04,'random_state':A_SEED,'n_jobs':args.threads}
    actual = stop['energy'].to_numpy()
    groups = pd.factorize(stop['station_id'])[0]
    counts = np.bincount(groups)

    def score(raw, _):
        with np.errstate(over='raise', invalid='raise'):
            energy = np.exp(raw.astype(float)) if mode == 'log_mse' else raw
        err = (np.clip(energy,0,MAX_PREDICTION['T2'])/actual - 1)**2
        return 't2_score', float(100 * (1 - np.sqrt(np.bincount(groups, weights=err)/counts).mean()))

    def target(rows):
        values = rows['energy'].to_numpy(dtype=float)
        return np.log(values) if mode == 'log_mse' else values

    def fit(rows, config, **validation):
        return train_tree(rows[features].to_numpy(dtype=np.float32), target(rows), features, config,
                          categorical_features=['station_idx'], backend='xgboost', device=args.device,
                          early_stopping_rounds=35, sample_weight=fit_weights(rows, mode), **validation)

    start = monotonic()
    selection = fit(earlier, params, X_valid=stop[features].to_numpy(dtype=np.float32), y_valid=target(stop),
                    evaluation=score, maximize=True)
    best = best_iteration(selection, backend='xgboost', fallback=params['n_estimators'])
    del selection
    tuned = dict(params, n_estimators=best)
    calibration_model = fit(pre_cal, tuned)
    cal_prediction = raw_energy(calibration_model, calibration, features, mode)
    factors = fit_station_factors(calibration['station_id'], cal_prediction, calibration['energy'])
    del calibration_model
    final = fit(training, tuned)
    estimates = raw_energy(final, forecast, features, mode)
    multiplier = forecast['station_id'].map(factors).fillna(1.0).to_numpy(dtype=float)
    cap = MAX_PREDICTION['T2']
    directory.mkdir(parents=True, exist_ok=True)
    save_tree_model(final, directory / f'{mode}.joblib')
    receipt = {'best_iteration': best, 'calibration_start': cal_start, 'inner_stop_start': stop_start,
               'fit_rows': len(training), 'calibration_rows': len(calibration),
               'station_factors': factors, 'effective_device': fitted_device(final.booster),
               'fit_and_prediction_seconds': monotonic()-start}
    return np.clip(estimates,0,cap), np.clip(estimates*multiplier,0,cap), receipt


def curve_integral(args, name, train_days, days, roster, origin):
    actual, forecast, digest = frozen_pooled_forecast(
        args.hik_run, origin, name, train_days, days, forecast_keys(days, roster), device=args.device,
    )
    if actual['data_root'] != str(args.data_root.resolve()):
        raise ValueError('The T1 curve uses a different data root.')
    energy = forecast.groupby(DAY_KEYS, sort=False, as_index=False).prediction.sum()
    energy['energy'] = (energy.pop('prediction') / 4.0).clip(0, MAX_PREDICTION['T2'])
    return energy, digest


def energy_arm_report(truth, eligible, forecasts, roster):
    reports = {}
    for name, prediction in forecasts.items():
        score = score_track('T2', truth, prediction, roster, allow_unscored_stations=True)
        detail = diagnostics(prediction, eligible)
        score['by_station'] = {station: {**values, 'score': 100*(1-values['relative_rmse'])}
                               for station, values in detail['by_station'].items()}
        score['by_energy_band'] = detail['by_actual_energy_band_mwh']
        reports[name] = score
    return reports


def energy_window(args, name, train_days, days, roster, weather, features, origin):
    _, eligible = load_daily_targets(weather_day_targets(args.data_root, train_days), roster)
    training = weather.merge(eligible, on=DAY_KEYS, validate='one_to_one')
    if len(training) != len(eligible):
        raise ValueError('Daily forecasts do not cover the eligible T2 training rows.')
    forecast = weather.loc[weather.date.isin([day.isoformat() for day in days])]
    directory = args.out / name
    directory.mkdir(parents=True, exist_ok=True)
    curve, curve_receipt = curve_integral(args, name, train_days, days, roster, origin)
    predictions = {'t1_integral':curve}
    fits = {}
    for mode in ENERGY_MODELS:
        print(f'[T2 {name}] {mode}', flush=True)
        raw, calibrated, fits[mode] = fit_energy(args, features, training, train_days, forecast, mode, directory)
        predictions[mode] = forecast[DAY_KEYS].assign(energy=raw)
        predictions[f'{mode}_calibrated'] = forecast[DAY_KEYS].assign(energy=calibrated)
    hashes = {arm: save_csv(directory/f'{arm}.csv', values) for arm, values in predictions.items()}
    truth, eligible_holdout = load_daily_targets(weather_day_targets(args.data_root, days), roster)
    scores = energy_arm_report(truth, eligible_holdout, predictions, roster)
    return {'train_days':len(train_days),'score_days':len(days),'train_rows':len(training),
            'predictions_sha256':hashes,'curve_training_receipt_sha256':curve_receipt,'fit':fits,'scores':scores}


def pooled_weights(is_short):
    weights = np.ones(len(is_short), dtype=float)
    for group in (is_short, ~is_short):
        count = int(group.sum())
        if count == 0:
            raise ValueError('Both source and target groups need usable T3 training points.')
        weights[group] = len(weights) / (2 * count)
    return weights


def fit_short(args, names, matrix, target, short_mask, early_mask, forecast, mode, params, directory):
    use = short_mask if mode == 'short_only' else np.ones(len(target), dtype=bool)
    selection, stopping = use & ~early_mask, short_mask & early_mask
    if not selection.any() or not stopping.any():
        raise ValueError('T3 train/early stop mask has no usable rows.')

    def fit(mask, config, **validation):
        weights = pooled_weights(short_mask[mask]) if mode == 'pooled_equal_group' else None
        return train_tree(matrix[mask], target[mask], names, config, categorical_features=['station_idx'],
                          early_stopping_rounds=int(params.get('early_stopping_rounds',100)),
                          sample_weight=weights, backend='xgboost', device=args.device, **validation)

    start = monotonic()
    selected = fit(selection,params,X_valid=matrix[stopping],y_valid=target[stopping])
    best = best_iteration(selected,backend='xgboost',fallback=params['n_estimators'])
    del selected
    final = fit(use,dict(params,n_estimators=best))
    power = predict_tree(final,forecast,backend='xgboost').astype(float)*CAPACITY_MW
    if not np.isfinite(power).all():
        raise ValueError('Nonfinite T3 power forecast.')
    save_tree_model(final,directory/f'{mode}.joblib')
    return np.clip(power,0,MAX_PREDICTION['T3']), {
        'best_iteration':best,'fit_seconds':monotonic()-start,'effective_device':fitted_device(final.booster),
        'fit_points':int(use.sum()),'target_points':int((use & short_mask).sum()),'early_stop_points':int(stopping.sum())}


def short_window(args, name, train_days, days, roster, short_roster, source):
    stations, spec, names, build_rows, params = source
    train_paths = cache_paths(args.workdir, train_days)
    test_paths = cache_paths(args.workdir, days)
    cache = {'train':cache_version(train_paths),'forecast':cache_version(test_paths)}
    context, sample, dtype = check_training_sample(
        train_paths, train_days, stations, roster, spec, names, build_rows, args.data_root,
    )
    history = labels(args.data_root, train_days, roster)
    valid = history.power_limit.eq(1).to_numpy() & np.isfinite(history.power.to_numpy(dtype=float))
    matrix = weather_rows(train_paths, stations, roster, spec, names, context, dtype)[valid]
    target = history.loc[valid,'power'].to_numpy(dtype=float) / CAPACITY_MW
    short_mask = history.loc[valid,'station_id'].isin(short_roster).to_numpy()
    start = train_days[-block_early_stop_days(train_days)].isoformat()
    early = history.loc[valid,'date'].ge(start).to_numpy()
    kept = [i for i, station in enumerate(roster) if station in short_roster]
    full_forecast = weather_rows(test_paths, stations, roster, spec, names, context,dtype)
    forecast = full_forecast.reshape(len(days),len(roster),96,len(names))[:,kept].reshape(-1,len(names))
    ordered_short = [roster[i] for i in kept]
    keys = forecast_keys(days, ordered_short)
    directory = args.out/name
    directory.mkdir(parents=True, exist_ok=True)
    predictions, fit = {}, {}
    for mode in SHORT_MODELS:
        print(f'[T3 {name}] {mode}',flush=True)
        values, fit[mode] = fit_short(args,names,matrix,target,short_mask,early,forecast,mode,params,directory)
        predictions[mode] = keys.assign(prediction=values)
    hashes = {mode:save_csv(directory/f'{mode}.csv', frame) for mode,frame in predictions.items()}
    truth = labels(args.data_root,days,ordered_short)
    reports = {mode: dict(score_track('T3',truth,frame,ordered_short,allow_unscored_stations=True),
                          by_station=station_scores(truth,frame)) for mode,frame in predictions.items()}
    return {'train_days':len(train_days),'score_days':len(days),'inner_stop_start':start,'cache':cache,
            'predictions_sha256':hashes,'fit':fit,'scores':reports,'sample_keys':sample}


def short_source(args, expected_roster):
    import sys
    sys.path.insert(0,str(args.companion_code/'src'))
    from tianji.config import load_config
    from tianji.constants import ALL_STATIONS
    from tianji.data.dataset import build_gbdt_rows, gbdt_feature_names, load_feature_spec

    roster = list(ALL_STATIONS)
    if len(roster) != len(expected_roster) or set(roster) != set(expected_roster):
        raise ValueError('T3 requires the complete 18-station roster.')
    config = args.config or args.companion_code/'configs/default.json'
    params = dict(load_config(str(config))['gbdt'], objective='huber', n_jobs=args.threads, random_state=A_SEED)
    spec = load_feature_spec(args.workdir,'train')
    return roster, (list(range(len(roster))),spec,list(gbdt_feature_names(spec)),build_gbdt_rows,params)


def block_delta_report(windows, reference):
    return {
        name: {arm:score[PRIMARY_SCORE] - result['scores'][reference][PRIMARY_SCORE]
               for arm,score in result['scores'].items()}
        for name,result in windows.items()
    }


def execute(args, settings):
    if args.track == 'T2':
        windows = list(t2_windows(args.split))
        roster = load_roster(args.data_root)
        origin = frozen_pooled_origin(args.hik_run)
        if {key:origin[key] for key in ('freeze_sha256','a_sha256')} != settings['hik_source']:
            raise ValueError('The T1 curve source changed during setup.')
        days = sorted({day for _,train,forecast in windows for day in [*train,*forecast]})
        weather, meta = energy_features(args,roster,days,args.workdir/'stats/feature_spec_train.json')
        def evaluate(name, train, forecast):
            return energy_window(args,name,train,forecast,roster,weather,meta['columns'],origin)
        reference = 'log_mse_calibrated'
        metadata = {'features':meta}
    else:
        windows = list(t3_windows())
        config_roster = pd.read_csv(args.data_root/'static/power_stations.csv', dtype={'station_id':str})['station_id'].tolist()
        short_roster = roster_for(args.data_root,'T3')
        regular = load_roster(args.data_root)
        if len(config_roster) != 18 or set(config_roster) != set(regular) | set(short_roster):
            raise ValueError('T3 weather roster must include 12 regular and six short-history stations.')
        companion_roster, source = short_source(args,config_roster)
        def evaluate(name, train, forecast):
            return short_window(args,name,train,forecast,companion_roster,short_roster,source)
        reference = 'short_only'
        metadata = {'feature_names':source[2]}
    reports = {}
    show_progress(args.out,args.track,0,len(windows),'开始')
    for index,(name, train, forecast) in enumerate(windows,1):
        report_path = args.out/name/'comparison.json'
        if report_path.is_file():
            report = read_json(report_path)
            for arm,digest in report['predictions_sha256'].items():
                if sha256(report_path.parent/f'{arm}.csv') != digest:
                    raise ValueError(f'{name}/{arm}: the completed forecast was changed.')
        else:
            report = evaluate(name,train,forecast)
            report['window'] = [day.isoformat() for day in forecast]
            write_json(report_path,report)
        reports[name] = report
        show_progress(args.out,args.track,index,len(windows),f'{name} 已计分')
    deltas = block_delta_report(reports,reference)
    report = {'track':args.track,'settings':settings,'windows':reports,'reference':reference,
              'score_delta_by_window':deltas,**metadata}
    write_json(args.out/'results.json',report)
    for name, result in reports.items():
        values = ' '.join(f'{arm}={score[PRIMARY_SCORE]:.5f}' for arm,score in result['scores'].items())
        print(f'[{args.track} {name}] {values}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('track', choices=('T2','T3'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--companion-code', type=Path, default=Path('/workspace/cpy/code'))
    parser.add_argument('--workdir', type=Path, default=Path('/workspace/cpy/code/workdir'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--hik-run', type=Path, default=Path('/workspace/qhl/runs/t1_champion_st_tree_gpu_v1'))
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error('Use 1 to 4 CPU threads.')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'.stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        event(args.out,args.track,status='started',backend='xgboost',device=args.device)
        try:
            spec = args.workdir/'stats/feature_spec_train.json'
            config = args.config or args.companion_code/'configs/default.json'
            settings = {'backend':'xgboost','backend_version':check_backend('xgboost',args.device),'device':args.device,
                        'threads':args.threads,'seed':A_SEED,'data_root':str(args.data_root.resolve()),
                        'workdir':str(args.workdir.resolve()),'split_sha256':sha256(args.split),
                        'feature_spec_sha256':sha256(spec),'companion_config_sha256':sha256(config),
                        'implementation_sha256':sha256(Path(__file__))}
            if args.track == 'T2':
                origin = frozen_pooled_origin(args.hik_run)
                settings['hik_source'] = {key:origin[key] for key in ('freeze_sha256','a_sha256')}
            setup_path = args.out/'setup.json'
            if setup_path.is_file() and read_json(setup_path) != settings:
                raise ValueError('Track configuration changed. Use a new output path.')
            if not setup_path.is_file():
                write_json(setup_path,settings)
            execute(args,settings)
        except (Exception,KeyboardInterrupt) as error:
            event(args.out,args.track,status='failed',detail=str(error))
            raise
        event(args.out,args.track,status='complete')


if __name__ == '__main__':
    main()
