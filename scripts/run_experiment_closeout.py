#!/usr/bin/env python3
"""Complete the old source-power run and launch qualified T1 data candidates."""
import argparse
import fcntl
from pathlib import Path
from time import monotonic
import traceback
from types import SimpleNamespace

from closeout_weather import (SAT_ARM, SOURCE_ARM, SatelliteFeatures, sat_screen, source_screen, source_weather)
from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import read_json
from run_companion_warm_t1 import B_SEEDS, SPLIT_PATH, chronological_dates, forecast_keys, sha256, write_json
from run_forecast_blends import freeze_json
from run_fusion_mechanisms import stage
from run_intraday_residuals import t1_v4_reference
from run_phase_anchor import T1_ARMS as PHASE_ARMS, historical_reference, phase_forecasts
from run_power_hurdle import t1_predict
from run_pressure_profile import summarize_b
from run_strict_warm_replay import replay_dates
from run_t1_issue_wind_error import pair_for
import run_t1_source_power as source_power
from run_wind_attribution import read_arm
from screen_companion_t1_stations import show_progress
from warm_shared_power import labels, load_roster


RUNS = Path('/workspace/qhl/runs')
SOFT_MIN_POINTS = 100


def verify_panel(folder, digest, keys, expected):
    path = folder / 'predictions_frozen.json'
    if sha256(path) != digest:
        raise ValueError(f'{path}: source-power forecast receipt hash differs.')
    meta = read_json(path)
    if set(meta['hashes']) != set(expected):
        raise ValueError(f'{path}: source-power forecast arms differ from the registered study.')
    for name in meta['hashes']:
        read_arm(folder,name,keys,'T1')
    return meta


def recover_source(args):
    folder = args.out
    a_path = folder / 'A/score.json'
    marker = folder / 'summary.json'
    if not a_path.is_file():
        return None
    plan = read_json(folder / 'plan.json')
    expected = {'script':sha256(Path(source_power.__file__)),'split':sha256(args.split),'threads':args.threads,
                'device':args.device,'seeds':list(B_SEEDS),'fraction':source_power.FRACTION,
                'ridge_mw2':source_power.RIDGE_MW2,'paths':{n:str(getattr(args,n).resolve()) for n in
                    ('data_root','wind_run','power_run','workdir','companion_code')}}
    if plan != expected:
        raise ValueError('The old source-power receipt has a different registered plan.')
    roster = load_roster(args.data_root)
    _, a_dates = chronological_dates(args.split,'A')
    a = read_json(a_path)
    gate = a['gate']
    expected_arms = ('baseline', *source_power.SOURCES, source_power.COMPACT, source_power.EQUAL,
                     source_power.SHRUNK, source_power.CONTROL)
    verified_a = verify_panel(folder/'A/forecasts',gate['prediction_lock'],forecast_keys(a_dates,roster),expected_arms)
    weights = folder/'A/weights.json'
    if sha256(weights) != verified_a['sources']['weights']:
        raise ValueError('The old source-power A weights changed.')
    summary = {'status':'complete','A':{'scores':{arm:row[PRIMARY_SCORE] for arm,row in a['comparison']['scores'].items()},
                                       'gate':gate,'weights':read_json(weights)['weights']}}
    if gate['promoted']:
        b = folder/'B/b_score.json'
        if not b.is_file():
            return None
        report = read_json(b)
        ledger = folder/'B/predictions_frozen.json'
        if report['prediction_lock_sha256'] != sha256(ledger):
            raise ValueError('The old source-power B score points to a different prediction ledger.')
        locks = read_json(ledger)
        _, b_dates = chronological_dates(args.split,'B')
        keys = forecast_keys(b_dates,roster)
        weights_b = weight_lock = None
        if gate['chosen'] == source_power.SHRUNK:
            weight_file = folder/'B/weights.json'
            weights_b, weight_lock = read_json(weight_file)['weights'], sha256(weight_file)
        for seed in B_SEEDS:
            meta = verify_panel(folder/'B'/str(seed)/'pair',locks[str(seed)],keys,('baseline','candidate'))
            if meta['sources']['gate'] != gate or meta['sources']['weights'] != weight_lock:
                raise ValueError('The old source-power B predictions use a different A gate or source weights.')
        summary['B'] = {**summarize_b(report),'weights':weights_b}
    if marker.is_file():
        existing = read_json(marker)
        if existing['status'] == 'complete' and existing != summary:
            raise ValueError('The old source-power summary does not match the verified detailed scores.')
    return summary


def source_reconciliation(args, *, allow_fit):
    legacy = SimpleNamespace(**{**vars(args),'out':args.source_run})
    legacy.out.mkdir(parents=True,exist_ok=True)
    with (legacy.out/'.run.lock').open('a') as file:
        fcntl.flock(file,fcntl.LOCK_EX | fcntl.LOCK_NB)
        recovered = recover_source(legacy)
        if recovered is None:
            if not allow_fit:
                return {'status':'waiting_for_conditional_forecasts','source':str(legacy.out/'summary.json')}
            source_power.execute(legacy)
            recovered = recover_source(legacy)
            action = 'fit_or_resume'
        else:
            marker = legacy.out/'summary.json'
            complete = marker.is_file() and read_json(marker)['status'] == 'complete'
            action = 'verified_existing' if complete else 'recovered_from_details'
        if recovered is None:
            raise ValueError('The comparator completed without verifiable predictions.')
        if action != 'verified_existing':
            write_json(legacy.out/'summary.json',recovered)
    return {'status':'complete','action':action,'scope':'historical_T1_v2_control',
            'source':str(legacy.out/'summary.json'),'result':recovered}


def check_prerequisites(args):
    obs = args.obs_run/'summary.json'
    quality = args.audit_run/'batch_overview.json'
    results = {}
    if obs.is_file():
        data = read_json(obs)
        accepted = [track for track in ('T1','T3') if data[track].get('B',{}).get('acceptance',{}).get('accepted')]
        results['obs_conditioning'] = {'status':'prerequisite_supported' if accepted else 'screened_out',
                                       'positive_tracks':accepted,'source_sha256':sha256(obs)}
    else:
        results['obs_conditioning'] = {'status':'prerequisite_report_missing','source':str(obs)}
    if quality.is_file():
        data = read_json(quality)
        counts = data['stages']['T1_recent_A']['fit_totals']
        unsupported = counts['isolated_jump_return']-counts['isolated_jump_return_wind_supported']
        results['soft_label_weights'] = {'status':'prerequisite_supported' if unsupported>=SOFT_MIN_POINTS else 'screened_out',
                                         'unsupported_candidates':unsupported,'minimum':SOFT_MIN_POINTS,
                                         'source_sha256':sha256(quality)}
    else:
        results['soft_label_weights'] = {'status':'prerequisite_report_missing','source':str(quality)}
    return results


def run_screens(args):
    train, _ = chronological_dates(args.split,'A')
    roster = load_roster(args.data_root)
    days, base, lock = historical_reference(args,train,roster)
    historical_train, _ = replay_dates(args.split)
    truth = labels(args.data_root,days,roster)
    screens = check_prerequisites(args)
    satellite = SatelliteFeatures(args.data_root,args.out/'satellite_cache')
    jobs = {
        SAT_ARM: lambda: {**sat_screen(base,truth,satellite.days(days,roster),roster),
                         'inputs':satellite.stamp(days)},
        SOURCE_ARM: lambda: run_source_screen(args,historical_train,days,roster,base,truth),
    }
    for i, (name, test) in enumerate(jobs.items()):
        show_progress(args.out,'候选训练期筛选',i,len(jobs),name)
        try:
            screens[name] = test()
        except Exception as error:
            traceback.print_exc()
            screens[name] = {'status':'failed','error':str(error),'promoted':False}
    show_progress(args.out,'候选训练期筛选',len(jobs),len(jobs),'已保存')
    freeze_json(args.out/'screens.json',{'warm_forecast_lock':lock,'results':screens})
    return screens,satellite


def run_source_screen(args, train, days, roster, base, truth):
    replacement, flagged, stamp = source_weather(args,train,days,roster)
    return {'status':'scored', **source_screen(base,truth,replacement,flagged),'weather':stamp}


def conditional_study(args, screened, satellite):
    names = tuple(name for name in (SAT_ARM,SOURCE_ARM) if screened[name]['promoted'])
    weather = {}

    def predict(phase, train, days, sites, seed, selected):
        keys = forecast_keys(days,sites)
        base, lock = t1_v4_reference(args,phase,keys,seed)
        arms, source = {'baseline':base}, {'v4':lock,'screen':sha256(args.out/'screens.json')}
        if SAT_ARM in selected:
            _, predictions, receipt = t1_predict(args,phase,train,days,sites,seed,(SAT_ARM,),
                component=SAT_ARM,additions=satellite,fractions={SAT_ARM:1.})
            arms[SAT_ARM],source[SAT_ARM] = predictions[SAT_ARM],receipt
        if SOURCE_ARM in selected:
            if phase not in weather:
                replacement, _, stamp = source_weather(args,train,days,sites)
                weather[phase] = replacement,stamp
            replacement, stamp = weather[phase]
            pair, receipt = pair_for(args,phase,seed,keys,replacement)
            fit = args.phase_run/'T1/phase_fit.json'
            arms[SOURCE_ARM] = phase_forecasts(pair['candidate'],read_json(fit)['sites'],(PHASE_ARMS[1],))[PHASE_ARMS[1]]
            source[SOURCE_ARM] = {'reference':receipt,'weather':stamp,'phase_fit':sha256(fit)}
        return keys,arms,source

    return predict,names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out':RUNS/'experiment_closeout_gpu_v1','data-root':Path('/workspace/qhl/data'),'split':SPLIT_PATH,
                'workdir':Path('/workspace/cpy/code/workdir'),'companion-code':Path('/workspace/cpy/code'),
                'source-run':RUNS/'t1_source_power_gpu_v1','power-run':RUNS/'t1_t3_fast_gpu_v1',
                'wind-run':RUNS/'wind_attribution_gpu_v1','warm-run':RUNS/'strict_warm_replay_gpu_v1',
                'phase-run':RUNS/'phase_anchor_cpu_v1','dynamic-run':RUNS/'t1_dynamic_gradient_gpu_v1',
                'obs-run':RUNS/'obs_vector_support_gpu_v1','audit-run':RUNS/'quality_diagnostics_cpu_v1'}
    for name,value in defaults.items():
        parser.add_argument(f'--{name}',type=Path,default=value)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    parser.add_argument('--threads',type=int,choices=range(1,5),default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'.run.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = ('run_experiment_closeout.py','run_t1_source_power.py','closeout_weather.py','run_power_hurdle.py')
        freeze_json(args.out/'plan.json',{'args':{k:str(v) for k,v in vars(args).items() if k!='split'},
            'split':sha256(args.split),'seeds':list(B_SEEDS),'soft_min_points':SOFT_MIN_POINTS,
            'code':{name:sha256(Path(__file__).with_name(name)) for name in files}})
        started,result,pending = monotonic(),{},{}
        show_progress(args.out,'剩余实验核对',0,3,'条件机制筛选')
        try:
            screened,satellite = run_screens(args)
            result['screens'] = screened
            study = conditional_study(args,screened,satellite)
            show_progress(args.out,'剩余实验核对',1,3,'冻结入围预测' if study[1] else '无入围候选')
            result['T1_v4'] = {'status':'complete',**stage(args,'T1',pending,study=study)} if study[1] else {
                'status':'not_triggered'}
        except Exception as error:
            traceback.print_exc()
            result['T1_v4'] = {'status':'failed','error':str(error)}
        write_json(args.out/'summary.json',result)
        show_progress(args.out,'剩余实验核对',2,3,'旧分源功率回执和统一复核')
        try:
            result['T1_source_power'] = source_reconciliation(args,allow_fit=result['T1_v4']['status']!='failed')
            if 'T1' in pending:
                result['T1_v4'].update(B=pending['T1'](), B_status='scored')
        except Exception as error:
            traceback.print_exc()
            if 'T1_source_power' in result:
                result['T1_v4'].update(status='failed', B_status='scoring_failed', error=str(error))
            else:
                result['T1_source_power'] = {'status':'failed','error':str(error)}
        result['seconds'] = round(monotonic()-started,2)
        write_json(args.out/'summary.json',result)
        show_progress(args.out,'剩余实验核对',3,3,'结果已保存')
        print(f'结果摘要 {args.out/"summary.json"}',flush=True)
        return int(any(value.get('status')=='failed' for value in (
            result['T1_source_power'],result['T1_v4'],*result.get('screens',{}).values())))


if __name__=='__main__':
    raise SystemExit(main())
