#!/usr/bin/env python3
"""Run H1/H2 with training-only screens and paired chronological confirmation."""
import argparse
from datetime import date
import fcntl
from pathlib import Path
import subprocess
from tempfile import mkdtemp
from time import monotonic
import traceback
from types import SimpleNamespace

import numpy as np

from advance_batch_b import keys_for, read_frozen_forecast
from batch_energy_short import energy_features, fit_energy, load_daily_targets, roster_for, weather_day_targets
from compare_t2_mlp import mix, select_mlp_runtime
from compare_warm_t1 import PRIMARY_SCORE, b_statistics
from confirm_working_baselines import average_seeds, keep_sites
from experiment_companion_t1 import assert_matched, identical_seed_groups, read_json
from run_companion_warm_t1 import A_SEED, B_SEEDS, EARLY_STOP_DAYS, SPLIT_PATH, chronological_dates, forecast_keys, run, sha256, write_json
from run_forecast_blends import blend, freeze_json, freeze_window
from run_pressure_profile import summarize_b
from run_steamlink_followup import old_t1, station_calibration, t2_scores, wind_alternatives
from run_t1_issue_wind_error import dynamic_baseline
from run_t1_t3_batch import paired_report
from run_wind_attribution import STATION, read_arm
from same_issue_weather import feature_spec
from score_forecasts import DAY_KEYS, MAX_PREDICTION
from screen_companion_t1_stations import block_comparisons, show_progress
from t1_dynamic_wind import NAMES as DYNAMIC_COLUMNS, append_dynamic
from warm_shared_power import fit_forecasts, labels, load_roster, power_weather


RUNS = Path('/workspace/qhl/runs')
EPOCHS = (8, 12, 16)
BASE_EPOCH = 'epoch_16'
CAPACITY = 'half_leaves'


def screen_windows(split):
    train, _ = chronological_dates(split, 'A')
    return [('inner_360', train[:360], train[360:450]), ('inner_450', train[:450], train[450:])]


def t1_tree(args, folder, train, days, seed, keys, reduced):
    name = CAPACITY if reduced else 'baseline'
    final = folder / name
    marker = final / 'training_receipt.json'
    if not marker.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        target = Path(mkdtemp(dir=folder, prefix=f'.{name}-attempt-'))
        options = SimpleNamespace(**vars(args))
        options.out, options.seed, options.phase = target, seed, 'A'
        options.train_block, options.smoke, options.check_only = None, False, False
        options.backend, options.components, options.point_partition = 'xgboost', [], 'full'
        options.early_stop_days, options.diagnostic = EARLY_STOP_DAYS, None

        def features(X, names, dates, sites, positions):
            return append_dynamic(args.data_root, args.dynamic_run/'native_weather', X, names, dates, sites, positions)

        run(options, extra_features=features, date_partition=(train, days),
            tree_capacity=CAPACITY if reduced else None, track_selection=True)
        target.rename(final)
    receipt = read_json(marker)
    params = receipt['experts'][0]
    if (receipt['chronological_dates'] != {'training': list(map(str, train)), 'forecast': list(map(str, days))}
            or receipt['seed'] != seed or set(params['selection_curves']) != {'training','validation'}
            or not set(DYNAMIC_COLUMNS).issubset(receipt['feature_names'])):
        raise ValueError('Mismatched H1 tree scope, features or training evaluation curves.')
    frame = read_frozen_forecast(final/receipt['prediction_file'], receipt['prediction_sha256'], keys, 'T1')
    return frame, receipt, sha256(marker)


def t1_pair(args, out, train, days, seed, phase=None):
    sites = load_roster(args.data_root)
    keys = forecast_keys(days, sites)
    wind, _, _, weather = wind_alternatives(args, train, days, sites, selected=CAPACITY)
    wind = keys.assign(prediction=wind.ravel())
    if phase is None:
        tree, old, old_lock = t1_tree(args, out, train, days, seed, keys, False)
        on_station = tree
        base = blend(tree, wind, .2, 'T1')
        route_lock = old_lock
    else:
        base, old_lock = dynamic_baseline(args, phase, keys, seed)
        on_station, route_lock = old_t1(args, phase, keys, seed)
        old_dir = args.dynamic_run / 'T1' / phase / str(seed) / 'dynamic_gradient'
        old = read_json(old_dir/'training_receipt.json')
        tree = read_frozen_forecast(old_dir/old['prediction_file'], old['prediction_sha256'], keys, 'T1')
        rebuilt = blend(keep_sites(tree, on_station, [STATION]), wind, .2, 'T1')
        if not np.allclose(rebuilt.prediction, base.prediction, atol=1e-6, rtol=0):
            raise ValueError('H1 fixed v3 baseline does not reproduce.')
    smaller, new, digest = t1_tree(args, out, train, days, seed, keys, True)
    assert_matched(old, new, ('seed','chronological_dates','companion_cache_metadata_sha256','device','threads',
                             'early_stopping_valid_rows','early_stopping_start_date','feature_names','config_sha256',
                             'valid_training_sample_sha256','companion_source_sha256'))
    old_params, new_params = (r['fit_parameters']['selection'] for r in (old,new))
    if new_params['max_leaves'] != max(2,old_params['max_leaves']//2):
        raise ValueError('Leaf count did not halve.')
    assert_matched(old_params,new_params,[name for name in old_params if name!='max_leaves'])
    smaller = blend(keep_sites(smaller, on_station, [STATION]), wind, .2, 'T1')
    arms, lock = freeze_window(out/'forecasts','T1',keys,{'baseline':base,'candidate':smaller},
        {'control':old_lock,'candidate':digest,'route':route_lock,'weather':weather})
    return sites, keys, arms, lock


def t1_score(args, out, days, bundle):
    sites, _, frames, lock = bundle
    rows = block_comparisons(labels(args.data_root, days, sites), frames, sites, ('candidate',))
    score = {a: data[PRIMARY_SCORE] for a,data in rows['scores'].items()}
    result = {'scores':score,'delta':score['candidate']-score['baseline'],'lock':lock}
    freeze_json(out/'score.json',{**rows,**result})
    return result


def run_t1(args):
    folder = args.out/'T1'
    parts = {}
    for name, fit, score in screen_windows(args.split):
        print(f'[T1 screen] {name}',flush=True)
        out = folder/name
        parts[name] = t1_score(args,out,score,t1_pair(args,out,fit,score,A_SEED))
    gain = float(np.mean([row['delta'] for row in parts.values()]))
    result = {'screen':parts,'screen_delta':gain}
    freeze_json(folder/'screen.json',result)
    if gain <= 0:
        return result
    fit, days = chronological_dates(args.split,'A')
    out = folder/'A'
    result['A'] = t1_score(args,out,days,t1_pair(args,out,fit,days,A_SEED,'A'))
    if result['A']['delta'] > 0:
        fit_b, days_b = chronological_dates(args.split,'B')
        if fit_b != fit + days:
            raise ValueError('T1 B must train on all A dates.')
        pairs, locks = {}, {}
        for seed in B_SEEDS:
            sites,keys,pairs[seed],locks[str(seed)] = t1_pair(args,folder/'B'/str(seed),fit_b,days_b,seed,'B')
        source = freeze_json(folder/'B/predictions_frozen.json',locks)
        result['B'] = summarize_b(paired_report(folder/'B','T1',labels(args.data_root,days_b,sites),
                                                sites,keys,pairs,source))
    return result


def scope(train, days, seed, device):
    return {'train': list(map(str,train)), 'forecast': list(map(str,days)), 'seed':seed, 'device':device}


def checkpoint_worker(args):
    contract = read_json(args.worker_plan)
    train, days = ([date.fromisoformat(d) for d in contract[k]] for k in ('train','forecast'))
    if not train or not days or max(train) >= min(days) or set(train)&set(days):
        raise ValueError('Noncausal MLP checkpoint worker dates.')
    if args.worker == 'weather':
        power_weather(args.data_root,train,days,args.mlp_cache,2,print)
    else:
        epochs = tuple(contract['epochs'])
        fit_forecasts(args.data_root,args.out,train,days,2,max(epochs),64,args.device,
                      cache_root=args.mlp_cache,seed=contract['seed'],checkpoint_epochs=epochs)


def epoch_forecasts(args, folder, train, days, seed, keys, epochs):
    location = folder/'mlp'
    folder.mkdir(parents=True,exist_ok=True)
    stamp = {**scope(train,days,seed,args.device),'epochs':list(epochs)}
    if not location.is_dir():
        pending = Path(mkdtemp(dir=folder,prefix='.mlp-attempt-'))
        plan = pending/'plan.json'
        freeze_json(plan,stamp)
        torch, weather = args.runtime
        options = [str(Path(__file__).resolve()),'--worker-plan',str(plan),'--data-root',str(args.data_root),
                   '--out',str(pending),'--device',args.device,'--mlp-cache',str(args.mlp_cache)]
        if torch != weather:
            subprocess.run([str(weather),'-u',*options,'--worker','weather'],check=True)
        subprocess.run([str(torch),'-u',*options,'--worker','mlp'],check=True)
        freeze_json(pending/'frozen.json',{
            'hashes':{str(e):sha256(pending/f'T2_epoch_{e}.csv') for e in epochs},
            'training':sha256(pending/'training_receipt.json')})
        pending.rename(location)
    old = read_json(location/'frozen.json')
    if read_json(location/'plan.json') != stamp or old['training'] != sha256(location/'training_receipt.json'):
        raise ValueError('MLP checkpoint plan or training receipt changed.')
    forecasts = {e:read_frozen_forecast(location/f'T2_epoch_{e}.csv',old['hashes'][str(e)],keys,'T2') for e in epochs}
    training = read_json(location/'training_receipt.json')['train_daily_rmse_mw']
    return forecasts, {'lock':sha256(location/'frozen.json'),'train_daily_rmse_mw':{str(e):training[e-1] for e in epochs}}


def inner_tree(args, folder, train, days, keys, sites):
    folder.mkdir(parents=True,exist_ok=True)
    target = folder/'direct'
    marker = target/'fit.json'
    fit, fit_info = energy_features(args,sites,train,feature_spec(args.workdir))
    pred, pred_info = energy_features(args,sites,days,feature_spec(args.workdir))
    if fit_info['columns'] != pred_info['columns']:
        raise ValueError('T2 direct tree feature mismatch.')
    plan = {'train_features':fit_info['sha256'],'pred_features':pred_info['sha256'],
            **scope(train,days,A_SEED,args.device)}
    if not marker.is_file():
        tmp = Path(mkdtemp(dir=folder,prefix='.tree-attempt-'))
        _, truth = load_daily_targets(weather_day_targets(args.data_root,train),sites)
        fit = fit.merge(truth,on=DAY_KEYS,validate='one_to_one')
        if len(fit) != len(truth):
            raise ValueError('T2 inner-fit matrix lost eligible labels.')
        values, _, stats = fit_energy(args,fit_info['columns'],fit,train,pred,'relative_mse',tmp,seed=A_SEED)
        pred[DAY_KEYS].assign(energy=values).to_csv(tmp/'forecast.csv',index=False)
        freeze_json(tmp/'fit.json',{'plan':plan,'forecast':sha256(tmp/'forecast.csv'),'stats':stats})
        tmp.rename(target)
    record = read_json(marker)
    if record['plan'] != plan:
        raise ValueError('T2 direct tree scope changed.')
    return read_frozen_forecast(target/'forecast.csv',record['forecast'],keys,'T2'),sha256(marker)


def original_t2(args, phase, seed, days, sites):
    folder = args.t2_prior/'T2'
    gate = read_json(folder/'a_score.json')['gate']
    if gate['selected'] != 'station_calibrated' or not gate['promoted']:
        raise ValueError('T2 source does not select the calibrated 75/25.')
    if phase == 'A':
        pair = folder/'A/forecasts'
        ref = gate['forecast_lock']
        name = 'station_calibrated'
    else:
        pair = folder/'B'/str(seed)/'forecasts'
        ref = read_json(folder/'B/predictions_frozen.json')[str(seed)]
        name = 'candidate'
    manifest = pair/'predictions_frozen.json'
    sources = read_json(manifest)['sources']
    if sha256(manifest) != ref or (phase=='B' and sources['gate'] != gate):
        raise ValueError('T2 source forecasts changed or use a different gate.')
    return read_arm(pair,name,keys_for(days,sites,'T2'),'T2'),ref,sources


def cached_direct(args, phase, seed, keys, calibration, sources):
    src = args.t2_prior/'T2'/phase/str(seed)/('calibration' if calibration else 'full')/'relative_mse'
    record = read_json(src/'fit.json')
    expected = sources['calibration']['history']['relative'] if calibration else sources['relative']
    if record != expected:
        raise ValueError('T2 frozen direct tree differs from its paired forecast receipt.')
    return read_frozen_forecast(src/'forecast.csv',record['prediction_sha256'],keys,'T2'),sha256(src/'fit.json')


def t2_pair(args, folder, train, days, seed, sites, epochs, phase=None):
    if phase is None:
        base = original_lock = sources = None
    else:
        base,original_lock,sources = original_t2(args,phase,seed,days,sites)
    tail = len(train)//5
    early, calibration = train[:-tail], train[-tail:]
    fits = {}
    for name,use,held in [('full',train,days),('calibration',early,calibration)]:
        keys = keys_for(held,sites,'T2')
        path = folder/name
        mlps, epoch_note = epoch_forecasts(args,path,use,held,seed,keys,epochs)
        if phase is None:
            direct, tree_lock = inner_tree(args,path,use,held,keys,sites)
        else:
            direct, tree_lock = cached_direct(args,phase,seed,keys,name=='calibration',sources)
        fits[name] = (mlps,direct,{'mlp':epoch_note,'tree_lock':tree_lock})
    _, actual = load_daily_targets(weather_day_targets(args.data_root,calibration),sites)
    arms = {'baseline':base} if base is not None else {}
    full, tree, full_meta = fits['full']
    cal, cal_tree, cal_meta = fits['calibration']
    factors = {}
    for e in epochs:
        factors[e], _ = station_calibration(mix(cal[e],cal_tree),actual)
        raw = mix(full[e],tree)
        multiplier = raw.station_id.map(factors[e]).fillna(1.)
        arms[f'epoch_{e}'] = raw.assign(energy=(raw.energy*multiplier).clip(0,MAX_PREDICTION['T2']))
    if base is None:
        arms['baseline'] = arms.pop(BASE_EPOCH)
    keys = keys_for(days,sites,'T2')
    frames, lock = freeze_window(folder/'forecasts','T2',keys,arms,{
        'original_lock':original_lock,'training':full_meta,'calibration':cal_meta,
        'factors':{str(e):v for e,v in factors.items()}})
    pure, pure_lock = freeze_window(folder/'mlp_forecasts','T2',keys,
                                   {f'mlp_epoch_{e}':v for e,v in full.items()},full_meta)
    return keys,frames,lock,pure,pure_lock


def t2_score(args, out, days, sites, bundle):
    _,arms,lock,pure,pure_lock = bundle
    scores, groups = t2_scores(args,days,sites,{**arms,**pure})
    row = {'scores':{name:value[PRIMARY_SCORE] for name,value in scores.items()},
           'groups':groups,'lock':lock,'mlp_lock':pure_lock}
    freeze_json(out/'score.json',{**row,'by_station':{name:val['by_station'] for name,val in scores.items()},
                                  'fit_epoch_notes':read_json(out/'forecasts/predictions_frozen.json')['sources']})
    return row


def run_t2(args):
    args.runtime = select_mlp_runtime(args)
    sites = roster_for(args.data_root,'T2')
    root = args.out/'T2'
    screens = {}
    for name,fit,days in screen_windows(args.split):
        print(f'[T2 screen] {name}',flush=True)
        out = root/name
        screens[name] = t2_score(args,out,days,sites,t2_pair(args,out,fit,days,A_SEED,sites,EPOCHS))
    gains = {e:float(np.mean([r['scores'][f'epoch_{e}']-r['scores']['baseline'] for r in screens.values()]))
             for e in EPOCHS[:-1]}
    best = max(gains,key=gains.get)
    gain = gains[best]
    result = {'screen':screens,'screen_deltas':{str(k):v for k,v in gains.items()},
              'selected_epoch':best if gain>0 and len(set(gains.values()))==len(gains) else 16}
    freeze_json(root/'screen.json',result)
    if result['selected_epoch']==16:
        return result
    fit, days = chronological_dates(args.split,'A')
    out = root/'A'
    result['A'] = t2_score(args,out,days,sites,t2_pair(args,out,fit,days,A_SEED,sites,(best,),'A'))
    name = f'epoch_{best}'
    if result['A']['scores'][name] <= result['A']['scores']['baseline']:
        return result
    fit_b, days_b = chronological_dates(args.split,'B')
    if fit_b != fit+days:
        raise ValueError('T2 B training must include all A dates.')
    pairs,locks = {},{}
    for seed in B_SEEDS:
        keys,frames,locks[str(seed)],_,_ = t2_pair(args,root/'B'/str(seed),fit_b,days_b,seed,sites,(best,),'B')
        pairs[seed] = {'baseline':frames['baseline'],'candidate':frames[name]}
    lock = freeze_json(root/'B/predictions_frozen.json',locks)
    scores = {str(s):t2_scores(args,days_b,sites,p)[0] for s,p in pairs.items()}
    ensemble, groups = t2_scores(args,days_b,sites,average_seeds(keys,pairs,'T2'))
    def difference(rows):
        return rows['candidate'][PRIMARY_SCORE]-rows['baseline'][PRIMARY_SCORE]
    deltas = {s:difference(v) for s,v in scores.items()}
    decision = b_statistics(list(deltas.values()),difference(ensemble))
    result['B'] = {'scores':{a:r[PRIMARY_SCORE] for a,r in ensemble.items()},'groups':groups,'acceptance':decision,
                   'seed_deltas':deltas,'prediction_lock':lock,
                   'identical_seed_groups':{a:identical_seed_groups({s:p[a].energy.to_numpy() for s,p in pairs.items()})
                                            for a in ('baseline','candidate')}}
    freeze_json(root/'B/score.json',result['B'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out':RUNS/'capacity_confirmation_gpu_v1','data-root':Path('/workspace/qhl/data'),
        'workdir':Path('/workspace/cpy/code/workdir'),'companion-code':Path('/workspace/cpy/code'),
        'split':SPLIT_PATH,'dynamic-run':RUNS/'t1_dynamic_gradient_gpu_v1','power-run':RUNS/'t1_t3_fast_gpu_v1',
        'mlp-run':RUNS/'warm_shared_power_v1','mlp-cache':RUNS/'warm_power_feature_cache',
        't2-prior':RUNS/'steamlink_followup_gpu_v1'}
    for name,value in defaults.items():
        parser.add_argument(f'--{name}',type=Path,default=value)
    for name in ('config','mlp-python','mlp-weather-python','worker-plan'):
        parser.add_argument(f'--{name}',type=Path)
    parser.add_argument('--worker',choices=('weather','mlp'))
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--threads',type=int,choices=range(1,5),default=4)
    args = parser.parse_args()
    if args.worker:
        checkpoint_worker(args)
        return 0
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'.run.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze_json(args.out/'plan.json',{'epochs':list(EPOCHS),'seeds':list(B_SEEDS),'split':sha256(args.split),
            'args':{key:str(value) for key,value in vars(args).items()},'code':{name:sha256(Path(__file__).with_name(name))
            for name in ('run_capacity_confirmation.py','run_companion_warm_t1.py','warm_shared_power.py','tree_model.py')}})
        start,result=monotonic(),{}
        for i,(track,stage) in enumerate((('T1',run_t1),('T2',run_t2))):
            show_progress(args.out,'超参数验收',i,2,track)
            try:
                result[track]={'status':'complete',**stage(args)}
            except Exception as error:
                traceback.print_exc()
                result[track]={'status':'failed','error':str(error)}
            write_json(args.out/'summary.json',result)
        result['seconds']=round(monotonic()-start,2)
        write_json(args.out/'summary.json',result)
        show_progress(args.out,'超参数验收',2,2,'完成')
        print(f'结果摘要 {args.out / "summary.json"}',flush=True)
        return int(any(result[t]['status']!='complete' for t in ('T1','T2')))


if __name__=='__main__':
    raise SystemExit(main())
