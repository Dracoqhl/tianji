#!/usr/bin/env python3
"""Test within-day phase and low-energy anchor corrections on frozen forecasts."""
import argparse
import fcntl
from functools import partial
from pathlib import Path
from time import monotonic
import traceback

import numpy as np

from advance_batch_b import keys_for
from experiment_companion_t1 import read_json
from run_capacity_confirmation import original_t2
from run_companion_warm_t1 import B_SEEDS, SPLIT_PATH, chronological_dates, forecast_keys, sha256, write_json
from run_forecast_blends import freeze_json
from run_fusion_mechanisms import finish_pending, stage
from run_strict_warm_replay import replay_dates
from run_t1_issue_wind_error import dynamic_baseline
from run_wind_attribution import STATION, read_arm
from score_forecasts import DAY_KEYS, MAX_PREDICTION, MIN_VALID_POINTS, POINT_KEYS
from screen_companion_t1_stations import show_progress
from warm_shared_power import labels


RUNS = Path('/workspace/qhl/runs')
SHIFT_FRACTION = .25
MIN_SITE_DAYS = 15
T1_ARMS = ('phase_15min', 'symmetric_smoothing')
T2_ARM = 'low_energy_anchor_taper'
ENERGY_BAND = (60., 180.)
PRIOR_ANCHOR = 't1_v3_integral_10'


def shift_steps(base):
    earlier = np.concatenate((base[:, 1:], base[:, -1:]), axis=1) - base
    later = np.concatenate((base[:, :1], base[:, :-1]), axis=1) - base
    return tuple(step - step.mean(axis=1, keepdims=True) for step in (earlier, later))


def learn_phase(history, truth, roster):
    aligned = history.merge(truth[POINT_KEYS + ['power', 'power_limit']], on=POINT_KEYS, how='left',
                            sort=False, validate='one_to_one', indicator=True)
    if not aligned['_merge'].eq('both').all():
        raise ValueError('Historical T1 forecast and labels have different point keys.')
    daily = aligned.iloc[::96][DAY_KEYS].reset_index(drop=True)
    base = aligned.prediction.to_numpy().reshape(-1, 96)
    target = aligned.power.to_numpy().reshape(base.shape)
    valid = np.isfinite(target) & aligned.power_limit.eq(1).to_numpy().reshape(base.shape)
    count = valid.sum(axis=1)
    candidates = [base, *(np.clip(base + SHIFT_FRACTION * step, 0, MAX_PREDICTION['T1'])
                          for step in shift_steps(base))]
    errors = [np.sqrt(np.where(valid, (values - target)**2, 0).sum(axis=1) / np.maximum(count, 1))
              for values in candidates]
    audit = {}
    for site in roster:
        usable = daily.station_id.eq(site).to_numpy() & (count >= MIN_VALID_POINTS)
        early = daily.date.lt('2025-07-01').to_numpy()
        blocks = []
        for mask in (usable & early, usable & ~early):
            n = int(mask.sum())
            gains = [float((errors[0][mask] - error[mask]).mean()) for error in errors[1:]] if n else [None, None]
            direction = None
            if n >= MIN_SITE_DAYS:
                if gains[0] > max(1e-8, gains[1] + 1e-8):
                    direction = 'earlier'
                elif gains[1] > max(1e-8, gains[0] + 1e-8):
                    direction = 'later'
            blocks.append({'scored_days': n, 'rmse_gain_mw': dict(zip(('earlier', 'later'), gains)),
                           'direction': direction})
        agreed = blocks[0]['direction'] if blocks[0]['direction'] == blocks[1]['direction'] else None
        audit[site] = {'direction': agreed if site != STATION else None, 'blocks': blocks}
    return audit


def phase_forecasts(base, audit, names):
    values = base.prediction.to_numpy().reshape(-1, 96)
    direction = np.array([audit[site]['direction'] for site in base.station_id.iloc[::96]])
    earlier, later = shift_steps(values)
    identified = np.isin(direction, ('earlier', 'later'))
    steps = {
        T1_ARMS[0]: np.where((direction == 'earlier')[:, None], earlier, later),
        T1_ARMS[1]: (earlier + later) / 2,
    }
    return {name: base.assign(prediction=np.clip(values + np.where(identified[:, None],
                SHIFT_FRACTION * steps[name], 0), 0, MAX_PREDICTION['T1']).ravel()) for name in names}


def historical_reference(args, train, sites):
    historical_train, historical_days = replay_dates(args.split)
    if not set(historical_train + historical_days).issubset(train):
        raise ValueError('T1 phase calibration dates escape the fit partition.')
    history = args.warm_run / 'T1'
    folder = history / 'forecasts'
    lock = sha256(folder / 'predictions_frozen.json')
    contract = read_json(args.warm_run / 'plan.json')['dates']
    if (read_json(history / 'score.json')['prediction_lock'] != lock
            or contract['train'] != list(map(str, historical_train))
            or contract['forecast'] != list(map(str, historical_days))
            or contract['data_root'] != str(args.data_root.resolve())):
        raise ValueError('T1 phase calibration must use the locked strict warm replay.')
    baseline = read_arm(folder, 'baseline', forecast_keys(historical_days, sites), 'T1')
    return historical_days, baseline, lock


def historical_phase(args, train, sites):
    historical_days, baseline, lock = historical_reference(args, train, sites)
    fitted = learn_phase(baseline, labels(args.data_root, historical_days, sites), sites)
    stamp = freeze_json(args.out / 'T1/phase_fit.json', {'calibration_source': lock, 'sites': fitted,
                                                        'shift_fraction': SHIFT_FRACTION})
    return fitted, stamp


def t1_study(args):
    fitted = None

    def predict(phase, train, days, sites, seed, names):
        nonlocal fitted
        if fitted is None:
            fitted = historical_phase(args, train, sites)
        audit, stamp = fitted
        keys = forecast_keys(days, sites)
        base, lock = dynamic_baseline(args, phase, keys, seed)
        return keys, {'baseline': base, **phase_forecasts(base, audit, names)}, {'v3': lock, 'phase': stamp}

    return predict, T1_ARMS


def taper_fraction(energy):
    lower, upper = ENERGY_BAND
    return np.clip((energy - lower) / (upper - lower), 0, 1)


def t2_v3_reference(args, phase, days, sites, seed):
    keys = keys_for(days, sites, 'T2')
    calibrated, v2, _ = original_t2(args, phase, seed, days, sites)
    root = args.fusion_run / 'T2'
    gate = read_json(root / 'gate.json')
    if gate['selected'] != PRIOR_ANCHOR or not gate['promoted']:
        raise ValueError('T2 taper requires the adopted v3 anchor.')
    if phase == 'A':
        folder = root / 'A/forecasts'
        expected = gate['screen_locks']['A']
        name = PRIOR_ANCHOR
    else:
        folder = root / 'B' / str(seed)
        ledger = root / 'B/predictions_frozen.json'
        report = read_json(root / 'B/score.json')
        if not report['acceptance']['accepted'] or report['prediction_lock'] != sha256(ledger):
            raise ValueError('T2 v3 B evidence is missing or changed.')
        expected = read_json(ledger)[str(seed)]
        name = 'candidate'
    marker = folder / 'predictions_frozen.json'
    lock = sha256(marker)
    meta = read_json(marker)['sources']
    if phase == 'B':
        if meta['gate'] != gate:
            raise ValueError('T2 v3 B uses an unexpected gate.')
        meta = meta['source']
    if lock != expected or meta['base_lock'] != v2:
        raise ValueError('T2 v2 and v3 are not linked to the same paired inputs.')
    base = read_arm(folder, name, keys, 'T2')
    return keys, calibrated, base, {'v3': lock, 'v2': v2, 'anchor': meta.get('t1_lock')}


def t2_predict(args, phase, _train, days, sites, seed, _names):
    keys, calibrated, base, source = t2_v3_reference(args, phase, days, sites, seed)
    fraction = taper_fraction(calibrated.energy)
    candidate = base.assign(energy=calibrated.energy + fraction * (base.energy - calibrated.energy))
    return keys, {'baseline': base, T2_ARM: candidate}, {
        'v3': source['v3'], 'v2': source['v2'], 'band_mwh': list(ENERGY_BAND),
        'tapered_station_days': int((fraction < 1).sum()), 'zero_anchor_station_days': int((fraction == 0).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {'out': RUNS / 'phase_anchor_cpu_v1', 'data-root': Path('/workspace/qhl/data'), 'split': SPLIT_PATH,
                'dynamic-run': RUNS / 't1_dynamic_gradient_gpu_v1', 'warm-run': RUNS / 'strict_warm_replay_gpu_v1',
                't2-prior': RUNS / 'steamlink_followup_gpu_v1', 'fusion-run': RUNS / 'fusion_mechanisms_cpu_v1'}
    for name, value in defaults.items():
        parser.add_argument(f'--{name}', type=Path, default=value)
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_json(args.out / 'plan.json', {'args': {name: str(value) for name, value in vars(args).items()},
                    'split': sha256(args.split), 'seeds': list(B_SEEDS), 'code': sha256(Path(__file__)),
                    'shift_fraction': SHIFT_FRACTION, 'minimum_site_days': MIN_SITE_DAYS, 'taper_band_mwh': list(ENERGY_BAND)})
        start, result, pending = monotonic(), {}, {}
        studies = {'T1': t1_study(args), 'T2': (partial(t2_predict, args), (T2_ARM,))}
        for i, (track, study) in enumerate(studies.items()):
            show_progress(args.out, '相位与积分权重', i, len(studies), track)
            try:
                result[track] = {'status': 'complete', **stage(args, track, pending, study=study)}
            except Exception as error:
                traceback.print_exc()
                result[track] = {'status': 'failed', 'error': str(error)}
            write_json(args.out / 'summary.json', result)
        finish_pending(args, result, pending)
        result['seconds'] = round(monotonic() - start, 2)
        write_json(args.out / 'summary.json', result)
        show_progress(args.out, '相位与积分权重', len(studies), len(studies), '结果已保存')
        print(f'结果摘要 {args.out / "summary.json"}', flush=True)
        return int(any(result[track]['status'] != 'complete' for track in studies))


if __name__ == '__main__':
    raise SystemExit(main())
