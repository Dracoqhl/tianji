#!/usr/bin/env python3
"""Diagnose warm-A curve errors and the effect of T2 valid-point extrapolation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from advance_batch_b import read_frozen_forecast
from batch_energy_short import energy_arm_report, weather_day_targets
from compare_warm_t1 import PRIMARY_SCORE
from experiment_companion_t1 import read_json
from independent_t2 import load_daily_targets
from run_companion_warm_t1 import SPLIT_PATH, date_roles, forecast_keys, sha256, write_json
from score_forecasts import DAY_KEYS, MAX_PREDICTION, MIN_VALID_POINTS, POINT_KEYS, align_predictions, score_track
from warm_shared_power import load_roster


def station_day_errors(truth, prediction):
    predicted = align_predictions(truth, prediction, POINT_KEYS, 'prediction', MAX_PREDICTION['T1'])
    points = truth.assign(prediction=predicted)
    usable = points.power_limit.eq(1) & np.isfinite(points.power)
    valid = points.loc[usable].copy()
    valid['error'] = valid.prediction - valid.power
    valid['error_sq'] = valid.error**2
    daily = points.groupby(DAY_KEYS).prediction.sum().mul(0.25).rename('energy_full').to_frame()
    statistics = valid.groupby(DAY_KEYS).agg(
        valid_points=('power', 'size'), bias_mw=('error', 'mean'), mse_mw2=('error_sq', 'mean'),
        energy_on_valid_points=('prediction', 'mean'),
    )
    daily = daily.join(statistics)
    daily['energy_on_valid_points'] = daily.energy_on_valid_points.mul(24).fillna(daily.energy_full)
    daily['shape_mse_mw2'] = (daily.mse_mw2 - daily.bias_mw**2).clip(lower=0)
    return daily.reset_index()


def t1_error_components(daily):
    scored = daily.loc[daily.valid_points.ge(MIN_VALID_POINTS)].assign(bias_mw2=lambda rows: rows.bias_mw**2)
    by_station = scored.groupby('station_id')[['mse_mw2', 'bias_mw2', 'shape_mse_mw2']].mean()
    average = by_station.mean()
    return {'aggregation': 'mean of station means over scoreable days',
            'average_squared_error_components_mw2': average.to_dict(),
            'daily_level_share_of_squared_error': float(average.bias_mw2 / average.mse_mw2)
            if average.mse_mw2 > 0 else None, 'per_station_components_mw2': by_station.to_dict(orient='index')}


def by_coverage(daily, eligible):
    rows = daily.merge(eligible.rename(columns={'energy': 'actual_energy'}), on=DAY_KEYS, validate='one_to_one')
    results = {}
    for name, block in (('complete_96', rows.loc[rows.valid_points.eq(96)]),
                        ('partial_72_to_95', rows.loc[rows.valid_points.between(MIN_VALID_POINTS, 95)])):
        summary = {'station_days': len(block), 'stations': block.station_id.nunique()}
        if not block.empty:
            errors = {}
            for estimate in ('energy_full', 'energy_on_valid_points'):
                relative_sq = ((block[estimate] - block.actual_energy) / block.actual_energy)**2
                errors[estimate] = float(np.sqrt(relative_sq.groupby(block.station_id).mean()).mean())
            summary['station_mean_relative_rmse'] = errors
            summary['mean_abs_energy_change_mwh'] = float((block.energy_full - block.energy_on_valid_points).abs().mean())
        results[name] = summary
    return results


def evaluate(truth, eligible, prediction, roster):
    daily = station_day_errors(truth, prediction)
    arms = {name: daily[DAY_KEYS].assign(energy=daily[name])
            for name in ('energy_full', 'energy_on_valid_points')}
    scores = energy_arm_report(truth, eligible, arms, roster)
    monthly = {}
    for month in sorted(truth.date.str[:7].unique()):
        selected = truth.date.str.startswith(month)
        day_selection = eligible.date.str.startswith(month)
        block = {name: forecast.loc[forecast.date.str.startswith(month)] for name, forecast in arms.items()}
        month_scores = energy_arm_report(truth.loc[selected], eligible.loc[day_selection], block, roster)
        monthly[month] = {
            'known_mask_delta_readme': month_scores['energy_on_valid_points'][PRIMARY_SCORE]
            - month_scores['energy_full'][PRIMARY_SCORE],
            'coverage': by_coverage(daily, eligible.loc[day_selection]),
        }
    return {
        'T1': score_track('T1', truth, prediction, roster),
        'T1_daily_bias_and_shape': t1_error_components(daily),
        'T2_integral': scores['energy_full'],
        'T2_known_mask_diagnostic': scores['energy_on_valid_points'],
        'T2_known_mask_delta_readme': scores['energy_on_valid_points'][PRIMARY_SCORE] - scores['energy_full'][PRIMARY_SCORE],
        'T2_by_valid_point_coverage': by_coverage(daily, eligible),
        'T2_monthly_known_mask': monthly,
    }


def adopted_forecast(policy_path, directory, days, roster, split):
    from advance_batch_b import keys_for

    policy = read_json(policy_path)
    directory = directory or Path(policy['a_run'])
    if policy['split_sha256'] != sha256(split):
        raise ValueError('The working T2 recipe uses another split.')
    manifest = directory / policy['a_forecasts']
    saved = read_json(manifest)
    score = read_json(directory / policy['a_score'])
    name = policy['name']
    weights = policy['components']
    if (score['prediction_lock_sha256'] != sha256(manifest) or score['gate']['candidate'] != name
            or score['gate']['status'] != 'promoted' or score['window'] != [str(day) for day in days]
            or any(not np.isfinite(w) or w <= 0 for w in weights.values()) or not np.isclose(sum(weights.values()), 1)):
        raise ValueError('The adopted T2 result differs from the frozen A prediction or recipe.')
    keys = keys_for(days, roster, 'T2')
    names = [*weights, name]
    forecasts = {arm: read_frozen_forecast(manifest.parent / f'{arm}.csv', saved['hashes'][arm], keys, 'T2')
                 for arm in names}
    reconstructed = sum(weight * forecasts[arm].energy for arm, weight in weights.items())
    if not np.allclose(reconstructed, forecasts[name].energy, rtol=0, atol=1e-9):
        raise ValueError('The stored T2 mixture differs from the adopted weighted components.')
    return name, forecasts[name], {'policy_sha256': sha256(policy_path), 'forecast_sha256': saved['hashes'][name]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/qhl/data'))
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--forecast', action='append', required=True, metavar='NAME=CSV')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--t2-policy', type=Path)
    parser.add_argument('--t2-adopted-run', type=Path)
    args = parser.parse_args()

    _, days = date_roles(args.split, 'A')
    roster = load_roster(args.data_root)
    keys = forecast_keys(days, roster)
    inputs = {}
    for value in args.forecast:
        name, sep, path = value.partition('=')
        if not sep or not name or not path or name in inputs:
            parser.error('Provide distinct NAME=CSV forecast arguments.')
        inputs[name] = Path(path)
    digests = {name: sha256(path) for name, path in inputs.items()}
    forecasts = {name: read_frozen_forecast(path, digests[name], keys, 'T1') for name, path in inputs.items()}
    adopted = None
    if args.t2_policy:
        adopted = adopted_forecast(args.t2_policy, args.t2_adopted_run, days, roster, args.split)
    truth, eligible = load_daily_targets(weather_day_targets(args.data_root, days), roster)
    scores = {name: evaluate(truth, eligible, prediction, roster) for name, prediction in forecasts.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'warm A only; label masks are used for scoring diagnostics',
              'split_sha256': sha256(args.split), 'forecast_sha256': digests, 'days': len(days), 'scores': scores}
    if adopted:
        name, forecast, provenance = adopted
        result = energy_arm_report(truth, eligible, {name: forecast}, roster)[name]
        report['working_t2_baseline'] = {'name': name, 'score': result, **provenance}
        print(f'[T2 A] 当前融合基线 {name} {result[PRIMARY_SCORE]:.6f}', flush=True)
    write_json(args.out, report)
    for name, row in scores.items():
        print(f'{name} T1={row["T1"][PRIMARY_SCORE]:.4f} T2={row["T2_integral"][PRIMARY_SCORE]:.4f} '
              f'T2_known_mask_delta={row["T2_known_mask_delta_readme"]:+.4f}')
    print(args.out)


if __name__ == '__main__':
    main()
