"""Training-only diagnostics for FY-4B motion and isolated forecast source anomalies."""
from hashlib import sha256 as digest
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from run_companion_warm_t1 import sha256
from run_steamlink_followup import corrected_sources, finite_mean, mapped
from run_wind_attribution import SOURCES, STATION, observed_curve, wind_labels
from same_issue_weather import archived_issue_weather
from score_forecasts import DAY_KEYS, MAX_PREDICTION, MIN_VALID_POINTS, POINT_KEYS
from screen_companion_t1_stations import block_comparisons, station_scores


SAT_ARM = 'fy4b_motion'
SOURCE_ARM = 'isolated_source_fallback'
MOTION = ('fy4b_motion_east_cells', 'fy4b_motion_north_cells', 'fy4b_motion_gain', 'fy4b_motion_brightness_change')
SOLVE_MONTHS = ('2025-05', '2025-06')
TEST_MONTHS = ('2025-07', '2025-08')
LEAD_POINTS = 24
SAT_MIN_DAYS = 80
SOURCE_MIN_DAYS = 15


def patch_signal(frame):
    """Return static evidence and local translation within a 13-cell square."""
    center = frame[1, 2:11, 2:11]
    old = frame[0, 2:11, 2:11]
    usable = np.isfinite(old) & np.isfinite(center)
    if usable.sum() < 65:
        return np.full(7, np.nan)
    earlier, latest = old[usable], center[usable]
    candidates = []
    for y in range(-2, 3):
        for x in range(-2, 3):
            shifted = frame[0, 2-y:11-y, 2-x:11-x]
            valid = np.isfinite(shifted) & np.isfinite(center)
            if valid.sum() < 65:
                continue
            a, b = shifted[valid], center[valid]
            a, b = a-a.mean(), b-b.mean()
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm >= len(a):
                candidates.append((float((a @ b) / norm), y, x))
    base = next((score for score, y, x in candidates if (y, x) == (0, 0)), None)
    if base is None:
        movement = (np.nan, np.nan, np.nan)
    else:
        score, north, east = max(candidates, key=lambda row: (row[0], -abs(row[1])-abs(row[2])))
        movement = (east, north, score-base) if score >= .5 and score-base >= .03 else (0, 0, max(0., score-base))
    return np.array([latest.mean(), latest.std(), float(usable.mean()), *movement, latest.mean()-earlier.mean()])


def read_satellite(path, coordinates):
    from netCDF4 import Dataset

    data = np.full((len(coordinates), 7), np.nan, dtype=np.float32)
    if not path.is_file():
        return data
    with Dataset(path) as ds:
        times, lat, lon = (np.ma.asarray(ds.variables[key][:], dtype=float).filled(np.nan)
                           for key in ('time','lat','lon'))
        field = ds.variables['tbb13']
        if (not np.array_equal(times, np.arange(-4,0)) or tuple(field.dimensions) != ('time','lat','lon')
                or not np.isfinite(lat).all() or not np.isfinite(lon).all()):
            raise ValueError(f'{path}: unexpected FY-4B dataset layout.')
        raw = np.ma.asarray(field[-2:], dtype=np.float32).filled(np.nan)
    if lat[0] > lat[-1]:
        lat, raw = lat[::-1], raw[:, ::-1, :]
    if lon[0] > lon[-1]:
        lon, raw = lon[::-1], raw[:, :, ::-1]
    if (np.diff(lat) <= 0).any() or (np.diff(lon) <= 0).any():
        raise ValueError(f'{path}: expected monotonic FY-4B axes.')
    raw = np.where((raw > 0) & np.isfinite(raw), raw, np.nan)
    for index, (y,x) in enumerate(coordinates):
        j, k = np.abs(lat-y).argmin(), np.abs(lon-x).argmin()
        if 6 <= j < len(lat)-6 and 6 <= k < len(lon)-6:
            data[index] = patch_signal(raw[:, j-6:j+7, k-6:k+7])
    return data


class SatelliteFeatures:
    names = MOTION

    def __init__(self, root, cache):
        static = root / 'static/power_stations.csv'
        table = pd.read_csv(static, dtype={'station_id':str}).set_index('station_id').sort_index()
        if not table.index.is_unique or len(table) != 18 or not np.isfinite(table[['latitude','longitude']]).all().all():
            raise ValueError('Expected eighteen unique farms with finite coordinates.')
        self.farms = table.index.tolist()
        self.coordinates = table[['latitude','longitude']].to_numpy()
        self.root = root
        self.version = digest((sha256(static)+sha256(Path(__file__))).encode()).hexdigest()
        self.cache = cache / self.version
        self.cache.mkdir(parents=True, exist_ok=True)

    def source(self, day):
        path = self.root / 'train' / f'{day:%Y%m%d}_0000/fy4b/fy4b_tbb13_input.nc'
        info = path.stat() if path.is_file() else None
        stamp = f'{day}:{info.st_size}:{info.st_mtime_ns}' if info else f'{day}:missing'
        return path, stamp

    def stamp(self, dates):
        key = digest('\n'.join(self.source(day)[1] for day in dates).encode()).hexdigest()
        return {'extractor':self.version, 'input_sha256':key, 'days':len(dates)}

    def days(self, days, roster):
        columns = [self.farms.index(site) for site in roster]
        values = []
        for i, day in enumerate(days, 1):
            path, stamp = self.source(day)
            receipt = self.cache / f'{day}-{digest(stamp.encode()).hexdigest()[:16]}.npy'
            if not receipt.is_file():
                data = read_satellite(path, self.coordinates)
                if self.source(day)[1] != stamp:
                    raise ValueError(f'{path}: FY-4B file changed.')
                with NamedTemporaryFile(dir=self.cache, suffix='.npy', delete=False) as temp:
                    np.save(temp, data, allow_pickle=False)
                Path(temp.name).replace(receipt)
            data = np.load(receipt, allow_pickle=False)
            if data.shape != (len(self.farms),7) or np.isinf(data).any():
                raise ValueError(f'{receipt}: cached FY-4B station features are invalid.')
            values.append(data[columns])
            if i in (1,len(days)) or i%100==0:
                print(f'[FY-4B] {i}/{len(days)} days', flush=True)
        return np.concatenate(values)

    def __call__(self, matrix, names, dates, roster, positions):
        if set(MOTION) & set(names):
            raise ValueError('Satellite motion feature names collide.')
        values = self.days(dates,roster)[:,3:][positions//96]
        return np.column_stack((matrix, values)), [*names,*MOTION]


def month_gain(base, truth, sites, outputs):
    # Score only the supplied month using an identical roster and eligibility per arm.
    scores = block_comparisons(truth, {'baseline': base, **outputs}, sites, tuple(outputs))['scores']
    values = {name: row['readme_unclipped_score'] for name,row in scores.items()}
    return {'scores': values, 'gains': {arm: values[arm]-values['baseline'] for arm in outputs}}


def sat_screen(base, truth, satellite, sites):
    paired = base.merge(truth[POINT_KEYS+['power','power_limit']], on=POINT_KEYS, how='left',
                        validate='one_to_one', sort=False, indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError('Satellite replay forecast/label keys differ.')
    daily = paired.iloc[::96][DAY_KEYS].reset_index(drop=True)
    valid = paired.power_limit.eq(1).to_numpy().reshape(-1,96) & np.isfinite(paired.power).to_numpy().reshape(-1,96)
    delta = (paired.power-paired.prediction).to_numpy().reshape(-1,96)
    short = valid[:,:LEAD_POINTS]
    response = np.where(short,delta[:,:LEAD_POINTS],0).sum(axis=1)/short.sum(axis=1).clip(1)
    supported = np.isfinite(satellite).all(axis=1) & (valid.sum(axis=1)>=72) & (short.sum(axis=1)>=18)
    months = daily.date.str[:7]
    supported &= daily.station_id.ne(STATION).to_numpy()
    training = months.isin(SOLVE_MONTHS).to_numpy() & supported
    counts = {month:int((supported & months.eq(month).to_numpy()).sum()) for month in (*SOLVE_MONTHS,*TEST_MONTHS)}
    if any(n < SAT_MIN_DAYS for n in counts.values()):
        return {'promoted':False,'status':'insufficient_paired_days','paired_station_days':counts}
    center, scale = satellite[training].mean(axis=0), satellite[training].std(axis=0).clip(.01)
    normalized = np.clip(np.nan_to_num((satellite-center)/scale,nan=0),-4,4)
    static = np.column_stack((np.ones(len(satellite)), normalized[:,:3]))
    motion = np.column_stack((static, normalized[:,3:]))
    forecasts = {}
    for name,matrix in [('static',static),('motion',motion)]:
        x = matrix[training]
        beta = np.linalg.solve(x.T@x/len(x)+.25*np.eye(x.shape[1]),x.T@response[training]/len(x))
        correction = np.where(supported, np.clip(.5*(matrix@beta),-.75,.75), 0)
        values = base.prediction.to_numpy().reshape(-1,96).copy()
        values[:,:LEAD_POINTS] += correction[:,None]
        forecasts[name] = base.assign(prediction=values.ravel().clip(0,MAX_PREDICTION['T1']))
    metrics = {}
    for month in TEST_MONTHS:
        pick = base.date.str.startswith(month)
        metrics[month] = month_gain(base.loc[pick],truth.loc[truth.date.str.startswith(month)],sites,
                                    {key:frame.loc[pick] for key,frame in forecasts.items()})
    promoted = all(row['gains']['motion']>0 and row['scores']['motion']-row['scores']['static']>=.05
                   for row in metrics.values())
    return {'promoted':promoted,'status':'scored','paired_station_days':counts,'months':metrics}


def sustained(mask):
    keep = np.zeros_like(mask)
    blocks = np.lib.stride_tricks.sliding_window_view(mask, 4, axis=-1).all(axis=-1)
    for i in range(4):
        keep[..., i:i+blocks.shape[-1]] |= blocks
    return keep


def isolated_sources(raw):
    complete = np.isfinite(raw).all(axis=0)
    excluded = []
    for source in range(3):
        others = raw[[i for i in range(3) if i != source]]
        excluded.append(sustained(complete & (np.abs(others[0]-others[1])<=2.)
                                  & (np.abs(raw[source]-others.mean(axis=0))>=8.)))
    return np.stack(excluded)


def source_weather(args, train, days, sites):
    n = len(train)
    data, columns, stamp = archived_issue_weather(args.workdir,args.data_root,train+days,sites,print)
    data = data.reshape(n+len(days),len(sites),96,len(columns))
    truth = wind_labels(args.data_root,train,sites)
    wind = truth.wind.to_numpy(dtype=float).reshape(n,len(sites),96)
    power = truth.power.to_numpy(dtype=float).reshape(wind.shape)
    valid = truth.power_limit.eq(1).to_numpy().reshape(wind.shape)
    curves = [observed_curve(wind[:,i],power[:,i],valid[:,i]) for i in range(len(sites))]
    estimates = corrected_sources(data,columns,wind,slice(0,n),slice(n,None))
    raw = np.stack([np.hypot(data[n:,...,columns.index(f'{source}_u100')],
                             data[n:,...,columns.index(f'{source}_v100')]) for source in SOURCES])
    exclude = isolated_sources(raw)
    exclude[:,:,sites.index(STATION)] = False
    baseline = finite_mean(estimates)
    survivors = finite_mean(np.where(exclude,np.nan,estimates))
    fallback = np.where(np.isfinite(survivors), survivors, baseline)
    fixed = mapped(baseline,curves)
    alternative = mapped(fallback,curves)
    return {'baseline_branch':fixed,'candidate_branch':alternative}, exclude.any(axis=0), stamp


def source_screen(base, truth, replacements, flagged):
    cohort = base.merge(truth[POINT_KEYS+['power','power_limit']],on=POINT_KEYS,how='left',
                        validate='one_to_one',sort=False,indicator=True)
    if not cohort['_merge'].eq('both').all():
        raise ValueError('Source replay forecast/label keys differ.')
    usable = cohort.power_limit.eq(1) & np.isfinite(cohort.power)
    site_days = cohort.iloc[::96][DAY_KEYS].reset_index(drop=True)
    relevant = (flagged.reshape(-1,96) & usable.to_numpy().reshape(-1,96)).any(axis=1)
    eligible = usable.to_numpy().reshape(-1,96).sum(axis=1)>=MIN_VALID_POINTS
    delta = .20*(replacements['candidate_branch']-replacements['baseline_branch']).ravel()
    candidate = base.assign(prediction=(base.prediction+delta).clip(0,MAX_PREDICTION['T1']))
    metrics = {}
    for month in TEST_MONTHS:
        use = relevant & eligible & site_days.date.str.startswith(month).to_numpy()
        count = int(use.sum())
        if count < SOURCE_MIN_DAYS:
            metrics[month] = {'affected_scored_days':count,'score':None}
            continue
        keys = site_days.loc[use]
        old, target, new = [item.merge(keys,on=DAY_KEYS,validate='many_to_one') for item in (base,truth,candidate)]
        left, right = station_scores(target,old), station_scores(target,new)
        scores = {site:right[site]['score']-score['score'] for site,score in left.items()}
        metrics[month] = {'affected_scored_days':count,'gain':float(np.mean(list(scores.values()))),
                          'station_deltas':scores}
    promoted = all(row.get('gain',0)>0 for row in metrics.values())
    return {'promoted':promoted,'months':metrics,'affected_points':int(flagged.sum())}
