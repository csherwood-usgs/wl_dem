"""
marconi_runup.py
csherwood@usgs.gov, 2026-11-13 (revised 2026-11-16), generated with Claude Opus 5.5

Rough waterline excursion (swash) statistics from CoastCam timestacks, mapped onto profiles of the daily
contour-map DEMs along the timestack pixel instruments.

Steps, for each timestack (*.ras.tiff, rows = time at 2 Hz, columns = pixel-instrument pixels):
1. Pick the .pix file valid at the timestack time from the ShoreScan camera_settings.json
   (read_camera_settings, utils_runup.get_coordinate_file_for_timestamp; .pix files are found by name in pix_dir),
   and split the .pix file (utils_runup.split_and_save_pixcoordinates, once per .pix file,
   YAML transect files) and the timestack (utils_runup.process_and_split_image, PNG transect images) into transects.
2. Rectify each transect's U, V pixels onto that day's DEM (dem_YYYY-MM-DD.tif) with the ShoreScan CIRN routines
   (utils_CIRN.dist_uv_to_xyz at z = 0, then utils_CIRN.get_elevations, ray marching to the DEM surface), using
   the calibration valid at the timestack time (marconi_scan.calib_files).
3. Classify each timestack pixel as wet (water or foam) or dry sand with the color index c = (R - B) / (R + G + B):
   sand is yellowish (c > 0), foam is near neutral (c ~ 0), water is bluish (c < 0). The dry-sand value c_sand
   comes from the landward end of the transect (75th percentile of its pixels in each row, median filtered in
   time), and a pixel is wet where c < wet_frac * c_sand (halfway to neutral by default). The instantaneous
   waterline is the last wet pixel before the first run of n_dry dry pixels, searching from the seaward end.
   A single Otsu threshold does not work here: with sand, foam and water it often splits water from foam + sand.
4. Map the waterline time series onto the profile, z(t), and report min, 2nd percentile, median, 98th percentile,
   max, and 4 std(z) (a rough significant swash), plus excursions above the measured water level.

Timestacks are skipped (with the reason in the status column) when there is no calibration, no .pix file, or the
timestack width differs from the number of .pix pixels. No calibration is expected for 2025-01-04 to 2025-01-07: at
CACO03 a guy wire broke on 2025-01-04 and the cameras moved, then moved again when it was reattached on 2025-01-07.

This is a quick estimate of the swash range, not a runup analysis (the SegFormer workflow does that).
The 98th percentile is of the whole waterline series, like eta2 in utils_runup.runupStatistics_CHI,
not of the runup maxima (R2).
"""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import cv2
import yaml
import rioxarray
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from PIL import Image

import utils_CIRN
import utils_runup
import marconi_scan as ms

RUNUP = {'jump': 100,          # pixel distance between .pix rows that starts a new transect (utils_runup threshold)
         'jump_rel': None,     # optional: also split at jumps > jump_rel * median step (None = jump only, as ShoreScan)
         'dark': 40.,          # mean gray (0-255) below which a timestack is skipped (night)
         'blur': 5,            # median filter (px) on the color index before thresholding
         'land_frac': 0.1,     # landward fraction of the transect used as the dry-sand reference
         't_sand': 61,         # median filter (samples) on the dry-sand reference
         'wet_frac': 0.5,      # wet where c < wet_frac * c_sand
         'c_min': 0.02,        # skip a transect if the dry-sand c is below this (sand and foam not separable)
         'n_dry': 5,           # dry pixels in a row that mark the landward edge of the swash
         't_med': 5,           # median filter (samples) on the waterline time series
         'max_error': 0.05,    # ray-marching tolerance (m) in utils_CIRN.get_elevations
         'wl_cols': ['water_level', 'setup', 'R2'],   # columns of image_scan.csv interpolated to timestack times
         'survey_days': ['2025-01-23', '2025-03-06'],
         'hz': 2.}

STAT_COLS = ['z_min', 'z_p02', 'z_med', 'z_p98', 'z_max']


def out_cols(cfg=RUNUP):
    """Fixed column order of runup_excursion.csv (rows of skipped timestacks leave most of them blank)."""
    return (['time', 'station', 'camera', 'path', 'eo_file', 'pix_file'] + cfg['wl_cols'] +
            ['transect', 'flipped', 'c_sand', 'c_sea', 'status', 'n', 'frac_z', 'clip_sea', 'clip_land'] +
            STAT_COLS + ['z_mean', 'S_sig', 's_min', 'x_min', 'y_min', 's_med', 'x_med', 'y_med',
                         's_max', 'x_max', 'y_max', 'beta_swash'])


def append_csv(rows, csv, cols):
    """Append rows (list of dicts) to csv with a fixed column order."""
    pd.DataFrame(rows).reindex(columns=cols).to_csv(csv, mode='a', header=not Path(csv).exists(), index=False,
                                                    date_format='%Y-%m-%d %H:%M:%S')


# ------------------------------------------------------------------ files
def list_timestacks(root, stations, cams=('c1', 'c2'), t0=None, t1=None):
    """Table of timestacks (*.ras.tif*) under root/station/cam/ (any depth), sorted by time. The station is taken
    from the file name (e.g. ...GMT.2025.CACO03.c1.ras.tiff), as in ShoreScan, not from the folder."""
    rows = [(ms.image_time(p), ms.station_name(p.name.split('.')[6]), cam, str(p))
            for st in stations for cam in cams for p in Path(root, st, cam).rglob('*.ras.tif*')]
    df = pd.DataFrame(rows, columns=['time', 'station', 'camera', 'path']).sort_values(['time', 'camera'])
    if t0 is not None:
        df = df[df.time >= pd.Timestamp(t0)]
    if t1 is not None:
        df = df[df.time < pd.Timestamp(t1) + pd.Timedelta('1D')]
    return df.reset_index(drop=True)


def split_threshold(pix_path, cfg=RUNUP):
    """Jump (px) between consecutive .pix rows that starts a new transect: cfg['jump'] (ShoreScan's rule), or with
    cfg['jump_rel'] set, min(jump, jump_rel * median step), which also splits transects whose ends are close together."""
    if cfg['jump_rel'] is None:
        return cfg['jump']
    p = np.loadtxt(pix_path)
    return min(cfg['jump'], cfg['jump_rel'] * np.median(np.hypot(*np.diff(p, axis=0).T)))


def split_pix(pix_path, yaml_dir, reverse, cfg=RUNUP):
    """U, V arrays for each transect of a .pix file (split with utils_runup; YAML files written once)."""
    stem = Path(pix_path).stem
    if not list(Path(yaml_dir).glob(f'{stem}_transect*.yaml')):
        utils_runup.split_and_save_pixcoordinates(str(pix_path), str(yaml_dir), reverse_data=reverse,
                                                  threshold=split_threshold(pix_path, cfg))
    out = {}
    for f in sorted(Path(yaml_dir).glob(f'{stem}_transect*.yaml')):
        d = yaml.safe_load(open(f))
        out[int(f.stem.split('_transect')[-1])] = (np.array(d['U'].split(','), dtype=float),
                                                   np.array(d['V'].split(','), dtype=float))
    return out


def split_stack(ts_path, pix_path, png_dir, reverse, cfg=RUNUP):
    """Paths of the transect PNGs of one timestack (split with utils_runup if not done already)."""
    base = Path(ts_path).name.rsplit('.', 1)[0]
    pngs = sorted(Path(png_dir).glob(f'{base}_transect*.png'))
    if not pngs:
        utils_runup.process_and_split_image(str(ts_path), str(pix_path), output_path=str(png_dir),
                                            reverse_data=reverse, threshold=split_threshold(pix_path, cfg))
        pngs = sorted(Path(png_dir).glob(f'{base}_transect*.png'))
    return {int(p.stem.split('_transect')[-1]): p for p in pngs}


# ------------------------------------------------------------------ profiles
def pix_profile(U, V, io, eo, dem, cfg=RUNUP):
    """x, y, z (UTM, m) of transect pixels on the DEM (z NaN where the ray misses it) and distance s (m) along the
    transect (its principal axis), positive seaward from the foot of the camera.
    dem is a rioxarray DataArray (or None: z = NaN, x, y at z = 0)."""
    tr = {'xyz': utils_CIRN.dist_uv_to_xyz(io, eo, U, V, 'z', 0.),
          'local_grid_origin': np.array([eo['x'], eo['y']]), 'local_grid_angle': 0.}
    z = np.full(U.size, np.nan)
    if dem is not None:
        tr, _ = utils_CIRN.get_elevations(dem, eo, tr, max_error=cfg['max_error'])
        zd = utils_CIRN.get_interp_dem(dem)((tr['xyz'][:, 0], tr['xyz'][:, 1]))
        hit = np.abs(zd - tr['xyz'][:, 2]) <= 2 * cfg['max_error']
        z = np.where(hit, zd, np.nan)
    x, y = tr['xyz'][:, 0], tr['xyz'][:, 1]
    p = np.column_stack([x, y]) - [eo['x'], eo['y']]
    q = p[np.isfinite(z)] if np.isfinite(z).sum() > 2 else p
    d = np.linalg.svd(q - q.mean(0))[2][0]
    d *= np.sign(d @ q.mean(0))
    return pd.DataFrame({'i': np.arange(U.size), 'u': U, 'v': V, 'x': x, 'y': y, 'z': z, 's': p @ d})


def read_dem(dem_dir, day):
    """Daily DEM as a rioxarray DataArray, or None if there is no file."""
    p = Path(dem_dir, f'dem_{day:%Y-%m-%d}.tif')
    return rioxarray.open_rasterio(p, masked=True) if p.exists() else None


# ------------------------------------------------------------------ waterline in the timestack
def waterline_index(I, cfg=RUNUP):
    """Waterline column (float) per time row of an RGB timestack whose column 0 is seaward, the wet mask, and the
    median color index of the dry-sand reference and of the seaward end."""
    f = I.astype(np.float32)
    c = cv2.medianBlur((f[..., 0] - f[..., 2]) / (f.sum(-1) + 1), cfg['blur'])
    nl = max(5, int(cfg['land_frac'] * c.shape[1]))
    c_sand = median_filter(np.percentile(c[:, -nl:], 75, axis=1), cfg['t_sand'], mode='nearest')
    wet = c < cfg['wet_frac'] * c_sand[:, None]
    n = cfg['n_dry']
    run = cv2.filter2D((~wet).astype(np.float32), -1, np.ones((1, n), np.float32),
                       anchor=(0, 0), borderType=cv2.BORDER_CONSTANT) > n - 0.5     # n dry px from column j
    run[:, -n + 1:] = False
    ri = np.where(run.any(1), run.argmax(1) - 1., I.shape[1] - 1.)
    return (median_filter(np.clip(ri, 0, None), cfg['t_med'], mode='nearest'), wet,
            np.median(c_sand), np.median(c[:, :nl]))


def excursion(ri, prof, nx, hz=RUNUP['hz']):
    """Waterline x, y, z time series on the profile and summary statistics."""
    x, y, z, s = (np.interp(ri, prof.i, prof[k]) for k in ['x', 'y', 'z', 's'])
    ok = np.isfinite(z)
    st = {'n': ri.size, 'frac_z': ok.mean(), 'clip_sea': np.mean(ri <= 0), 'clip_land': np.mean(ri >= nx - 1)}
    if ok.sum() > 10:
        q = np.percentile(z[ok], [0, 2, 50, 98, 100])
        st |= dict(zip(STAT_COLS, q)) | {'z_mean': z[ok].mean(), 'S_sig': 4 * z[ok].std()}
        for k, v in zip(['min', 'med', 'max'], q[[0, 2, 4]]):
            j = np.nanargmin(np.abs(z - v))
            st |= {f's_{k}': s[j], f'x_{k}': x[j], f'y_{k}': y[j]}
        sw = (prof.z >= q[1]) & (prof.z <= q[3])
        if sw.sum() > 3:
            st['beta_swash'] = -np.polyfit(prof.s[sw], prof.z[sw], 1)[0]
    return pd.DataFrame({'t': np.arange(ri.size) / hz, 'ri': ri, 'x': x, 'y': y, 'z': z}), st


# ------------------------------------------------------------------ pixel instruments
def read_camera_settings(path, pix_dir=None):
    """ShoreScan camera settings {station: {camera: {'reverse_flag', 'coordinate_files': {(start, end): .pix path}}}},
    parsed as in utils_runup.get_camera_settings (without its file dialog). The .pix files are found by name in pix_dir
    (default: the folder of the JSON file); folders in the JSON (e.g. DATA/) are ignored."""
    cs = json.load(open(path))
    d = Path(pix_dir or Path(path).parent)
    for st in cs.values():
        for c in st.values():
            c['coordinate_files'] = {tuple(datetime.fromisoformat(t) for t in k.split('|')): str(d / Path(v).name)
                                     for k, v in c['coordinate_files'].items()}
    return cs


def pix_for(r, cam_set):
    """(.pix path or None, reverse flag) valid for a timestack row (station, camera, time)."""
    c = cam_set.get(r.station, {}).get(r.camera)
    f = utils_runup.get_coordinate_file_for_timestamp(c, r.time.to_pydatetime()) if c else None
    return (Path(f) if f else None), (c or {}).get('reverse_flag', False)


def width_survey(ts, cam_set):
    """ts with the .pix file, its number of pixels, and the timestack width (from the TIFF header only;
    NaN for a TIFF that cannot be read)."""
    npix, rows = {}, []
    for r in ts.itertuples():
        p, _ = pix_for(r, cam_set)
        if p is not None and p not in npix:
            npix[p] = len(np.loadtxt(p))
        try:
            w = Image.open(r.path).size[0]
        except OSError:          # corrupt timestack file
            w = np.nan
        rows.append((p.name if p else None, npix.get(p, np.nan), w))
    return ts.assign(**dict(zip(['pix_file', 'n_pix', 'width'], zip(*rows))))


# ------------------------------------------------------------------ driver
def process_all(ts, calib_dir, calib_table, cam_set, dem_dir, scan, out_dir, cfg=RUNUP, fig_dir=None, fig_every=0):
    """Process timestack table ts (from list_timestacks); appends to out_dir/runup_excursion.csv and
    out_dir/runup_profiles.csv and resumes if rerun. Waterline series go to out_dir/timestacks/series.
    cam_set: camera settings from read_camera_settings (date-dependent .pix files).
    With fig_dir and fig_every > 0, a figure for every fig_every-th timestack transect."""
    png_dir, yaml_dir, ser_dir = (Path(out_dir, 'timestacks', d) for d in ['png', 'pix_yaml', 'series'])
    for d in [png_dir, yaml_dir, ser_dir]:
        d.mkdir(parents=True, exist_ok=True)
    csv, pcsv = Path(out_dir, 'runup_excursion.csv'), Path(out_dir, 'runup_profiles.csv')
    done = set(pd.read_csv(csv).path) if csv.exists() else set()
    pdone = set(pd.read_csv(pcsv, usecols=['key']).key) if pcsv.exists() else set()
    sc = {c: g.sort_values('time') for c, g in scan.groupby('camera')}
    pix, profs, cal, dems, k = {}, {}, {}, {}, 0
    cols = out_cols(cfg)
    for r in ts[~ts.path.isin(done)].itertuples():
        eo_f, io_f = ms.calib_files(calib_table, r.time, r.station, r.camera)
        pix_path, rev = pix_for(r, cam_set)
        rec = {'time': r.time, 'station': r.station, 'camera': r.camera, 'path': r.path, 'eo_file': eo_f,
               'pix_file': pix_path.name if pix_path else None}
        g = sc.get(r.camera)
        if g is not None:
            tt = g.time.values.astype('int64')
            rec |= {c: np.interp(np.int64(r.time.value), tt, g[c]) for c in cfg['wl_cols'] if c in g}
        why = 'no calibration' if eo_f is None else 'no pix file' if pix_path is None else None
        if why is None:
            if pix_path not in pix:
                pix[pix_path] = split_pix(pix_path, yaml_dir, rev, cfg)
            npix = sum(u.size for u, _ in pix[pix_path].values())
            I0 = cv2.imread(r.path)
            why = ('unreadable' if I0 is None else f'width {I0.shape[1]} != {npix} pix' if I0.shape[1] != npix
                   else 'dark' if I0.mean() < cfg['dark'] else None)
        if why:
            append_csv([rec | {'status': why}], csv, cols)
            continue
        if eo_f not in cal:
            cal[eo_f] = (ms.read_yaml(Path(calib_dir, io_f)), ms.read_yaml(Path(calib_dir, eo_f)))
        io, eo = cal[eo_f]
        day = r.time.floor('D')
        if day not in dems:
            dems = {day: read_dem(dem_dir, day)}           # keep one day in memory
        pngs = split_stack(r.path, pix_path, png_dir, rev, cfg)
        rows = []
        for tn, (U, V) in pix[pix_path].items():
            key = f'{day:%Y-%m-%d}|{eo_f}|{pix_path.stem}|{tn}'
            if key not in profs:
                profs[key] = pix_profile(U, V, io, eo, dems[day], cfg)
                if key not in pdone:
                    profs[key].assign(key=key, day=f'{day:%Y-%m-%d}', camera=r.camera, transect=tn).to_csv(
                        pcsv, mode='a', header=not pcsv.exists(), index=False)
                    pdone.add(key)
            prof = profs[key]
            I = cv2.cvtColor(cv2.imread(str(pngs[tn])), cv2.COLOR_BGR2RGB)
            flip = prof.s.iloc[0] < prof.s.iloc[-1]          # put the seaward end first
            if flip:
                I, prof = I[:, ::-1], prof.iloc[::-1].assign(i=np.arange(len(prof)))
            ri, wet, c_sand, c_sea = waterline_index(I, cfg)
            rec_t = rec | {'transect': tn, 'flipped': flip, 'c_sand': c_sand, 'c_sea': c_sea}
            if c_sand < cfg['c_min']:
                rows.append(rec_t | {'status': 'low contrast'})
                continue
            ser, st = excursion(ri, prof, I.shape[1], cfg['hz'])
            ser.to_csv(ser_dir / f'{Path(r.path).name}_transect{tn}.csv', index=False, float_format='%.3f')
            rows.append(rec_t | {'status': 'ok'} | st)
            if fig_dir is not None and fig_every and k % fig_every == 0:
                plot_stack(Path(fig_dir, f'{Path(r.path).name}_transect{tn}.png'), I, ser, prof, rows[-1])
            k += 1
        append_csv(rows, csv, cols)
    res = pd.read_csv(csv)
    res['time'] = pd.to_datetime(res.time, format='mixed')     # earlier files may mix date-only midnight times
    for c in [c for c in cfg['wl_cols'] if c in res and c != 'water_level']:
        res[f'{c}_stockdon_z'] = res.water_level + res[c]
    for c in STAT_COLS:
        res[f'eta_{c[2:]}'] = res[c] - res.water_level if 'water_level' in res else np.nan
    return res


# ------------------------------------------------------------------ figures
def plot_pix_periods(fig_dir, ts, scan, calib_table, cam_set, cfg=RUNUP, tz='America/New_York'):
    """One figure per camera, calibration period, and .pix file in ts: the .pix transects, numbered, on the sharpest
    timex image of that period closest to local noon (scan quality 4 when scan has a quality column).
    Returns the figure paths."""
    ts = ts.assign(eo_file=[ms.calib_files(calib_table, r.time, r.station, r.camera)[0] for r in ts.itertuples()],
                   pix=[str(pix_for(r, cam_set)[0]) for r in ts.itertuples()])
    paths = []
    for (c, eo, _), g in ts[ts.pix != 'None'].dropna(subset=['eo_file']).groupby(['camera', 'eo_file', 'pix']):
        w = scan[(scan.camera == c) & (scan.time >= g.time.min()) & (scan.time <= g.time.max())]
        if 'quality' in w:
            w = w[w.quality == w.quality.max()]
        if len(w) == 0:
            continue
        loc = w.time.dt.tz_localize('UTC').dt.tz_convert(tz)
        r = w.iloc[np.argmin(np.abs(loc.dt.hour + loc.dt.minute / 60 - 12))]
        pix_path, rev = pix_for(g.iloc[0], cam_set)
        p = np.loadtxt(pix_path)[::-1 if rev else 1]
        jumps = np.where(np.hypot(*np.diff(p, axis=0).T) > split_threshold(pix_path, cfg))[0] + 1
        edges = np.r_[0, jumps, len(p)]
        img = cv2.imread(r.path)
        fig, ax = plt.subplots(figsize=(12, 12 * img.shape[0] / img.shape[1]))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            ax.plot(p[a:b, 0], p[a:b, 1], '.', ms=1.5, c=f'C{k % 10}')
            ax.plot(*p[a], 'o', c=f'C{k % 10}', mfc='none', ms=7)
            ax.annotate(f'{k}', p[a], xytext=(6, 6), textcoords='offset points', color='w', fontsize=11,
                        fontweight='bold', bbox={'fc': f'C{k % 10}', 'ec': 'none', 'alpha': 0.8, 'pad': 2})
        ax.set_xlabel('U (px)')
        ax.set_ylabel('V (px)')
        ax.set_title(f'{c}, {eo}: {g.time.min():%Y-%m-%d} to {g.time.max():%Y-%m-%d}, {len(g)} timestacks\n'
                     f'{pix_path.name}: {len(edges) - 1} transects, {np.diff(edges).tolist()} px (circle = first pixel); '
                     f'image {Path(r.path).name}', fontsize=9)
        fig.tight_layout()
        path = Path(fig_dir, f'pix_transects_{c}_{Path(eo).stem}_{pix_path.stem}.png')
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_pix_check(path, ts_path, pix_path, reverse=False, cfg=RUNUP):
    """a. .pix pixels colored by transect on the timex image of the same time (if there is one),
    b. the timestack with the transect boundaries found by the split."""
    p = np.loadtxt(pix_path)[::-1 if reverse else 1]
    jumps = np.where(np.hypot(*np.diff(p, axis=0).T) > split_threshold(pix_path, cfg))[0] + 1
    edges = np.r_[0, jumps, len(p)]
    I = cv2.cvtColor(cv2.imread(str(ts_path)), cv2.COLOR_BGR2RGB)
    timex = Path(str(ts_path).replace('.ras.tiff', '.timex.jpg').replace('.ras.tif', '.timex.jpg'))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={'width_ratios': [1.3, 1]})
    if timex.exists():
        ax[0].imshow(cv2.cvtColor(cv2.imread(str(timex)), cv2.COLOR_BGR2RGB))
    else:
        ax[0].invert_yaxis()
    ax[1].imshow(I, aspect='auto')
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        ax[0].plot(p[a:b, 0], p[a:b, 1], '.', ms=1, c=f'C{k}', label=f'transect {k}: {b - a} px')
        ax[0].plot(*p[a], 'o', c=f'C{k}', mfc='none')
        ax[1].axvline(a - 0.5, c=f'C{k}', lw=1.5)
    ax[0].legend(fontsize=7, markerscale=8)
    ax[0].set_xlabel('U (px)')
    ax[0].set_ylabel('V (px)')
    ax[1].set_xlabel('Timestack column')
    ax[1].set_ylabel('Time sample')
    for a, l, c in zip(ax, 'ab', ['w' if timex.exists() else 'k', 'w']):
        a.text(0.02, 0.98, l, transform=a.transAxes, va='top', fontsize=12, color=c)
    fig.suptitle(f'{Path(ts_path).name}; {Path(pix_path).name}, split at jumps > '
                 f'{split_threshold(pix_path, cfg):.0f} px (open circle = first pixel)', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return fig, np.diff(edges)
def survey_lines(ax, t0, t1, cfg=RUNUP):
    """Thick light gray lines on the beach survey days within t0 to t1."""
    for d in pd.to_datetime(cfg['survey_days']):
        if t0 <= d <= t1:
            ax.axvline(d, c='0.85', lw=4, zorder=0)


def plot_stack(path, I, ser, prof, st):
    """a. timestack (seaward on the left) with the waterline, b. waterline elevation series,
    c. DEM profile with the swash range."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1.3, 1.3]})
    ax[0].imshow(I, aspect='auto', extent=[-0.5, I.shape[1] - 0.5, ser.t.iloc[-1], 0])
    ax[0].plot(ser.ri, ser.t, c='r', lw=0.6)
    ax[0].set_xlabel('Pixel (seaward to landward)')
    ax[0].set_ylabel('Time (s)')
    ax[1].plot(ser.t, ser.z, c='k', lw=0.6)
    refs = [('water_level', 'water level', 'b'), ('setup', 'water level + setup (Stockdon)', 'c'),
            ('R2', 'water level + R2 (Stockdon)', 'm')]
    for k, lab, c in refs:
        if k in st:
            v = st['water_level'] + (st[k] if k != 'water_level' else 0)
            ax[1].axhline(v, c=c, ls='--', lw=1, label=lab)
            ax[2].axhline(v, c=c, ls='--', lw=1)
    for k, c in zip(['z_med', 'z_p98', 'z_max'], ['g', 'orange', 'r']):
        if k in st:
            ax[1].axhline(st[k], c=c, lw=1, label=k[2:])
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Waterline elevation (m, NAVD88)')
    ax[1].legend(fontsize=7, loc='lower right')
    ax[2].plot(prof.s, prof.z, c='k')
    if 'z_med' in st:
        sw = (prof.z >= st['z_min']) & (prof.z <= st['z_max'])
        ax[2].plot(prof.s[sw], prof.z[sw], c='tab:orange', lw=4, alpha=0.6, label='swash range')
        for k, c in zip(['med', 'max'], ['g', 'r']):
            ax[2].plot(st[f's_{k}'], st[f'z_{k}'], 'o', c=c, label=k)
        ax[2].legend(fontsize=7)
    ax[2].set_xlabel('Distance along transect (m)')
    ax[2].set_ylabel('Elevation (m, NAVD88)')
    ax[2].invert_xaxis()
    for a, l in zip(ax, 'abc'):
        a.text(0.02, 0.98, l, transform=a.transAxes, va='top', fontsize=12,
               color='w' if l == 'a' else 'k')
    fig.suptitle(f"{Path(st['path']).name}, transect {st['transect']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_series(path, res, cfg=RUNUP):
    """Waterline median, 98th percentile, and max elevations with water level and Stockdon levels, one panel
    per camera and transect."""
    res = res.assign(time=pd.to_datetime(res.time, format='mixed'))
    ok = res[res.status == 'ok']
    keys = sorted(ok.groupby(['camera', 'transect']).groups)
    fig, ax = plt.subplots(len(keys), 1, figsize=(12, 2.8 * len(keys)), sharex=True, squeeze=False)
    for a, (c, tn), l in zip(ax[:, 0], keys, 'abcdefghij'):
        g = ok[(ok.camera == c) & (ok.transect == tn)].sort_values('time')
        survey_lines(a, ok.time.min(), ok.time.max(), cfg)
        a.plot(g.time, g.water_level, c='b', lw=1, label='water level')
        if 'R2_stockdon_z' in g:
            a.plot(g.time, g.R2_stockdon_z, c='m', lw=1, label='water level + R2 (Stockdon)')
        a.vlines(g.time, g.z_min, g.z_max, color='0.6', lw=1, label='min to max')
        for k, m, col in [('z_med', 'o', 'g'), ('z_p98', '^', 'orange')]:
            a.plot(g.time, g[k], m, c=col, ms=3, label=k[2:])
        a.set_ylabel('Elevation (m, NAVD88)')
        a.text(0.01, 0.97, l, transform=a.transAxes, va='top', fontsize=12)
        a.set_title(f'{c}, transect {tn}', fontsize=9)
    ax[0, 0].legend(fontsize=7, ncol=5, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def plot_day(path, res, profiles, day):
    """DEM profiles of one day with the swash range of each timestack (median and max as points,
    colored by hour), one panel per camera and transect."""
    res = res.assign(time=pd.to_datetime(res.time, format='mixed'))
    ok = res[(res.status == 'ok') & (res.time.dt.floor('D') == day)]
    pr = profiles[profiles.day == day]
    keys = sorted(ok.groupby(['camera', 'transect']).groups)
    if not keys:
        return None
    fig, ax = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.5), squeeze=False)
    for a, (c, tn), l in zip(ax[0], keys, 'abcdefghij'):
        p = pr[(pr.camera == c) & (pr.transect == tn)]
        g = ok[(ok.camera == c) & (ok.transect == tn)]
        a.plot(p.s, p.z, c='k', lw=1.5)
        hr = g.time.dt.hour + g.time.dt.minute / 60
        a.scatter(g.s_min, g.z_min, c=hr, cmap='viridis', vmin=0, vmax=24, s=20, marker='v')
        a.scatter(g.s_med, g.z_med, c=hr, cmap='viridis', vmin=0, vmax=24, s=15, marker='o')
        sc = a.scatter(g.s_max, g.z_max, c=hr, cmap='viridis', vmin=0, vmax=24, s=20, marker='^')
        a.invert_xaxis()
        a.set_xlabel('Distance along transect (m)')
        a.set_ylabel('Elevation (m, NAVD88)')
        a.set_title(f'{c}, transect {tn}', fontsize=9)
        a.text(0.02, 0.97, l, transform=a.transAxes, va='top', fontsize=12)
    fig.colorbar(sc, ax=ax[0, -1], label='Hour (UTC)')
    fig.suptitle(f'{day:%Y-%m-%d}: DEM profiles with waterline min (v), median (o), and max (^)',
                 fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return path
