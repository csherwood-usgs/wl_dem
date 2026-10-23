# csherwood@usgs.gov, 2026-10-21, generated with Claude Opus 5.5
"""
First-pass scan of Marconi CoastCam timex images: calibration lookup, image quality index,
wave forcing and Stockdon et al. (2006) runup, and two camera-pointing checks
(sea horizon vs. calibration; feature-based rotation vs. a reference image).

Quality index: 0 = night, 1 = too blurry (fog, rain, sand on lens), 2 = sharp but sunglint,
4 = sharp, high quality. -1 = unreadable (empty or corrupt) file. NaN = no calibration for that station/time.

The feature-based pointing check follows the idea in emarsjanik/marconi-camera-dem
(pointing_check.py): a camera that has only rotated relates the two views by a pure rotation,
found from matched viewing rays with the Kabsch (1976) solution inside RANSAC.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import cv2
import yaml
from scipy.optimize import least_squares
from cirn_utils import undistortUV, intrinsicsExtrinsics2P, xyz_to_dist_uv, CIRNangles2R

G = 9.81
R_EARTH = 6.371e6
K_REFRACT = 0.13          # terrestrial refraction coefficient for horizon dip

THRESH = {'dark': 40.,    # mean gray (0-255) below which image is night
          'blur': 20.,    # Laplacian variance (quarter-res, below horizon) below which image is blurry
          'glint': 0.02,  # fraction of saturated pixels in sea band above which image has sunglint
          'sat': 250,     # gray value counted as saturated
          'sea_band': 400,  # rows (full res) below horizon used for glint test
          'hz_win': 150,  # +/- rows searched for the horizon edge
          'hz_ref': 2.,   # max horizon-fit residual (px) for an image to serve as pointing reference
          'pt_every': pd.Timedelta('60min'),  # min time between pointing checks per camera
          'pt_scale': 0.5,  # image scale for SIFT
          'pt_min_inliers': 40}


# ------------------------------------------------------------------ files and calibration
def image_time(path):
    """UTC time from the epoch seconds that start the file name."""
    return pd.Timestamp(int(Path(path).name.split('.')[0]), unit='s')


def station_name(s):
    """Station name as used in the calibration table: caco-03, CACO.03, caco03 -> CACO03."""
    return re.sub(r'[^A-Z0-9]', '', s.upper())


def list_timex(root, stations, cams=('c1', 'c2'), t0=None, t1=None):
    """Table of timex images anywhere under root/station/cam/ (e.g. cam/2024/352_Dec.17/), sorted by time."""
    rows = [(image_time(p), station_name(st), cam, str(p))
            for st in stations for cam in cams for p in Path(root, st, cam).rglob('*.timex.*')]
    df = pd.DataFrame(rows, columns=['time', 'station', 'camera', 'path']).sort_values(['time', 'camera'])
    if t0 is not None:
        df = df[(df.time >= t0) & (df.time <= t1)]
    return df.reset_index(drop=True)


def read_calib_table(path):
    """Calibration lookup table with columns start, end, c1, c1_EO, c1_IO, c2, c2_EO, c2_IO."""
    t = pd.read_csv(path).iloc[:, :8]
    t.columns = ['start', 'end', 'c1', 'c1_EO', 'c1_IO', 'c2', 'c2_EO', 'c2_IO']
    t = t.dropna(subset=['start'])
    for c in ['c1', 'c1_EO', 'c1_IO', 'c2', 'c2_EO', 'c2_IO']:
        t[c] = t[c].str.strip()
    t['start'] = pd.to_datetime(t.start, format='mixed')    # accepts 2/10/2021 and 2021-02-10
    t['end'] = pd.to_datetime(t.end, format='mixed').fillna(pd.Timestamp('2100-01-01'))
    return t.reset_index(drop=True)


def calib_files(table, time, station, cam):
    """(EO, IO) file names valid for station_cam at time (start <= time < end), else (None, None)."""
    ok = (table.start <= time) & (time < table.end) & (table[cam] == f'{station}_{cam}')
    if not ok.any():
        return None, None
    r = table[ok].iloc[-1]
    return r[f'{cam}_EO'], r[f'{cam}_IO']


def read_yaml(path):
    return yaml.safe_load(open(path))


# ------------------------------------------------------------------ waves and runup
def wavenumber(T, h):
    """Linear dispersion: Fenton and McKee (1990) explicit estimate plus three Newton steps."""
    w = 2 * np.pi / T
    k0 = w**2 / G
    k = k0 / np.tanh((k0 * h)**0.75)**(2 / 3)
    for _ in range(3):
        f = G * k * np.tanh(k * h) - w**2
        k = k - f / (G * np.tanh(k * h) + G * k * h / np.cosh(k * h)**2)
    return k


def deshoal(H, T, h):
    """Deep-water height H0 = H sqrt(Cg/Cg0), linear shoaling only (no refraction)."""
    k = wavenumber(T, h)
    kh = k * h
    cg = 0.5 * (1 + 2 * kh / np.sinh(2 * kh)) * (2 * np.pi / T) / k
    return H * np.sqrt(cg / (G * T / (4 * np.pi)))


def stockdon(H0, Tp, beta):
    """Setup and 2% runup (m) from Stockdon et al. (2006), eqs. 10, 18, 19 (dissipative if xi0 < 0.3)."""
    L0 = G * Tp**2 / (2 * np.pi)
    HL = np.sqrt(H0 * L0)
    xi0 = beta / np.sqrt(H0 / L0)
    setup = np.where(xi0 < 0.3, 0.016 * HL, 0.35 * beta * HL)
    R2 = np.where(xi0 < 0.3, 0.043 * HL,
                  1.1 * (0.35 * beta * HL + 0.5 * np.sqrt(H0 * L0 * (0.563 * beta**2 + 0.004))))
    return setup, R2


def wave_forcing(waves, times, beta=0.12):
    """Waves and water level interpolated to times, with H0, setup, and R2 (NaN outside record)."""
    x = waves.time.values.astype('datetime64[s]').astype('int64')
    xi = pd.DatetimeIndex(times).values.astype('datetime64[s]').astype('int64')
    out = pd.DataFrame({c: np.interp(xi, x, waves[c].values, left=np.nan, right=np.nan)
                        for c in ['wh_4061', 'wp_peak', 'water_level', 'water_depth']})
    out['wvdir'] = np.interp(xi, x, np.unwrap(np.radians(waves.wvdir.values)), left=np.nan, right=np.nan)
    out['wvdir'] = np.degrees(out.wvdir) % 360
    out['H0'] = deshoal(out.wh_4061, out.wp_peak, out.water_depth)
    out['setup'], out['R2'] = stockdon(out.H0, out.wp_peak, beta)
    return out


# ------------------------------------------------------------------ sun
def sun_position(times, lat, lon):
    """Solar elevation and azimuth (deg, azimuth CW from north) for UTC times
    (Astronomical Almanac low-precision algorithm; Michalsky, 1988)."""
    n = pd.DatetimeIndex(times).to_julian_date().values - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = np.radians(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 4e-7 * n)
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    ha = np.radians(((18.697374558 + 24.06570982441908 * n) % 24) * 15 + lon) - ra
    la = np.radians(lat)
    el = np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha))
    az = np.arctan2(-np.sin(ha), np.tan(dec) * np.cos(la) - np.sin(la) * np.cos(ha))
    return np.degrees(el), np.degrees(az) % 360


# ------------------------------------------------------------------ projection helpers
def project(io, eo, xyz):
    """Distorted (U, V) of world points, and a flag for points in front of the camera whose
    undistorted position is near the frame (where the lens model is valid)."""
    UVd, flag = xyz_to_dist_uv(io, eo, xyz)
    P = intrinsicsExtrinsics2P(io, eo)[0]
    q = P @ np.vstack([xyz.T, np.ones(len(xyz))])
    u, v = q[0] / q[2], q[1] / q[2]
    ok = (flag > 0) & (u > -0.1 * io['NU']) & (u < 1.1 * io['NU']) & (v > -0.1 * io['NV']) & (v < 1.1 * io['NV'])
    return UVd[0], UVd[1], ok


def horizon_rows(io, eo, cols, D=20000.):
    """Predicted sea-horizon row at image columns, including dip with refraction, else NaN."""
    dip = np.sqrt(2 * eo['z'] * (1 - K_REFRACT) / R_EARTH)
    az = np.radians(eo['azimuth'] + np.linspace(-80, 80, 2000))
    xyz = np.column_stack([eo['x'] + D * np.sin(az), eo['y'] + D * np.cos(az),
                           np.full(az.size, eo['z'] - D * np.tan(dip))])
    U, V, ok = project(io, eo, xyz)
    o = np.argsort(U[ok])
    return np.interp(cols, U[ok][o], V[ok][o], left=np.nan, right=np.nan)


def observed_horizon(gray, cols, pred, win):
    """Row of the strongest vertical gradient within +/- win of the predicted row, and its strength
    (after the observed_horizon idea in marconi-camera-dem/estimate_eo_rotation.py)."""
    gy = np.abs(cv2.Sobel(cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.5), cv2.CV_32F, 0, 1, ksize=5))
    row, strength = np.full(len(cols), np.nan), np.full(len(cols), np.nan)
    for i, (c, r0) in enumerate(zip(cols, pred)):
        if np.isfinite(r0):
            a, b = int(max(0, r0 - win)), int(min(gray.shape[0], r0 + win))
            prof = gy[a:b, c - 6:c + 6].mean(axis=1)
            row[i], strength[i] = a + np.argmax(prof), prof.max()
    return row, strength


def fit_horizon(io, eo, cols, obs):
    """Tilt and roll changes (deg) that move the predicted horizon onto obs (robust least squares),
    and the median absolute residual (px)."""
    ok = np.isfinite(obs)
    def res(p):
        e = dict(eo, tilt=eo['tilt'] + p[0], roll=eo['roll'] + p[1])
        return np.nan_to_num(horizon_rows(io, e, cols[ok]) - obs[ok], nan=200.)
    s = least_squares(res, [0., 0.], loss='soft_l1', f_scale=5., diff_step=1e-3)
    return s.x, np.median(np.abs(s.fun))


def sun_glint_in_view(io, eo, sun_el, sun_az, margin=0.5):
    """True if the specular reflection of the sun falls within the frame expanded by margin."""
    if sun_el <= 0:
        return False
    e, a = np.radians(sun_el), np.radians(sun_az)
    d = 1000.
    xyz = np.array([[eo['x'] + d * np.cos(e) * np.sin(a), eo['y'] + d * np.cos(e) * np.cos(a), eo['z'] - d * np.sin(e)]])
    P, _, R, IC = intrinsicsExtrinsics2P(io, eo)
    q = P @ np.r_[xyz[0], 1.]
    u, v = q[0] / q[2], q[1] / q[2]
    front = (R @ IC @ np.r_[xyz[0], 1.])[2] > 0
    return bool(front and -margin * io['NU'] < u < (1 + margin) * io['NU'] and -margin * io['NV'] < v < (1 + margin) * io['NV'])


# ------------------------------------------------------------------ feature-based pointing
def rays(u, v, io):
    """Distorted pixels -> unit viewing rays in camera coordinates (CIRN sign convention)."""
    U, V = undistortUV(u, v, io)
    r = np.column_stack([(io['coU'] - U) / io['fx'], (io['coV'] - V) / io['fy'], np.ones_like(U)])
    return r / np.linalg.norm(r, axis=1)[:, None]


def R_to_angles(R):
    """Inverse of CIRNangles2R -> azimuth, tilt, roll (deg)."""
    tilt = np.arccos(np.clip(-R[2, 2], -1, 1))
    az = np.arctan2(R[2, 0], R[2, 1])
    roll = np.arctan2(-R[0, 2], R[1, 2])
    return np.degrees(az) % 360, np.degrees(tilt), np.degrees(roll)


def kabsch(world, cam):
    """Rotation R minimizing sum |cam - R world|^2 for unit vectors (Kabsch, 1976)."""
    U, _, Vt = np.linalg.svd(world.T @ cam)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def ang_err(R, world, cam):
    return np.degrees(np.arccos(np.clip(np.sum((world @ R.T) * cam, axis=1), -1, 1)))


def solve_rotation(world, cam, thresh=0.1, iters=1000, seed=0):
    """RANSAC on pairs of rays, then least-squares refit on inliers."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        inl = ang_err(kabsch(*(x[rng.choice(len(world), 2, replace=False)] for x in (world, cam))), world, cam) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    for _ in range(3):
        R = kabsch(world[best], cam[best])
        best = ang_err(R, world, cam) < thresh
    return R, best


def features(gray, mask, scale):
    """SIFT keypoints (full-res pixel coordinates) and descriptors on a CLAHE-equalized image."""
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    kp, des = cv2.SIFT_create(nfeatures=5000).detectAndCompute(g, m)
    return np.array([k.pt for k in kp]).reshape(-1, 2) / scale, des


def land_mask(shape, hz_full, margin=40):
    """Mask (uint8) of pixels below the horizon + margin, excluding sky with moving clouds."""
    hz = np.interp(np.arange(shape[1]), np.linspace(0, shape[1] - 1, len(hz_full)), np.nan_to_num(hz_full))
    return ((np.arange(shape[0])[:, None] > hz[None, :] + margin) * 255).astype(np.uint8)


def pointing(ref, pts, des, io, ratio=0.75):
    """Rotation of the new view relative to the reference: returns (n_inliers, az, tilt, roll)."""
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ref['des'], des, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(good) < 8:
        return 0, np.nan, np.nan, np.nan
    p1 = ref['pts'][[m.queryIdx for m in good]]
    p2 = pts[[m.trainIdx for m in good]]
    world = rays(p1[:, 0], p1[:, 1], io) @ ref['R']
    R, inl = solve_rotation(world, rays(p2[:, 0], p2[:, 1], io))
    return int(inl.sum()), *R_to_angles(R)


# ------------------------------------------------------------------ the scan
def classify(brightness, sharpness, glint_frac, th):
    """Quality index from image metrics (glint_view, the sun-geometry flag, is kept as a diagnostic only)."""
    if brightness < th['dark']:
        return 0
    if sharpness < th['blur']:
        return 1
    if glint_frac > th['glint']:
        return 2
    return 4


def scan_image(path, io, eo, sun_el, sun_az, th):
    """Quality metrics and horizon check for one image. Returns (dict, gray image, predicted horizon)."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:                     # empty or corrupt file
        return {'quality': -1, 'sun_el': sun_el, 'sun_az': sun_az}, None, None
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    r = {'brightness': small.mean(), 'sun_el': sun_el, 'sun_az': sun_az}
    if r['brightness'] < th['dark']:
        r['quality'] = 0
        return r, gray, None
    cols = np.arange(100, gray.shape[1] - 100, 50)
    pred = horizon_rows(io, eo, cols)
    obs, strength = observed_horizon(gray, cols, pred, th['hz_win'])
    (dtilt, droll), fres = fit_horizon(io, eo, cols, obs)
    hz_s = np.interp(np.arange(small.shape[1]) * 4, cols, np.nan_to_num(pred)) / 4
    below = np.arange(small.shape[0])[:, None] > hz_s[None, :]
    band = below & (np.arange(small.shape[0])[:, None] < hz_s[None, :] + th['sea_band'] / 4)
    lap = cv2.Laplacian(small, cv2.CV_64F)
    r.update(sharpness=lap[below].var(), glint_frac=(small[band] >= th['sat']).mean(),
             glint_view=sun_glint_in_view(io, eo, sun_el, sun_az),
             hz_offset=np.nanmedian(obs - pred), hz_strength=np.nanmedian(strength),
             hz_dtilt=dtilt, hz_droll=droll, hz_fit_px=fres)
    r['quality'] = classify(r['brightness'], r['sharpness'], r['glint_frac'], th)
    return r, gray, pred


def scan(imgs, waves, table, calib_dir, out_csv, lat, lon, beta=0.12, th=THRESH, checkpoint=200):
    """Scan images (from list_timex) in time order: calibration files, waves and runup, quality metrics and
    horizon check. Writes/appends rows to out_csv and resumes by skipping paths already in it.
    Pointing columns are filled afterward by pointing_pass()."""
    done = pd.read_csv(out_csv, parse_dates=['time']) if Path(out_csv).exists() else pd.DataFrame(columns=['path'])
    imgs = imgs[~imgs.path.isin(done.path)].reset_index(drop=True)
    imgs = pd.concat([imgs, wave_forcing(waves, imgs.time, beta)], axis=1)
    imgs['sun_el'], imgs['sun_az'] = sun_position(imgs.time, lat, lon)
    cal, rows = {}, []
    for i, im in imgs.iterrows():
        eo_file, io_file = calib_files(table, im.time, im.station, im.camera)
        row = im.to_dict() | {'eo_file': eo_file, 'io_file': io_file, 'quality': np.nan}
        if eo_file is not None:
            if eo_file not in cal:
                cal[eo_file] = (read_yaml(Path(calib_dir, io_file)), read_yaml(Path(calib_dir, eo_file)))
            row |= scan_image(im.path, *cal[eo_file], im.sun_el, im.sun_az, th)[0]
        rows.append(row)
        if len(rows) == checkpoint or i == len(imgs) - 1:
            write_rows(rows, out_csv)
            print(f'{i + 1}/{len(imgs)}  {im.time}')
            rows = []
    return pd.read_csv(out_csv, parse_dates=['time'])


def pointing_pass(df, calib_dir, th=THRESH):
    """Camera rotation (d_az, d_tilt, d_roll, deg) of quality-4 images, at most one per th['pt_every'] per camera,
    relative to a reference image for each camera and EO file: the first quality-4 image whose horizon is
    found cleanly (hz_fit_px < th['hz_ref']), flagged is_ref. The reference need not match the calibration;
    its own horizon fit (hz_dtilt, hz_droll) tells how well the calibration fits at that time.
    Returns df with the pointing columns replaced."""
    df = df.copy()
    for c in ['pt_ref', 'pt_inliers', 'd_az', 'd_tilt', 'd_roll']:
        df[c] = np.nan
    df['pt_ref'] = df.pt_ref.astype(object)
    df['is_ref'] = False
    for (cam, eo_file), g in df[df.quality == 4].sort_values('time').groupby(['camera', 'eo_file']):
        io, eo = read_yaml(Path(calib_dir, g.io_file.iloc[0])), read_yaml(Path(calib_dir, eo_file))
        R0 = CIRNangles2R(*np.radians([eo['azimuth'], eo['tilt'], eo['roll']]))
        mask = None
        cand = g[g.hz_fit_px < th['hz_ref']]
        if len(cand) == 0:
            print(f'{cam} {eo_file}: no image with a clean horizon for a reference')
            continue
        ri = cand.index[0]
        keep = g.time.dt.floor(th['pt_every']).drop_duplicates().index.union([ri])
        ref = None
        for i in [ri] + [k for k in keep if k != ri]:
            gray = cv2.imread(df.path[i], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = land_mask(gray.shape, horizon_rows(io, eo, np.arange(0, gray.shape[1], 50)))
            pts, des = features(gray, mask, th['pt_scale'])
            if ref is None:
                ref = {'pts': pts, 'des': des, 'R': R0}
                df.loc[i, 'is_ref'] = True
            n, az, tilt, roll = pointing(ref, pts, des, io)
            df.loc[i, ['pt_ref', 'pt_inliers']] = [Path(df.path[ri]).name, n]
            if n >= th['pt_min_inliers']:
                df.loc[i, ['d_az', 'd_tilt', 'd_roll']] = [(az - eo['azimuth'] + 180) % 360 - 180,
                                                          tilt - eo['tilt'], roll - eo['roll']]
        print(f'{cam} {eo_file}: reference {df.time[ri]}, horizon fit at reference dtilt {df.hz_dtilt[ri]:+.2f}, '
              f'droll {df.hz_droll[ri]:+.2f} (deg); {len(keep)} images checked')
    return df


COLS = ['time', 'quality', 'wh_4061', 'wp_peak', 'R2', 'path', 'station', 'camera', 'wvdir', 'water_level',
        'water_depth', 'H0', 'setup', 'brightness', 'sharpness', 'glint_frac', 'glint_view', 'sun_el', 'sun_az',
        'hz_offset', 'hz_strength', 'hz_dtilt', 'hz_droll', 'hz_fit_px', 'is_ref', 'pt_ref', 'pt_inliers',
        'd_az', 'd_tilt', 'd_roll', 'eo_file', 'io_file']


def write_rows(rows, out_csv):
    df = pd.DataFrame(rows).reindex(columns=COLS)
    df.to_csv(out_csv, mode='a', header=not Path(out_csv).exists(), index=False, float_format='%.4g')


# ------------------------------------------------------------------ running list of EO changes
def eo_changes(df, calib_dir, tol=0.15, min_inliers=40):
    """Group daily median pointing changes into runs of constant pointing per camera and EO file.
    A new run starts when a day's median differs from the running run median by more than tol (deg)
    in any angle. d_az, d_tilt, d_roll are changes since the reference image (first run = 0), and moved is
    True when any exceeds tol. azimuth, tilt, roll = calibration + change, i.e. assume the reference matched
    the calibration; hz_dtilt, hz_droll (horizon fit vs. calibration) test that. Nothing is applied."""
    d = df[df.pt_inliers >= min_inliers].copy()
    d['day'] = d.time.dt.floor('D')
    daily = d.groupby(['camera', 'eo_file', 'day'])[['d_az', 'd_tilt', 'd_roll', 'hz_dtilt', 'hz_droll']].median().reset_index()
    io_file = d.groupby(['camera', 'eo_file']).io_file.first()
    out = []
    for (cam, eo_file), g in daily.groupby(['camera', 'eo_file']):
        eo = read_yaml(Path(calib_dir, eo_file))
        run = [g.iloc[0]]
        for _, r in list(g.iterrows())[1:] + [(None, None)]:
            med = pd.DataFrame(run)[['d_az', 'd_tilt', 'd_roll', 'hz_dtilt', 'hz_droll']].median()
            if r is not None and (np.abs(r[['d_az', 'd_tilt', 'd_roll']] - med[['d_az', 'd_tilt', 'd_roll']]) <= tol).all():
                run.append(r)
                continue
            out.append({'camera': cam, 'eo_file': eo_file, 'io_file': io_file[cam, eo_file], 'first_day': run[0].day.date(), 'last_day': run[-1].day.date(),
                        'n_days': len(run), 'd_az': med.d_az, 'd_tilt': med.d_tilt, 'd_roll': med.d_roll,
                        'hz_dtilt': med.hz_dtilt, 'hz_droll': med.hz_droll,
                        'azimuth': eo['azimuth'] + med.d_az, 'tilt': eo['tilt'] + med.d_tilt, 'roll': eo['roll'] + med.d_roll,
                        'moved': bool((np.abs(med[['d_az', 'd_tilt', 'd_roll']]) > tol).any())})
            run = [r]
    return pd.DataFrame(out)
