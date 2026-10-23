# csherwood@usgs.gov, 2026-10-23, generated with Claude Opus 5.5
"""
Daily intertidal DEMs of Marconi Beach from timex waterlines (step 2 of the workflow).

1. Waterline detection in the oblique image: Otsu threshold of the normalized red-minus-blue index
   (R - B)/(R + G + B), which is strongly negative for water and near zero for sand; the sea is the
   connected water region just below the predicted horizon, and the waterline is its edge (excluding the
   horizon cut and image edges). Pixel coordinates are stored, so elevations can be reassigned later.
2. Elevation of each waterline: z = water level + Stockdon et al. (2006) setup, with deep-water height
   from reverse shoaling (marconi_scan.wave_forcing).
3. Rectification: pixels -> UTM at that z with CIRN dist_uv_to_xyz (calibration EO; pointing changes
   are not applied).
4. Gridding (three-day window): weighted local plane fit at each 2 m cell (tricube weights, radius R),
   in the spirit of the scale-controlled interpolation of Plant et al. (2002). Images whose median
   residual to the first DEM is an outlier are dropped and the DEM is refit.
5. Shore-normal transects (fixed for all days, from the principal axis of all waterline points); the
   median transect slope replaces the initial slope for one iteration of the setup calculation.

Uncertainty per cell: sigma^2 = se_fit^2 + sigma_wl^2 + (f_setup * setup)^2, where se_fit is the sandwich
standard error of the plane intercept inflated by n_points/n_images (points along one waterline share
the same elevation error), sigma_wl is the water-level uncertainty, and f_setup the fractional setup error.

Elevations are in the water-level datum (assumed NAVD88); coordinates are those of the EO files (UTM 19N).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from cirn_utils import dist_uv_to_xyz
import marconi_scan as ms

CFG = {'scale': 0.5,        # image scale for waterline detection
       'hz_margin': 40,     # px (full res) below the predicted horizon where the region of interest starts
       'seed_band': 120,    # px (full res) below that, used to pick the sea region
       'edge': 6,           # px (scaled) margin from the region-of-interest edge excluded from the waterline
       'rmin': 30.,         # min and max horizontal range (m) of waterline points from the camera
       'rmax': 450.,
       'thin': 0.5,         # rectified waterline points thinned to one per thin x thin m
       'min_wl_pts': 50,    # min waterline points for an image to be used
       'dx': 2.,            # DEM grid spacing (m)
       'radius': 5.,        # local-fit radius (m)
       'min_img': 3,        # min distinct images in a cell's neighborhood
       'min_pts': 10,       # min points in a cell's neighborhood
       'sig_max': 0.3,      # cells with uncertainty above this (m) are blanked
       'resid_max': 0.25,   # images with |median residual| above this (m) are dropped ...
       'resid_z': 3.5,      # ... or more than this many robust standard deviations from the others
       'min_day_img': 6,    # min usable images in the three-day window
       'sigma_wl': 0.05,    # water-level uncertainty (m)
       'f_setup': 0.3,      # fractional uncertainty of the Stockdon setup
       'beta0': 0.12,       # initial foreshore slope
       'beta_lim': (0.02, 0.3),
       'tr_spacing': 10.,   # alongshore spacing of transects (m)
       'tr_ds': 1.,         # sample spacing along transects (m)
       'min_span': 10.,     # min cross-shore length (m) of DEM along a transect for a slope
       'crs': 'EPSG:6348',  # NAD83(2011) / UTM zone 19N
       'zlim': (-1.5, 2.5), # color limits for PNGs (m)
       'siglim': (0., 0.3)}


# ------------------------------------------------------------------ waterline detection
def water_index(img):
    """(R - B)/(R + G + B) from a BGR image: negative for water, ~0 for sand."""
    b, g, r = [img[..., i].astype(np.float32) for i in range(3)]
    return (r - b) / (r + g + b + 1.)


def otsu(vals, nbins=256):
    """Otsu (1979) threshold of a 1-D array."""
    h, e = np.histogram(vals, nbins)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1)**2)]


def detect_waterline(path, io, eo, cfg=CFG):
    """Waterline pixels (full-res distorted u, v) of one timex image, and the index threshold."""
    s = cfg['scale']
    img = cv2.resize(cv2.imread(path), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    idx = cv2.GaussianBlur(water_index(img), (0, 0), 2)
    nv, nu = idx.shape
    hz = ms.horizon_rows(io, eo, np.arange(nu) / s)
    top = np.nan_to_num(hz, nan=-cfg['hz_margin']) * s + cfg['hz_margin'] * s
    rows = np.arange(nv)[:, None]
    roi = rows > top[None, :]
    seed = roi & (rows < top[None, :] + cfg['seed_band'] * s)
    thr = otsu(idx[roi])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    water = cv2.morphologyEx(((idx < thr) & roi).astype(np.uint8), cv2.MORPH_CLOSE, k)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, k)
    n, lab = cv2.connectedComponents(water, connectivity=8)
    counts = np.bincount(lab[seed], minlength=n)
    counts[0] = 0
    if counts.max() == 0:
        return np.array([]), np.array([]), thr
    sea = (lab == np.argmax(counts)).astype(np.uint8)
    c = max(cv2.findContours(sea, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)[:, 0, :]
    inner = cv2.erode(roi.astype(np.uint8), np.ones((2 * cfg['edge'] + 1,) * 2, np.uint8))
    inner[:, :cfg['edge']] = inner[:, -cfg['edge']:] = 0
    inner[-cfg['edge']:, :] = 0
    c = c[inner[c[:, 1], c[:, 0]] > 0]
    return (c[:, 0] + 0.5) / s - 0.5, (c[:, 1] + 0.5) / s - 0.5, thr


def detect_all(scan, calib_dir, out_csv, cfg=CFG, checkpoint=100):
    """Detect waterlines in all images with quality > 1; append (path, u, v) to out_csv (resumes).
    Returns the per-image index thresholds and point counts of this run."""
    done = set(pd.read_csv(out_csv, usecols=['path']).path) if Path(out_csv).exists() else set()
    todo = scan[(scan.quality > 1) & ~scan.path.isin(done)].reset_index(drop=True)
    cal, rows, info = {}, [], []
    for i, r in todo.iterrows():
        if r.eo_file not in cal:
            cal[r.eo_file] = (ms.read_yaml(Path(calib_dir, r.io_file)), ms.read_yaml(Path(calib_dir, r.eo_file)))
        u, v, thr = detect_waterline(r.path, *cal[r.eo_file], cfg)
        rows.append(pd.DataFrame({'path': r.path, 'u': u if len(u) else [np.nan], 'v': v if len(v) else [np.nan]}))
        info.append((r.path, thr, len(u)))
        if len(rows) == checkpoint or i == len(todo) - 1:
            pd.concat(rows).to_csv(out_csv, mode='a', header=not Path(out_csv).exists(), index=False, float_format='%.1f')
            print(f'{i + 1}/{len(todo)}  {r.time}')
            rows = []
    return pd.DataFrame(info, columns=['path', 'wl_thresh', 'wl_npx'])


# ------------------------------------------------------------------ elevations and rectification
def waterline_points(imgs, wl_px, calib_dir, beta, cfg=CFG, cal=None):
    """Rectified waterline points of images (rows of the scan table) with z = water level + setup(beta).
    Returns a DataFrame x, y, z, setup, img (index into imgs)."""
    cal = {} if cal is None else cal
    out = []
    for i, r in imgs.iterrows():
        if r.path not in wl_px.groups:
            continue
        p = wl_px.get_group(r.path).dropna()
        if r.eo_file not in cal:
            cal[r.eo_file] = (ms.read_yaml(Path(calib_dir, r.io_file)), ms.read_yaml(Path(calib_dir, r.eo_file)))
        io, eo = cal[r.eo_file]
        setup = float(ms.stockdon(r.H0, r.wp_peak, beta)[0])
        z = r.water_level + setup
        xyz = dist_uv_to_xyz(io, eo, p.u.values, p.v.values, 'z', z)
        rng = np.hypot(xyz[:, 0] - eo['x'], xyz[:, 1] - eo['y'])
        xyz = xyz[(rng > cfg['rmin']) & (rng < cfg['rmax'])]
        xy = np.unique(np.round(xyz[:, :2] / cfg['thin']) * cfg['thin'], axis=0)
        if len(xy) >= cfg['min_wl_pts']:
            out.append(pd.DataFrame({'x': xy[:, 0], 'y': xy[:, 1], 'z': z, 'setup': setup, 'img': i}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['x', 'y', 'z', 'setup', 'img'])


# ------------------------------------------------------------------ grid and gridding
def make_grid(x, y, dx, pad=10.):
    """Cell-center coordinates (1-D, ascending) covering the 0.5-99.5 percentile range of points."""
    xg = np.arange(np.floor((np.percentile(x, 0.5) - pad) / dx) * dx, np.percentile(x, 99.5) + pad, dx) + dx / 2
    yg = np.arange(np.floor((np.percentile(y, 0.5) - pad) / dx) * dx, np.percentile(y, 99.5) + pad, dx) + dx / 2
    return xg, yg


def local_fit(pts, xg, yg, cfg=CFG):
    """Weighted local plane fit z = a + b dx + c dy at each cell center (tricube weights within radius).
    Returns dict of 2-D grids (ny, nx): z, sigma, se, rms, n_img, n_pts, setup."""
    X, Y = np.meshgrid(xg, yg)
    cells = np.column_stack([X.ravel(), Y.ravel()])
    R = cfg['radius']
    sd = cKDTree(cells).sparse_distance_matrix(cKDTree(pts[['x', 'y']].values), R, output_type='ndarray')
    ic, ip, d = sd['i'], sd['j'], sd['v']
    w = (1 - (d / R)**3)**3
    dx = pts.x.values[ip] - cells[ic, 0]
    dy = pts.y.values[ip] - cells[ic, 1]
    z = pts.z.values[ip]
    nc = len(cells)
    A = np.stack([np.ones_like(dx), dx, dy], 1)
    M = np.zeros((nc, 3, 3))
    Mw2 = np.zeros((nc, 3, 3))
    bvec = np.zeros((nc, 3))
    for a in range(3):
        bvec[:, a] = np.bincount(ic, w * A[:, a] * z, nc)
        for b in range(3):
            M[:, a, b] = np.bincount(ic, w * A[:, a] * A[:, b], nc)
            Mw2[:, a, b] = np.bincount(ic, w**2 * A[:, a] * A[:, b], nc)
    n_pts = np.bincount(ic, minlength=nc)
    n_img = np.bincount(np.unique(np.column_stack([ic, pts.img.values[ip]]), axis=0)[:, 0], minlength=nc)
    sw = np.bincount(ic, w, nc)
    cx, cy = np.bincount(ic, w * dx, nc) / np.maximum(sw, 1e-12), np.bincount(ic, w * dy, nc) / np.maximum(sw, 1e-12)
    ok = (n_pts >= cfg['min_pts']) & (n_img >= cfg['min_img']) & (np.hypot(cx, cy) < R / 2)
    ok[ok] &= np.linalg.cond(M[ok]) < 1e8
    coef = np.full((nc, 3), np.nan)
    coef[ok] = np.linalg.solve(M[ok], bvec[ok][..., None])[..., 0]
    r = z - np.sum(A * coef[ic], 1)
    ssr = np.bincount(ic, w * np.nan_to_num(r)**2, nc)
    s2 = ssr / np.maximum(sw, 1e-12) * n_pts / np.maximum(n_pts - 3, 1)
    Minv = np.full((nc, 3, 3), np.nan)
    Minv[ok] = np.linalg.inv(M[ok])
    var_a = (Minv @ Mw2 @ Minv)[:, 0, 0] * s2 * n_pts / np.maximum(n_img, 1)
    setup = np.bincount(ic, w * pts.setup.values[ip], nc) / np.maximum(sw, 1e-12)
    se = np.sqrt(var_a)
    sig = np.sqrt(se**2 + cfg['sigma_wl']**2 + (cfg['f_setup'] * setup)**2)
    zc = coef[:, 0]
    bad = ~ok | (sig > cfg['sig_max'])
    zc[bad] = sig[bad] = se[bad] = np.nan
    shp = X.shape
    return {'z': zc.reshape(shp), 'sigma': sig.reshape(shp), 'se': se.reshape(shp),
            'rms': np.sqrt(ssr / np.maximum(sw, 1e-12)).reshape(shp), 'n_img': n_img.reshape(shp),
            'n_pts': n_pts.reshape(shp), 'setup': setup.reshape(shp)}


def sample(grid, xg, yg, x, y):
    """Bilinear sample of a grid (ny, nx) at points; NaN outside or next to blank cells."""
    return RegularGridInterpolator((yg, xg), grid, bounds_error=False, fill_value=np.nan)(np.column_stack([y, x]))


def image_residuals(pts, dem, xg, yg):
    """Median residual (point z - DEM) and count per image."""
    r = pts.z.values - sample(dem, xg, yg, pts.x.values, pts.y.values)
    return pd.DataFrame({'img': pts.img, 'r': r}).dropna().groupby('img').r.agg(['median', 'count'])


def dem_with_check(pts, xg, yg, cfg=CFG):
    """Local-fit DEM, then drop images whose median residual is an outlier and refit.
    Returns (grids, per-image residual table with 'used')."""
    g = local_fit(pts, xg, yg, cfg)
    res = image_residuals(pts, g['z'], xg, yg)
    med = res['median'].median()
    mad = 1.4826 * np.median(np.abs(res['median'] - med))
    res['used'] = (np.abs(res['median']) < cfg['resid_max']) & (np.abs(res['median'] - med) < cfg['resid_z'] * max(mad, 0.02))
    drop = res.index[~res.used]
    if len(drop):
        g = local_fit(pts[~pts.img.isin(drop)], xg, yg, cfg)
    res2 = image_residuals(pts, g['z'], xg, yg)
    res['resid'] = res2['median']
    return g, res


# ------------------------------------------------------------------ transects and slopes
def shore_frame(x, y, z):
    """Origin (centroid), alongshore and offshore unit vectors from the principal axes of waterline points;
    offshore points toward lower elevations. Also returns the offshore azimuth (deg CW from north)."""
    o = np.array([x.mean(), y.mean()])
    evals, evecs = np.linalg.eigh(np.cov(np.vstack([x - o[0], y - o[1]])))
    along, off = evecs[:, 1], evecs[:, 0]
    if np.corrcoef((x - o[0]) * off[0] + (y - o[1]) * off[1], z)[0, 1] > 0:
        off = -off
    return o, along, off, np.degrees(np.arctan2(off[0], off[1])) % 360


def make_transects(x, y, o, along, off, cfg=CFG):
    """Shore-normal transects spanning the waterline points. Returns table (id, s_along) and sample
    coordinates: arrays X, Y (n_transects, n_samples) and cross-shore distance s (positive offshore)."""
    a = (x - o[0]) * along[0] + (y - o[1]) * along[1]
    c = (x - o[0]) * off[0] + (y - o[1]) * off[1]
    sa = np.arange(np.ceil(np.percentile(a, 1) / cfg['tr_spacing']), np.percentile(a, 99) / cfg['tr_spacing']) * cfg['tr_spacing']
    s = np.arange(np.percentile(c, 0.5) - 10, np.percentile(c, 99.5) + 10, cfg['tr_ds'])
    X = o[0] + sa[:, None] * along[0] + s[None, :] * off[0]
    Y = o[1] + sa[:, None] * along[1] + s[None, :] * off[1]
    return pd.DataFrame({'id': np.arange(len(sa)), 's_along': sa,
                         'x0': X[:, 0], 'y0': Y[:, 0], 'x1': X[:, -1], 'y1': Y[:, -1]}), X, Y, s


def transect_slopes(dem, xg, yg, X, Y, s, cfg=CFG):
    """Least-squares slope (-dz/ds, positive for a beach dropping seaward) of the DEM along each transect,
    with the cross-shore span and elevation range used; NaN where the valid span < min_span."""
    Z = sample(dem, xg, yg, X.ravel(), Y.ravel()).reshape(X.shape)
    out = []
    for zt in Z:
        ok = np.isfinite(zt)
        if ok.sum() < 2 or np.ptp(s[ok]) < cfg['min_span']:
            out.append((np.nan, 0., np.nan, np.nan))
            continue
        p = np.polyfit(s[ok], zt[ok], 1)
        out.append((-p[0], np.ptp(s[ok]), zt[ok].min(), zt[ok].max()))
    return pd.DataFrame(out, columns=['slope', 'span', 'zmin', 'zmax']), Z


# ------------------------------------------------------------------ output
def write_tif(path, grid, xg, yg, crs, desc):
    """Single-band float32 GeoTIFF (north up) from a grid with ascending yg and cell-center coordinates."""
    dx = xg[1] - xg[0]
    with rasterio.open(path, 'w', driver='GTiff', height=len(yg), width=len(xg), count=1, dtype='float32',
                       crs=crs, transform=from_origin(xg[0] - dx / 2, yg[-1] + dx / 2, dx, dx),
                       nodata=np.nan, compress='deflate') as f:
        f.write(np.flipud(grid).astype(np.float32), 1)
        f.set_band_description(1, desc)


def dem_png(path, g, xg, yg, title, tr=None, cfg=CFG):
    """Two-panel map (a elevation, b uncertainty) with fixed extent and color limits, for animation."""
    dx = xg[1] - xg[0]
    ext = [xg[0] - dx / 2, xg[-1] + dx / 2, yg[0] - dx / 2, yg[-1] + dx / 2]
    pw = np.clip(7 * (ext[1] - ext[0]) / (ext[3] - ext[2]), 1.2, 7)    # panel width for a 7 in tall map
    fig, ax = plt.subplots(1, 2, figsize=(2 * pw + 3.5, 8), sharey=True)
    im0 = ax[0].imshow(g['z'], origin='lower', extent=ext, cmap='viridis', vmin=cfg['zlim'][0], vmax=cfg['zlim'][1])
    im1 = ax[1].imshow(g['sigma'], origin='lower', extent=ext, cmap='magma_r', vmin=cfg['siglim'][0], vmax=cfg['siglim'][1])
    if tr is not None:
        for _, t in tr.iterrows():
            ax[0].plot([t.x0, t.x1], [t.y0, t.y1], c='0.6', lw=0.3)
    plt.colorbar(im0, ax=ax[0], shrink=0.8, label='Elevation (m)')
    plt.colorbar(im1, ax=ax[1], shrink=0.8, label='Uncertainty (m)')
    for a, l in zip(ax, 'ab'):
        a.set_aspect('equal')
        a.set_xlabel('Easting (m)')
        a.ticklabel_format(useOffset=False, style='plain')
        a.tick_params(labelsize=7)
        a.xaxis.set_major_locator(plt.MaxNLocator(3))
        a.text(0.02, 0.98, l, transform=a.transAxes, va='top')
    ax[0].set_ylabel('Northing (m)')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ one day
def slope_stats(sl):
    v = sl.slope.dropna()
    return {'slope_n': len(v), 'slope_median': v.median(), 'slope_mean': v.mean(), 'slope_std': v.std(),
            'slope_p10': v.quantile(0.1), 'slope_p90': v.quantile(0.9)}


def grid_stats(g, dx):
    ok = np.isfinite(g['z'])
    s = g['sigma'][ok]
    return {'dem_n_cells': int(ok.sum()), 'dem_area_m2': ok.sum() * dx**2,
            'dem_zmin': np.nanmin(g['z']) if ok.any() else np.nan, 'dem_zmax': np.nanmax(g['z']) if ok.any() else np.nan,
            'unc_median': np.median(s) if ok.any() else np.nan, 'unc_mean': s.mean() if ok.any() else np.nan,
            'unc_p90': np.percentile(s, 90) if ok.any() else np.nan}


def process_day(day, scan, wl_px, calib_dir, xg, yg, trX, trY, trs, cfg=CFG, cal=None):
    """DEM for one day from images of the day before, of, and after. Two passes: setup with beta0, then with
    the median transect slope of the first DEM. Returns (grids, transect slopes, per-image table, day stats)."""
    w = scan[(scan.quality > 1) & (scan.time >= day - pd.Timedelta('1D')) & (scan.time < day + pd.Timedelta('2D'))
             & scan.H0.notna() & scan.water_level.notna()]
    blank = {k: np.full((len(yg), len(xg)), np.nan) for k in ['z', 'sigma']}
    st = {'day': day.date(), 'n_img_window': len(w)}
    beta, g, res, sl = cfg['beta0'], blank, pd.DataFrame(columns=['median', 'count', 'used', 'resid']), None
    for it in range(2):
        pts = waterline_points(w, wl_px, calib_dir, beta, cfg, cal)
        if pts.img.nunique() < cfg['min_day_img']:
            st['status'] = 'too few images'
            break
        g, res = dem_with_check(pts, xg, yg, cfg)
        sl, _ = transect_slopes(g['z'], xg, yg, trX, trY, trs, cfg)
        st |= {f'slope_median_{it}': sl.slope.median(), f'beta_{it}': beta}
        if it == 0 and np.isfinite(sl.slope.median()):
            beta = float(np.clip(sl.slope.median(), *cfg['beta_lim']))
        st['status'] = 'ok'
    st |= {'beta_used': beta, 'n_img_used': int(res.used.sum()) if len(res) else 0,
           'n_img_dropped': int((~res.used.astype(bool)).sum()) if len(res) else 0}
    st |= grid_stats(g, cfg['dx'])
    if sl is not None:
        st |= slope_stats(sl)
    res['path'] = w.loc[res.index, 'path'].values if len(res) else []
    res['setup_dem'] = [float(ms.stockdon(w.H0[i], w.wp_peak[i], beta)[0]) for i in res.index]
    return g, sl, res, st


def run_days(scan, wl_px, calib_dir, days, xg, yg, tr, trX, trY, trs, out_dir, fig_dir, cfg=CFG):
    """Process each day: GeoTIFFs (elevation, uncertainty) to out_dir, PNG maps to fig_dir.
    Returns (day table, per-image table for each image's own day, transect slopes for all days)."""
    days_out, img_out, sl_out, cal = [], [], [], {}
    for day in days:
        g, sl, res, st = process_day(day, scan, wl_px, calib_dir, xg, yg, trX, trY, trs, cfg, cal)
        tag = f'{day:%Y-%m-%d}'
        write_tif(Path(out_dir, f'dem_{tag}.tif'), g['z'], xg, yg, cfg['crs'], 'elevation (m)')
        write_tif(Path(out_dir, f'unc_{tag}.tif'), g['sigma'], xg, yg, cfg['crs'], 'uncertainty (m)')
        dem_png(Path(fig_dir, f'dem_{tag}.png'), g, xg, yg,
                f"{tag}: {st['status']}, {st.get('n_img_used', 0)} images, slope {st.get('slope_median', np.nan):.3f}", tr, cfg)
        own = res[pd.to_datetime(scan.loc[res.index, 'time']).dt.floor('D').values == day] if len(res) else res
        img_out.append(own.assign(day=day.date()))
        if sl is not None:
            sl_out.append(sl.assign(day=day.date(), id=tr.id.values))
        days_out.append(st)
        print(tag, st['status'], st.get('dem_n_cells', 0), 'cells')
    return pd.DataFrame(days_out), pd.concat(img_out), pd.concat(sl_out) if sl_out else pd.DataFrame()
