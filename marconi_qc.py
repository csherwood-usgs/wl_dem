# csherwood@usgs.gov, 2026-10-27, generated with Claude Opus 5.5
"""
Quality-control products for the Marconi daily DEMs:
* a rectified composite (c1 and c2) of the timex images closest to local noon, projected onto the day's DEM
  surface, with the rectified waterline points of the three-day window and the DEM contours overlaid;
* rectified pixel-instrument (timestack) transects, found by intersecting each pixel's ray with the DEM.

Uses the CIRN projection functions in cirn_utils and the date-specific calibration files of each image.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from cirn_utils import dist_uv_to_xyz, xyz_to_dist_uv
import marconi_scan as ms
import marconi_dem as md
import marconi_movie as mm

QC = {'tz': 'America/New_York',   # local time zone for choosing the noon images
      'noon': 12.,                # target local hour
      'dx_img': 0.5,              # composite grid spacing (m)
      'z_fallback': 0.,           # surface elevation (m) where there is no DEM at all
      'n_iter': 10,               # iterations for intersecting pixel rays with the DEM
      'pix_colors': {'c1': 'w', 'c2': 'tab:orange'},
      'pts_size': 0.4,            # marker size of waterline points
      'map_h': 8.}                # map panel height (in)


def cal_pair(r, calib_dir, cal):
    """(io, eo) for a scan row, cached by EO file name."""
    if r.eo_file not in cal:
        cal[r.eo_file] = (ms.read_yaml(Path(calib_dir, r.io_file)), ms.read_yaml(Path(calib_dir, r.eo_file)))
    return cal[r.eo_file]


def noon_images(scan, day, cfg=QC):
    """For each camera, the quality > 1 image of a (UTC) day closest to local noon."""
    w = scan[(scan.quality > 1) & (scan.time.dt.floor('D') == day)].copy()
    loc = w.time.dt.tz_localize('UTC').dt.tz_convert(cfg['tz'])
    w['dh'] = np.abs(loc.dt.hour + loc.dt.minute / 60 - cfg['noon'])
    return w.sort_values('dh').groupby('camera').head(1).sort_values('camera')


def surface(dem, xd, yd, X, Y, z0):
    """DEM sampled on X, Y (bilinear), with blank cells filled from the nearest DEM cell; z0 if no DEM."""
    if not np.isfinite(dem).any():
        return np.full(X.shape, z0)
    i = distance_transform_edt(~np.isfinite(dem), return_distances=False, return_indices=True)
    f = RegularGridInterpolator((yd, xd), dem[tuple(i)], bounds_error=False, fill_value=None)
    return f(np.column_stack([Y.ravel(), X.ravel()])).reshape(X.shape)


def rectify(img, io, eo, X, Y, Z):
    """Image (BGR) resampled onto world points X, Y, Z; returns RGB float (NaN outside the view) and range (m)."""
    uv, flag = xyz_to_dist_uv(io, eo, np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    u, v = uv[0].reshape(X.shape).astype(np.float32), uv[1].reshape(X.shape).astype(np.float32)
    ok = (flag.reshape(X.shape) > 0) & (u >= 0) & (u < img.shape[1] - 1) & (v >= 0) & (v < img.shape[0] - 1)
    out = cv2.remap(img, u, v, cv2.INTER_LINEAR)[..., ::-1].astype(float) / 255
    out[~ok] = np.nan
    rng = np.where(ok, np.hypot(X - eo['x'], Y - eo['y']), np.inf)
    return out, rng


def composite(rows, calib_dir, X, Y, Z, cal):
    """Composite of the images in rows (one per camera): each cell from the camera closest to it."""
    ims, rngs = zip(*[rectify(cv2.imread(r.path), *cal_pair(r, calib_dir, cal), X, Y, Z) for _, r in rows.iterrows()])
    k = np.argmin(np.stack(rngs), 0)
    return np.take_along_axis(np.stack(ims), k[None, ..., None], 0)[0]


def read_pix(path):
    """Pixel-instrument (u, v) file: two columns, one pixel per line."""
    return np.loadtxt(path, ndmin=2)[:, :2]


def pix_to_xyz(io, eo, uv, surf, z0=0., n_iter=10):
    """Ground positions of pixels: each ray intersected with a surface z = surf(x, y) by fixed-point iteration
    (z0 where surf is NaN). Returns x, y and the surface z (NaN off the surface)."""
    z = np.full(len(uv), z0)
    for _ in range(n_iter):
        xyz = dist_uv_to_xyz(io, eo, uv[:, 0], uv[:, 1], 'z', z)
        zs = surf(xyz[:, 0], xyz[:, 1])
        z = np.where(np.isfinite(zs), zs, z0)
    return xyz[:, 0], xyz[:, 1], zs


def day_pix(day, rows, pix, calib_dir, dem, xd, yd, cal, cfg=QC):
    """Rectified pixel instruments of each camera in rows, on the day's DEM. pix: dict camera -> (n, 2) uv."""
    f = RegularGridInterpolator((yd, xd), dem, bounds_error=False, fill_value=np.nan)
    surf = lambda x, y: f(np.column_stack([y, x]))
    out = []
    for _, r in rows[rows.camera.isin(pix)].iterrows():
        uv = pix[r.camera]
        x, y, z = pix_to_xyz(*cal_pair(r, calib_dir, cal), uv, surf, cfg['z_fallback'], cfg['n_iter'])
        out.append(pd.DataFrame({'day': day.date(), 'camera': r.camera, 'eo_file': r.eo_file, 'i': np.arange(len(uv)),
                                 'u': uv[:, 0], 'v': uv[:, 1], 'x': x, 'y': y, 'z': z}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def plot_pix(ax, p, cfg=QC):
    """Pixel-instrument transects (one line per camera), labeled at the end farthest from the camera
    that is within the axis limits (set the limits first)."""
    for cam, g in p.groupby('camera'):
        ax.plot(g.x, g.y, c=cfg['pix_colors'].get(cam, 'k'), lw=1.2)
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        g = g[(g.x > x0) & (g.x < x1) & (g.y > y0) & (g.y < y1)]
        if len(g) == 0:
            continue
        e = g.loc[g.v.idxmin()]
        ax.text(e.x, e.y, f' {cam}', c=cfg['pix_colors'].get(cam, 'k'), fontsize=8, va='center')


def window_points(day, scan, wl_px, calib_dir, beta, dem, xd, yd, cfg=md.CFG, cal=None):
    """Rectified waterline points of the three-day window (setup with beta), each flagged 'used' unless its
    image's median residual to the day's DEM exceeds resid_max or the image has no overlap with the DEM."""
    w = scan[(scan.quality > 1) & (scan.time >= day - pd.Timedelta('1D')) & (scan.time < day + pd.Timedelta('2D'))
             & scan.H0.notna() & scan.water_level.notna()]
    pts = md.waterline_points(w, wl_px, calib_dir, beta, cfg, cal)
    if len(pts) == 0 or not np.isfinite(dem).any():
        return pts.assign(used=False)
    res = md.image_residuals(pts, dem, xd, yd)
    good = res.index[np.abs(res['median']) < cfg['resid_max']]
    return pts.assign(used=pts.img.isin(good))


def qc_figure(path, day, comp, xc, yc, dem, xd, yd, pts, pix, title, xl, yl, cfg=QC, mcfg=mm.MOVIE):
    """Two panels on the noon composite: a waterline points (colored by elevation; images rejected
    against the DEM in red), b DEM with contours; pixel-instrument transects on both."""
    d = xc[1] - xc[0]
    ext = [xc[0] - d / 2, xc[-1] + d / 2, yc[0] - d / 2, yc[-1] + d / 2]
    pw = max(cfg['map_h'] * (xl[1] - xl[0]) / (yl[1] - yl[0]), 2.5)    # panel width (in)
    fig, ax = plt.subplots(1, 2, figsize=(2 * pw + 1.6, cfg['map_h'] + 2.2), sharex=True, sharey=True)
    for a in ax:
        a.imshow(np.nan_to_num(comp, nan=1.), origin='lower', extent=ext)
    if len(pts):
        u = pts[pts.used]
        sc = ax[0].scatter(u.x, u.y, c=u.z, s=cfg['pts_size'], cmap='viridis', vmin=mcfg['zlim'][0],
                           vmax=mcfg['zlim'][1], lw=0)
        ax[0].scatter(pts.x[~pts.used], pts.y[~pts.used], c='r', s=cfg['pts_size'], lw=0)
    else:
        sc = plt.cm.ScalarMappable(plt.Normalize(*mcfg['zlim']), 'viridis')
    dd = xd[1] - xd[0]
    im = ax[1].imshow(dem, origin='lower', extent=[xd[0] - dd / 2, xd[-1] + dd / 2, yd[0] - dd / 2, yd[-1] + dd / 2],
                      cmap='viridis', vmin=mcfg['zlim'][0], vmax=mcfg['zlim'][1], alpha=0.6)
    if np.isfinite(dem).any():
        lev = np.arange(np.floor(np.nanmin(dem)), np.nanmax(dem) + 0.01, mcfg['contours'])
        cs = ax[1].contour(xd, yd, dem, levels=lev, colors='k', linewidths=0.7)
        ax[1].clabel(cs, fmt='%g', fontsize=7)
    for a, h, lab, l in zip(ax, [sc, im], ['Waterline elevation (m)', 'Elevation (m)'], 'ab'):
        a.set_xlim(xl)
        a.set_ylim(yl)
        if len(pix):
            plot_pix(a, pix, cfg)
        a.set_aspect('equal')
        a.ticklabel_format(useOffset=False, style='plain')
        a.tick_params(labelsize=7)
        a.xaxis.set_major_locator(plt.MaxNLocator(3))
        a.set_xlabel('Easting (m)')
        cb = plt.colorbar(h, ax=a, orientation='horizontal', shrink=0.9, pad=0.07, aspect=15, label=lab)
        cb.locator = plt.MultipleLocator(1.)
        cb.update_ticks()
        a.text(0.03, 0.985, l, transform=a.transAxes, va='top', fontsize=11,
               bbox={'fc': 'w', 'ec': 'none', 'alpha': 0.7, 'pad': 1})
    ax[0].set_ylabel('Northing (m)')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_qc_days(days, scan, wl_px, calib_dir, dem_dir, fig_dir, pix, beta, status, cfg=QC, dcfg=md.CFG,
                mcfg=mm.MOVIE):
    """QC figure (fig_dir/qc_YYYY-MM-DD.png) for each day and the rectified pixel instruments of all days.
    beta, status: dicts day -> slope used for setup and title text."""
    xl, yl = mm.data_bounds(dem_dir, days, mcfg)
    xc = np.arange(xl[0], xl[1], cfg['dx_img']) + cfg['dx_img'] / 2
    yc = np.arange(yl[0], yl[1], cfg['dx_img']) + cfg['dx_img'] / 2
    X, Y = np.meshgrid(xc, yc)
    cal, allpix = {}, []
    for day in days:
        dem, xd, yd = mm.read_tif(Path(dem_dir, f'dem_{day:%Y-%m-%d}.tif'))
        dem, yd = dem[::-1], yd[::-1]
        rows = noon_images(scan, day, cfg)
        if len(rows) == 0:
            print(f'{day:%Y-%m-%d} no daytime images')
            continue
        z0 = rows.water_level.mean() if rows.water_level.notna().any() else cfg['z_fallback']
        comp = composite(rows, calib_dir, X, Y, surface(dem, xd, yd, X, Y, z0), cal)
        pts = window_points(day, scan, wl_px, calib_dir, beta.get(day, dcfg['beta0']), dem, xd, yd, dcfg, cal)
        p = day_pix(day, rows, pix, calib_dir, dem, xd, yd, cal, cfg)
        allpix.append(p)
        t = ', '.join(f'{r.camera} {r.time:%H:%M}' for r in rows.itertuples())
        qc_figure(Path(fig_dir, f'qc_{day:%Y-%m-%d}.png'), day, comp, xc, yc, dem, xd, yd, pts, p,
                  f"{day:%Y-%m-%d}  {status.get(day, '')}\ncomposite {t} UTC; "
                  f"{int(pts.used.sum()) if len(pts) else 0} points used, {int((~pts.used).sum()) if len(pts) else 0} rejected (red)",
                  xl, yl, cfg, mcfg)
        print(f'{day:%Y-%m-%d}  {len(pts)} points')
    return pd.concat(allpix, ignore_index=True) if allpix else pd.DataFrame()
