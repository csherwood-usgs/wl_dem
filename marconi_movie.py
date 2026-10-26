# csherwood@usgs.gov, 2026-10-26, generated with Claude Opus 5.5
"""
Movie frames of the daily Marconi DEMs (from the GeoTIFFs written by marconi_dem.run_days) and .avi assembly.
Each frame: a Hs and b water level (static, with a bar at the DEM day), then c elevation with 1 m contours,
d change from the previous non-empty DEM (blue accretion, red erosion), and e uncertainty, all at the same
scale and bounds.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import rasterio

MOVIE = {'xlim': (None, 420200.),   # map bounds (m); None = from the extent of data in all DEMs
         'ylim': (None, None),
         'pad': 4.,                 # m added around the data extent
         'zlim': (-1.5, 2.5),       # elevation color limits (m)
         'dzlim': (-0.5, 0.5),      # change color limits (m)
         'siglim': (0., 0.3),       # uncertainty color limits (m)
         'contours': 1.,            # contour interval (m)
         'window': True,            # shade the three-day window in a and b
         'figsize': (10, 13),       # constant frame size (in)
         'dpi': 100}


def read_tif(path):
    """Grid (north up) and cell-center coordinates x (ascending) and y (descending)."""
    with rasterio.open(path) as f:
        g = f.read(1)
        t = f.transform
    return g, t.c + t.a * (np.arange(g.shape[1]) + 0.5), t.f + t.e * (np.arange(g.shape[0]) + 0.5)


def data_bounds(dem_dir, days, cfg=MOVIE):
    """Map bounds: the extent of valid cells over all DEMs plus pad, overridden by cfg xlim/ylim where set."""
    lo, hi = np.array([np.inf, np.inf]), -np.array([np.inf, np.inf])
    for d in days:
        g, x, y = read_tif(Path(dem_dir, f'dem_{d:%Y-%m-%d}.tif'))
        r, c = np.nonzero(np.isfinite(g))
        if len(r):
            lo = np.minimum(lo, [x[c].min(), y[r].min()])
            hi = np.maximum(hi, [x[c].max(), y[r].max()])
    xl = [cfg['xlim'][0] or lo[0] - cfg['pad'], cfg['xlim'][1] or hi[0] + cfg['pad']]
    yl = [cfg['ylim'][0] or lo[1] - cfg['pad'], cfg['ylim'][1] or hi[1] + cfg['pad']]
    return xl, yl


def frame(path, day, dem_dir, waves, prev_day, xl, yl, title='', cfg=MOVIE):
    """One five-panel movie frame saved as PNG. prev_day: date of the DEM to difference against, or None."""
    g, x, y = read_tif(Path(dem_dir, f'dem_{day:%Y-%m-%d}.tif'))
    s = read_tif(Path(dem_dir, f'unc_{day:%Y-%m-%d}.tif'))[0]
    dz = g - read_tif(Path(dem_dir, f'dem_{prev_day:%Y-%m-%d}.tif'))[0] if prev_day is not None else np.full_like(g, np.nan)
    dx = x[1] - x[0]
    ext = [x[0] - dx / 2, x[-1] + dx / 2, y[-1] - dx / 2, y[0] + dx / 2]

    fig = plt.figure(figsize=cfg['figsize'], dpi=cfg['dpi'])
    top = fig.add_gridspec(2, 1, hspace=0.08, left=0.09, right=0.97, top=0.955, bottom=0.79)
    bot = fig.add_gridspec(1, 3, wspace=0.08, left=0.09, right=0.97, top=0.735, bottom=0.03)
    a = fig.add_subplot(top[0])
    b = fig.add_subplot(top[1], sharex=a)
    a.plot(waves.time, waves.wh_4061, c='tab:blue', lw=0.8)
    b.plot(waves.time, waves.water_level, c='k', lw=0.6)
    a.set_ylabel(r'$H_s$ (m)')
    b.set_ylabel('Water level (m)')
    for ax in (a, b):
        ax.axvline(day + pd.Timedelta('12h'), c='tab:red', lw=2)
        if cfg['window']:
            ax.axvspan(day - pd.Timedelta('1D'), day + pd.Timedelta('2D'), color='tab:red', alpha=0.12, lw=0)
        ax.set_xlim(waves.time.iloc[0], waves.time.iloc[-1])
    plt.setp(a.get_xticklabels(), visible=False)

    m = [fig.add_subplot(bot[i]) for i in range(3)]
    im = [m[0].imshow(g, extent=ext, cmap='viridis', vmin=cfg['zlim'][0], vmax=cfg['zlim'][1]),
          m[1].imshow(dz, extent=ext, cmap='RdBu', vmin=cfg['dzlim'][0], vmax=cfg['dzlim'][1]),
          m[2].imshow(s, extent=ext, cmap='magma_r', vmin=cfg['siglim'][0], vmax=cfg['siglim'][1])]
    if np.isfinite(g).any():
        lev = np.arange(np.floor(np.nanmin(g)), np.nanmax(g) + 0.01, cfg['contours'])
        cs = m[0].contour(x, y, g, levels=lev, colors='k', linewidths=0.6)
        m[0].clabel(cs, fmt='%g', fontsize=7)
    labels = ['Elevation (m)', 'Change (m)', 'Uncertainty (m)']
    m[0].set_title(f'{day:%Y-%m-%d}', fontsize=9)
    m[1].set_title(f'minus {prev_day:%Y-%m-%d}' if prev_day is not None else 'no earlier DEM', fontsize=9)
    m[2].set_title(f'{day:%Y-%m-%d}', fontsize=9)
    for ax, i, lab in zip(m, im, labels):
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_aspect('equal')
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_xlabel('Easting (m)', fontsize=8, labelpad=1)
        plt.colorbar(i, ax=ax, orientation='horizontal', shrink=0.9, pad=0.05, aspect=15, label=lab)
    for ax in m[1:]:
        ax.set_yticklabels([])
    m[0].set_ylabel('Northing (m)')
    for ax, l, xy in zip([a, b] + m, 'abcde', [(0.01, 0.96)] * 2 + [(0.05, 0.99)] * 3):
        ax.text(*xy, l, transform=ax.transAxes, va='top', fontsize=11,
                bbox={'fc': 'w', 'ec': 'none', 'alpha': 0.7, 'pad': 1})
    fig.suptitle(f'{day:%Y-%m-%d}  {title}', fontsize=12)
    fig.savefig(path, dpi=cfg['dpi'])
    plt.close(fig)


def make_frames(dem_dir, frame_dir, days, waves, status=None, cfg=MOVIE):
    """Frames for all days (frame_####.png). The change panel uses the previous day with a non-empty DEM.
    status: optional dict day -> title text. Returns list of frame paths."""
    xl, yl = data_bounds(dem_dir, days, cfg)
    prev, paths = None, []
    for i, d in enumerate(days):
        p = Path(frame_dir, f'frame_{i:04d}.png')
        frame(p, d, dem_dir, waves, prev, xl, yl, (status or {}).get(d, ''), cfg)
        if np.isfinite(read_tif(Path(dem_dir, f'dem_{d:%Y-%m-%d}.tif'))[0]).any():
            prev = d
        paths.append(p)
    return paths


def write_avi(paths, out, fps=4):
    """Motion-JPEG .avi from PNG frames, with OpenCV's built-in MJPEG encoder (needs no FFmpeg backend).
    Frames are resized to the size of the first one."""
    h, w = cv2.imread(str(paths[0])).shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f'cannot open {out} for writing')
    for p in paths:
        vw.write(cv2.resize(cv2.imread(str(p)), (w, h)))
    vw.release()
