# csherwood@usgs.gov, 2026-10-28, generated with Claude Opus 5.5
"""
Hand-picked SAM prompt points for each camera calibration (EO) period.

For each (camera, EO file) period, one good image near local noon and mid-tide is shown with the Otsu
waterlines of images spanning the period's tide range drawn on it. The user clicks land points, then water
points, along the beach. Together they serve two purposes in marconi_sam:
* prompts for SAM (land = 0, background; water = 1, foreground);
* a polygon (land points, then water points in reverse order) that bounds where the waterline may be.

Prompts are saved as rows of a .csv: camera, station, io_file, eo_file, t_first, t_last (period covered by the
images), image (path clicked on), label (land/water), sam_label (0/1), k (click order), u, v (full-res pixels).
Clicking needs an interactive matplotlib backend (%matplotlib qt in Jupyter).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

PROMPT = {'tz': 'America/New_York',   # local time zone
          'hours': (10, 14),          # local hours searched for the image to click on
          'n_overlay': 12,            # Otsu waterlines drawn, spanning the period's tide range
          'figsize': (15, 9)}


def periods(scan):
    """One row per (camera, EO file): station, IO file, first and last image time, number of usable images."""
    w = scan[scan.quality > 1]
    return (w.groupby(['camera', 'eo_file'])
            .agg(station=('station', 'first'), io_file=('io_file', 'first'), t_first=('time', 'min'),
                 t_last=('time', 'max'), n_img=('path', 'size'))
            .reset_index().sort_values(['camera', 't_first']).reset_index(drop=True))


def pick_image(scan, camera, eo_file, wl_px, cfg=PROMPT):
    """Image of a period to click on: highest quality, within the local hours if possible, with a waterline,
    and with the water level closest to the period median (mid-tide)."""
    w = scan[(scan.camera == camera) & (scan.eo_file == eo_file) & (scan.quality > 1) & scan.path.isin(wl_px.groups)]
    w = w[w.quality == w.quality.max()]
    h = w.time.dt.tz_localize('UTC').dt.tz_convert(cfg['tz']).dt.hour
    if ((h >= cfg['hours'][0]) & (h < cfg['hours'][1])).any():
        w = w[(h >= cfg['hours'][0]) & (h < cfg['hours'][1])]
    return w.loc[(w.water_level - w.water_level.median()).abs().idxmin()]


def overlay_paths(scan, camera, eo_file, wl_px, cfg=PROMPT):
    """Images of a period spanning its range of water level (for the waterline overlay)."""
    w = scan[(scan.camera == camera) & (scan.eo_file == eo_file) & (scan.quality > 1)
             & scan.path.isin(wl_px.groups)].sort_values('water_level')
    return w.iloc[np.unique(np.linspace(0, len(w) - 1, cfg['n_overlay']).astype(int))] if len(w) else w


def show_period(ax, img_row, over, wl_px):
    """Image with the Otsu waterlines of the overlay images, colored by water level."""
    ax.imshow(cv2.cvtColor(cv2.imread(img_row.path), cv2.COLOR_BGR2RGB))
    sc = None
    for r in over.itertuples():
        p = wl_px.get_group(r.path)
        sc = ax.scatter(p.u, p.v, c=np.full(len(p), r.water_level), s=0.3, cmap='coolwarm', lw=0,
                        vmin=over.water_level.min(), vmax=over.water_level.max())
    ax.set_axis_off()
    return sc


def polygon(pr):
    """Closed polygon (n, 2) from one period's prompts: land points in click order, then water points in reverse
    (water points are flipped first if they were clicked in the opposite direction along the beach)."""
    land = pr[pr.label == 'land'].sort_values('k')[['u', 'v']].values
    water = pr[pr.label == 'water'].sort_values('k')[['u', 'v']].values
    if np.hypot(*(land[0] - water[0])) > np.hypot(*(land[0] - water[-1])):
        water = water[::-1]
    return np.vstack([land, water[::-1]])


def click_period(per, img_row, over, wl_px, cfg=PROMPT):
    """Collect land then water clicks on one period's image (left click adds, right click or Backspace removes
    the last, Enter finishes). Returns the prompt rows, after drawing the polygon for a look."""
    fig, ax = plt.subplots(figsize=cfg['figsize'])
    sc = show_period(ax, img_row, over, wl_px)
    if sc is not None:
        fig.colorbar(sc, ax=ax, shrink=0.6, label='Water level (m)')
    rows = []
    for label, sam_label, c in [('land', 0, 'yellow'), ('water', 1, 'cyan')]:
        ax.set_title(f'{per.camera} {per.eo_file}  ({per.t_first:%Y-%m-%d} to {per.t_last:%Y-%m-%d}): '
                     f'click {label.upper()} points along the beach, then press Enter', fontsize=11)
        fig.canvas.draw()
        uv = np.array(plt.ginput(n=-1, timeout=0, show_clicks=True))
        ax.plot(uv[:, 0], uv[:, 1], 'o', mfc=c, mec='k', ms=7)
        rows += [(label, sam_label, k, u, v) for k, (u, v) in enumerate(uv)]
    pr = pd.DataFrame(rows, columns=['label', 'sam_label', 'k', 'u', 'v'])
    pg = polygon(pr)
    ax.fill(pg[:, 0], pg[:, 1], fc='none', ec='w', lw=1.5, ls='--')
    ax.set_title(f'{per.camera} {per.eo_file}: {len(pr)} points (close the window to continue)', fontsize=11)
    fig.canvas.draw()
    plt.show(block=True)
    return pr.assign(camera=per.camera, station=per.station, io_file=per.io_file, eo_file=per.eo_file,
                     t_first=per.t_first, t_last=per.t_last, image=img_row.path)


def collect(scan, wl_px, out_csv, redo=(), cfg=PROMPT):
    """Click prompts for every period not yet in out_csv (and those whose EO file is in redo); each period
    is written as soon as it is done, so the loop can be stopped and restarted."""
    pers = periods(scan)
    done = pd.read_csv(out_csv) if Path(out_csv).exists() else pd.DataFrame(columns=['camera', 'eo_file'])
    done = done[~done.eo_file.isin(redo)]
    todo = pers[~(pers.camera + pers.eo_file).isin(done.camera + done.eo_file)]
    print(f'{len(todo)} of {len(pers)} periods to do')
    for per in todo.itertuples():
        pr = click_period(per, pick_image(scan, per.camera, per.eo_file, wl_px, cfg),
                          overlay_paths(scan, per.camera, per.eo_file, wl_px, cfg), wl_px, cfg)
        done = pd.concat([done, pr], ignore_index=True)
        done.to_csv(out_csv, index=False, float_format='%.1f')
        print(per.camera, per.eo_file, (pr.label == 'land').sum(), 'land,', (pr.label == 'water').sum(), 'water')
    return done


def review(prompts, scan, wl_px, path, cfg=PROMPT):
    """Grid of all periods: clicked image, overlay waterlines, prompts and polygon."""
    keys = prompts[['camera', 'eo_file', 'image']].drop_duplicates().reset_index(drop=True)
    nc = min(3, len(keys))
    nr = int(np.ceil(len(keys) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(6 * nc, 4 * nr), squeeze=False)
    for a, k, l in zip(ax.flat, keys.itertuples(), 'abcdefghijklmnopqrstuvwxyz' * 3):
        pr = prompts[(prompts.camera == k.camera) & (prompts.eo_file == k.eo_file)]
        show_period(a, scan[scan.path == k.image].iloc[0], overlay_paths(scan, k.camera, k.eo_file, wl_px, cfg), wl_px)
        for lab, c in [('land', 'yellow'), ('water', 'cyan')]:
            p = pr[pr.label == lab]
            a.plot(p.u, p.v, 'o', mfc=c, mec='k', ms=4)
        pg = polygon(pr)
        a.fill(pg[:, 0], pg[:, 1], fc='none', ec='w', lw=1, ls='--')
        a.set_title(f'{k.camera} {k.eo_file}', fontsize=8)
        a.text(0.01, 0.98, l, transform=a.transAxes, va='top', color='w')
    for a in ax.flat[len(keys):]:
        a.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig
