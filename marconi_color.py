# csherwood@usgs.gov, 2026-10-28, generated with Claude Opus 5.5 (var and bright features, snow class 2026-10-29)
"""
Pixel classes (sand, whitewater, water, snow) for the Marconi images, used to put the waterline at the
dry / wet transition and to steer SAM there.

Following the idea of CoastSat (Vos et al., 2019a, b), pixels are classified and the shoreline is the boundary
between the dry classes (sand, snow) and the wet classes (whitewater, water), so whitewater in the surf zone
is not taken as the beach or as the edge of the sea. The classifier is a Gaussian (quadratic discriminant)
model fit to hand-labeled pixels. Features come from the timex image and, when available, the var and bright
images of the same collection (same file name with .var. or .bright. in place of .timex.):
    r = R / (R + G + B), g = G / (R + G + B)   timex chromaticity (sand warm, water blue, whitewater and snow
                                               neutral)
    b = ln(V / V_ref)                          timex brightness relative to the median around the clicked
                                               land points
    q = ln(sigma / sigma_ref)                  var image (gray level) relative to its median around the clicked
                                               land points: whitewater comes and goes during the collection,
                                               so it is far more variable than static sand or snow
    e = ln(V_bright / V)                       bright excursion: how much brighter the brightest moment was
                                               than the mean; large for intermittent foam, near zero for
                                               anything static (sand, snow)
with V = (R + G + B) / 3 and sigma the var-image gray level. The var and bright features separate whitewater
from snow, which have the same color in a timex, and help with glint and dark wet sand. Which products are used
is set by COLOR['products'] and stored with the model. Posterior class probabilities assume equal priors;
classes without labels are left out of the model.

The model is used three ways (marconi_sam): per-image SAM prompts placed on either side of the dry/wet
boundary, choice among SAM's candidate masks by agreement with the classes, and a stand-alone class waterline.

Labels are saved as rows of a .csv: path, camera, eo_file, cls, u, v (full-res pixels of each click); features
are computed from the images when the model is fit, so the clicks never need redoing when features change.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import marconi_prompts as mp

CLASSES = ['sand', 'whitewater', 'water', 'snow']
DRY = ['sand', 'snow']
COLOR = {'scale': 0.5,          # image scale for classification (as for waterline detection)
         'products': ('timex', 'var', 'bright'),   # images used for features ('timex' must be first)
         'blur': 1.,            # Gaussian blur (px, scaled) before computing features
         'radius': 4,           # px (full res) around each click sampled for training
         'ridge': 1e-6,         # added to covariance diagonals
         'p_dry': 0.5,          # dry (sand + snow) probability below which a pixel is wet
         'band': (6, 30),       # px (scaled) from the dry/wet boundary within which prompt points are placed
         'n_color_prompts': 8,  # foreground and background prompt points per image from the classes
         'seed': 0,
         'cls_colors': {'sand': 'yellow', 'whitewater': 'w', 'water': 'b', 'snow': 'm'},
         'cls_rgb': {'sand': '#e8c766', 'whitewater': 'w', 'water': '#2b5d8a', 'snow': '#d36fd3'},
         'figsize': (15, 9)}
FEATURES = {'timex': ['r', 'g', 'ln(V/Vref)'], 'var': ['ln(sigma/sigma_ref)'], 'bright': ['ln(Vbright/V)']}


# ------------------------------------------------------------------ features and model
def product_path(path, product):
    """Path of another product (var, bright, dark, snap) of the same collection as a timex image."""
    return str(path).replace('.timex.', f'.{product}.')


def load_scaled(path, cfg=COLOR):
    s = cfg['scale']
    return cv2.resize(cv2.imread(str(path)), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def feature_names(products):
    return sum([FEATURES[p] for p in products], [])


def features(path, ref, cfg=COLOR, products=None):
    """Feature image (nv, nu, n) of a timex image and its other products at classification scale.
    ref: boolean mask of the reference region (see land_ref)."""
    products = cfg['products'] if products is None else products
    blur = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), cfg['blur']) + 1.
    f = blur(load_scaled(path, cfg))
    v = f.mean(2)
    F = [f[..., 2] / (3 * v), f[..., 1] / (3 * v), np.log(v / np.median(v[ref]))]
    if 'var' in products:
        q = blur(load_scaled(product_path(path, 'var'), cfg)).mean(2)
        F.append(np.log(q / np.median(q[ref])))
    if 'bright' in products:
        F.append(np.log(blur(load_scaled(product_path(path, 'bright'), cfg)).mean(2) / v))
    return np.dstack(F)


def poly_mask(poly, shape, s):
    """Boolean mask of a polygon (full-res u, v) at scale s."""
    return cv2.fillPoly(np.zeros(shape, np.uint8), [np.round(poly * s).astype(np.int32)], 1) > 0


def land_ref(prompt, shape, s, r=8):
    """Reference mask for brightness and variance: disks of r px (scaled) around the clicked land points."""
    m = np.zeros(shape, np.uint8)
    for u, v in np.round(prompt[0][prompt[1] == 0] * s).astype(int):
        cv2.circle(m, (u, v), r, 1, -1)
    return m > 0


def sample_labels(labels, prompts, cfg=COLOR):
    """Feature samples (n, k), class name, and image path for every pixel within radius of each click.
    prompts: dict from marconi_sam.read_prompts (its land points give the reference region)."""
    s, X, y, img_id = cfg['scale'], [], [], []
    rad = max(1, int(round(cfg['radius'] * s)))
    dv, du = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    disk = dv**2 + du**2 <= rad**2
    dv, du = dv[disk], du[disk]
    for p, g in labels.groupby('path'):
        shp = load_scaled(p, cfg).shape[:2]
        F = features(p, land_ref(prompts[(g.camera.iloc[0], g.eo_file.iloc[0])], shp, s), cfg)
        for r in g.itertuples():
            vv = np.clip(int(round(r.v * s)) + dv, 0, shp[0] - 1)
            uu = np.clip(int(round(r.u * s)) + du, 0, shp[1] - 1)
            X.append(F[vv, uu])
            y += [r.cls] * len(vv)
            img_id += [p] * len(vv)
    return np.vstack(X), np.array(y), np.array(img_id)


def fit(X, y, cfg=COLOR):
    """Gaussian class model for the classes present in y: dict of class names, products, means (k, n),
    inverse covariances (k, n, n) and log determinants (k)."""
    cls = [c for c in CLASSES if (y == c).any()]
    mu = np.array([X[y == c].mean(0) for c in cls])
    cov = np.array([np.cov(X[y == c].T) + cfg['ridge'] * np.eye(X.shape[1]) for c in cls])
    return {'classes': np.array(cls), 'products': np.array(cfg['products']), 'mu': mu,
            'icov': np.linalg.inv(cov), 'logdet': np.linalg.slogdet(cov)[1]}


def posterior(F, model):
    """Class probabilities (..., k) of feature array F (..., n), equal priors, in the order of model['classes']."""
    d = F[..., None, :] - model['mu']
    ll = -0.5 * (np.einsum('...ki,kij,...kj->...k', d, model['icov'], d) + model['logdet'])
    ll -= ll.max(-1, keepdims=True)
    p = np.exp(ll)
    return p / p.sum(-1, keepdims=True)


def p_dry(post, model):
    """Probability of the dry classes (sand, snow)."""
    return post[..., np.isin(model['classes'], DRY)].sum(-1)


def save_model(path, model):
    np.savez(path, **model)


def load_model(path):
    return dict(np.load(path))


def cross_validate(X, y, img_id, cfg=COLOR):
    """Leave-one-image-out confusion matrix (rows true, columns predicted; fractions of each true class)."""
    pred = np.empty(len(y), dtype=object)
    for p in np.unique(img_id):
        k = img_id == p
        m = fit(X[~k], y[~k], cfg)
        pred[k] = m['classes'][posterior(X[k], m).argmax(-1)]
    cls = [c for c in CLASSES if (y == c).any()]
    cm = pd.crosstab(pd.Categorical(y, cls), pd.Categorical(pred, cls), normalize='index', dropna=False)
    cm.index.name, cm.columns.name = 'true', 'predicted'
    return cm


# ------------------------------------------------------------------ classes, waterline and prompts in one image
def classify(path, prompt, model, cfg=COLOR):
    """Posterior probabilities (nv, nu, k) and polygon mask for one image (scaled) and its period's prompt,
    with the products the model was fit with."""
    shp = load_scaled(path, cfg).shape[:2]
    roi = poly_mask(prompt[2], shp, cfg['scale'])
    F = features(path, land_ref(prompt, shp, cfg['scale']), cfg, tuple(model['products']))
    return posterior(F, model), roi


def pick_component(mask, pts, s, r=15):
    """The connected component of mask with the most pixels within r px (scaled) of pts (full-res u, v)."""
    n, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    near = np.zeros(mask.shape, np.uint8)
    for u, v in np.round(pts * s).astype(int):
        cv2.circle(near, (u, v), r, 1, -1)
    counts = np.bincount(cc[near > 0], minlength=n)
    counts[0] = 0
    return (cc == np.argmax(counts)) if counts.max() > 0 else np.zeros(mask.shape, bool)


def edge_points(sea, roi, s, edge):
    """Edge of the sea region (full-res u, v) away from the boundary of roi."""
    c = max(cv2.findContours(sea.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)[:, 0, :]
    inner = cv2.erode(roi.astype(np.uint8), np.ones((2 * edge + 1,) * 2, np.uint8))
    inner[:, :edge] = inner[:, -edge:] = 0
    inner[-edge:, :] = 0
    c = c[inner[c[:, 1], c[:, 0]] > 0]
    return (c[:, 0] + 0.5) / s - 0.5, (c[:, 1] + 0.5) / s - 0.5


def wet_region(post, roi, water_pts, model, cfg=COLOR):
    """Wet (whitewater or water) region connected to the clicked water points, inside the polygon."""
    return pick_component((p_dry(post, model) < cfg['p_dry']) & roi, water_pts, cfg['scale'])


def color_prompts(wet, roi, land_pts, cfg=COLOR):
    """Per-image SAM prompts (full-res u, v; labels) on either side of the dry/wet boundary: foreground in the
    wet region and background on the dry side, each between band px of the boundary, spread along the beach (bins
    along the line through the first and last clicked land points)."""
    s, (b0, b1), n = cfg['scale'], cfg['band'], cfg['n_color_prompts']
    dw = cv2.distanceTransform(wet.astype(np.uint8), cv2.DIST_L2, 5)
    dl = cv2.distanceTransform((~wet & roi).astype(np.uint8), cv2.DIST_L2, 5)
    dl[~roi] = 0
    a = (land_pts[-1] - land_pts[0]) * s
    a /= np.hypot(*a)
    rng = np.random.default_rng(cfg['seed'])
    out, lab = [], []
    for d, l in [(dw, 1), (dl, 0)]:
        v, u = np.nonzero((d >= b0) & (d <= b1))
        if len(u) == 0:
            continue
        t = u * a[0] + v * a[1]
        k = np.minimum(((t - t.min()) / (np.ptp(t) + 1e-9) * n).astype(int), n - 1)
        for b in np.unique(k):
            j = rng.choice(np.nonzero(k == b)[0])
            out.append([u[j], v[j]])
            lab.append(l)
    return (np.array(out, float).reshape(-1, 2) + 0.5) / s - 0.5, np.array(lab, int)


def color_waterline(path, prompt, model, cfg=COLOR, edge=6):
    """Waterline pixels (full-res u, v) at the dry / wet boundary of one image, from its hand-picked prompt."""
    pts, lab, poly = prompt
    post, roi = classify(path, prompt, model, cfg)
    wet = wet_region(post, roi, pts[lab == 1], model, cfg)
    if not wet.any():
        return np.array([]), np.array([])
    return edge_points(wet, roi, cfg['scale'], edge)


def detect_all_color(scan, out_csv, prompts, model, cfg=COLOR, paths=None, checkpoint=200):
    """Class waterlines for images (quality > 1) in periods with hand-picked prompts; appends (path, u, v) to
    out_csv and resumes if rerun."""
    done = set(pd.read_csv(out_csv, usecols=['path']).path) if Path(out_csv).exists() else set()
    todo = scan[(scan.quality > 1) & ~scan.path.isin(done)
                & pd.Series(list(zip(scan.camera, scan.eo_file)), index=scan.index).isin(prompts)]
    if paths is not None:
        todo = todo[todo.path.isin(paths)]
    todo = todo.reset_index(drop=True)
    rows = []
    for i, r in todo.iterrows():
        u, v = color_waterline(r.path, prompts[(r.camera, r.eo_file)], model, cfg)
        rows.append(pd.DataFrame({'path': r.path, 'u': u if len(u) else [np.nan], 'v': v if len(v) else [np.nan]}))
        if len(rows) == checkpoint or i == len(todo) - 1:
            pd.concat(rows).to_csv(out_csv, mode='a', header=not Path(out_csv).exists(), index=False, float_format='%.1f')
            print(f'{i + 1}/{len(todo)}  {r.time}')
            rows = []


# ------------------------------------------------------------------ choosing and labeling images
def label_images(scan, prompts, worst=(), per_period=4, extra=(), cfg=COLOR):
    """Images to label: the paths in worst and extra plus, for each period with prompts, per_period images
    spread over time (to catch different weather, light and snow) and the one with the largest deep-water
    wave height (quality > 1)."""
    out = [scan[scan.path.isin(list(worst) + list(extra))]]
    for (cam, eo), g in scan[scan.quality > 1].groupby(['camera', 'eo_file']):
        if (cam, eo) not in prompts or len(g) == 0:
            continue
        g = g.sort_values('time')
        k = np.unique(np.linspace(0, len(g) - 1, per_period).astype(int))
        out += [g.iloc[k], g.loc[[g.H0.idxmax()]]]
    return pd.concat(out).drop_duplicates('path').reset_index(drop=True)


def click_labels(r, prompt, wl_list, cfg=COLOR):
    """Collect clicks for each class on one image (left click adds, right click or Backspace removes the last,
    Enter finishes; Enter with no clicks skips the class). wl_list: [(name, color, u, v)] waterlines drawn."""
    fig, ax = plt.subplots(figsize=cfg['figsize'])
    ax.imshow(cv2.cvtColor(cv2.imread(r.path), cv2.COLOR_BGR2RGB))
    pg = prompt[2]
    ax.fill(pg[:, 0], pg[:, 1], fc='none', ec='w', lw=1, ls='--')
    for name, c, u, v in wl_list:
        ax.plot(u, v, '.', c=c, ms=0.6, label=name)
    if wl_list:
        ax.legend(loc='upper right', markerscale=15, fontsize=8)
    ax.set_axis_off()
    rows = []
    for cls in CLASSES:
        ax.set_title(f'{r.camera} {r.time} UTC, H0 {r.H0:.1f} m: click {cls.upper()} points, then Enter '
                     f'(Enter alone if there is none)', fontsize=11)
        fig.canvas.draw()
        uv = np.array(plt.ginput(n=-1, timeout=0, show_clicks=True)).reshape(-1, 2)
        ax.plot(uv[:, 0], uv[:, 1], 'o', mfc=cfg['cls_colors'][cls], mec='k', ms=6)
        rows += [(cls, u, v) for u, v in uv]
    plt.close(fig)
    return pd.DataFrame(rows, columns=['cls', 'u', 'v']).assign(path=r.path, camera=r.camera, eo_file=r.eo_file)


def collect_labels(imgs, prompts, out_csv, wl_groups=(), cfg=COLOR):
    """Click class labels on each image in imgs not yet in out_csv; saved after each image.
    wl_groups: [(name, color, grouped (path, u, v) table)] waterlines drawn for reference."""
    done = pd.read_csv(out_csv) if Path(out_csv).exists() else pd.DataFrame(columns=['path'])
    todo = imgs[~imgs.path.isin(done.path)]
    print(f'{len(todo)} of {len(imgs)} images to label')
    for r in todo.itertuples():
        wl = [(n, c, g.get_group(r.path).u, g.get_group(r.path).v) for n, c, g in wl_groups if r.path in g.groups]
        lab = click_labels(r, prompts[(r.camera, r.eo_file)], wl, cfg)
        done = pd.concat([done, lab], ignore_index=True)
        done.to_csv(out_csv, index=False, float_format='%.1f')
        print(r.camera, r.time, lab.cls.value_counts().to_dict())
    return done


def class_figure(path, rows, prompts, model, wl_groups=(), cfg=COLOR):
    """For each image in rows: the timex with waterlines, and the classes inside the polygon (sand yellow,
    whitewater white, water blue, snow magenta) over a dimmed timex."""
    n = len(rows)
    fig, ax = plt.subplots(n, 2, figsize=(12, 3.6 * n), squeeze=False)
    cmap = plt.matplotlib.colors.ListedColormap([cfg['cls_rgb'][c] for c in model['classes']])
    for i, r in enumerate(rows.itertuples()):
        img = load_scaled(r.path, cfg)
        post, roi = classify(r.path, prompts[(r.camera, r.eo_file)], model, cfg)
        cls = np.where(roi, post.argmax(-1), np.nan)
        ax[i, 0].imshow(cv2.cvtColor(cv2.imread(r.path), cv2.COLOR_BGR2RGB))
        for name, c, g in wl_groups:
            if r.path in g.groups:
                p = g.get_group(r.path)
                ax[i, 0].plot(p.u, p.v, '.', c=c, ms=0.5, label=name)
        s = cfg['scale']
        ax[i, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) // 2, extent=[-0.5 / s, (img.shape[1] - 0.5) / s,
                                                                          (img.shape[0] - 0.5) / s, -0.5 / s])
        ax[i, 1].imshow(cls, cmap=cmap, vmin=-0.5, vmax=len(model['classes']) - 0.5, interpolation='nearest',
                        extent=[-0.5 / s, (img.shape[1] - 0.5) / s, (img.shape[0] - 0.5) / s, -0.5 / s])
        ax[i, 0].set_title(f'{r.camera} {r.time} UTC', fontsize=8)
        for a, l in zip(ax[i], [chr(97 + 2 * i), chr(98 + 2 * i)]):
            a.set_axis_off()
            a.text(0.01, 0.98, l, transform=a.transAxes, va='top', color='w')
    if wl_groups:
        ax[0, 0].legend(loc='upper right', markerscale=15, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return fig


def products_figure(path, rows, prompts, cfg=COLOR):
    """For each image in rows: timex, var and bright images, and the var and bright features inside the polygon."""
    n = len(rows)
    fig, ax = plt.subplots(n, 5, figsize=(20, 3.2 * n), squeeze=False)
    for i, r in enumerate(rows.itertuples()):
        pr = prompts[(r.camera, r.eo_file)]
        shp = load_scaled(r.path, cfg).shape[:2]
        roi = poly_mask(pr[2], shp, cfg['scale'])
        F = features(r.path, land_ref(pr, shp, cfg['scale']), cfg, ('timex', 'var', 'bright'))
        for j, prod in enumerate(['timex', 'var', 'bright']):
            ax[i, j].imshow(cv2.cvtColor(load_scaled(product_path(r.path, prod), cfg), cv2.COLOR_BGR2RGB))
            ax[i, j].set_title(f'{r.camera} {r.time} {prod}', fontsize=8)
        for j, (k, lab, lim) in enumerate([(3, 'ln(sigma/sigma_ref) (-)', (-1, 4)), (4, 'ln(Vbright/V) (-)', (0, 1))]):
            h = ax[i, 3 + j].imshow(np.where(roi, F[..., k], np.nan), cmap='viridis', vmin=lim[0], vmax=lim[1])
            plt.colorbar(h, ax=ax[i, 3 + j], shrink=0.7, label=lab)
        for j, a in enumerate(ax[i]):
            a.set_axis_off()
            a.text(0.02, 0.97, chr(97 + (5 * i + j) % 26), transform=a.transAxes, va='top',
                   bbox={'fc': 'w', 'ec': 'none', 'alpha': 0.7, 'pad': 1})
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    return fig
