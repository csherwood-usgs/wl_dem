# csherwood@usgs.gov, 2026-10-27, generated with Claude Opus 5.5 (hand-picked and color prompts added 2026-10-28)
"""
Waterline detection with the Segment Anything Model (SAM; Kirillov et al., 2023), run in parallel with the
Otsu color-index detector in marconi_dem, plus a pixel-distance comparison of the two.

The idea follows ShoreScan (csherwood-usgs/ShoreScan, c_Workflow.py): prompt SAM with points in the sea and
keep the best-scoring mask. Differences from ShoreScan: the model is loaded once, the prompts come from the
calibration geometry (positive points in a band just below the predicted sea horizon, negative points along
the bottom of the image, which is beach), and the waterline is the edge of the sea mask (as for the Otsu
detector) rather than the lowest mask row in each column, so it works for alongshore views.

Hand-picked prompts (marconi_prompts / sam_prompts.ipynb) replace the geometric ones for the EO periods they
cover: land points are background, water points foreground, and the polygon they outline limits where the
waterline may be. Periods without hand-picked prompts fall back to the geometric prompts.

With a color model (marconi_color) as well, each image also gets prompts on either side of its own
sand / (whitewater + water) color boundary, SAM returns its three candidate masks, and the one that agrees best
with the color classes (intersection over union of the wet region inside the polygon) is kept. This keeps
whitewater with the sea, so the waterline is not drawn at the outer edge of the surf zone.

Output has the same (path, u, v) format as marconi_dem.detect_all, so marconi_dem can build DEMs from
either file.

Requires torch, torchvision, segment_anything and a SAM checkpoint (see qc_waterlines.ipynb).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree
import marconi_scan as ms
import marconi_dem as md
import marconi_prompts as mp
import marconi_color as mc

SAM_CFG = md.CFG | {'n_prompt': 12,        # positive (sea) and negative (beach) prompt points per image
                    'neg_margin': 40,      # px (full res) above the image bottom for the negative points
                    'u_margin': 0.05,      # fraction of image width excluded at each side for prompts
                    'multimask': False}    # SAM multimask output with hand-picked prompts (many points: False)


def read_prompts(path):
    """Hand-picked prompts: dict (camera, eo_file) -> (points (n, 2) full-res u, v; SAM labels; polygon (m, 2))."""
    p = pd.read_csv(path)
    return {k: (g[['u', 'v']].values, g.sam_label.values.astype(int), mp.polygon(g))
            for k, g in p.groupby(['camera', 'eo_file'])}


def load_sam(checkpoint, model_type='vit_h', device=None):
    """SamPredictor for a checkpoint (vit_h, vit_l or vit_b), on the GPU if one is available."""
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    return SamPredictor(sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device))


def sam_prompts(io, eo, nu, nv, cfg=SAM_CFG):
    """Prompt points (full res u, v) and labels: sea just below the horizon (1) and beach at the bottom (0)."""
    u = np.linspace(cfg['u_margin'] * nu, (1 - cfg['u_margin']) * nu, cfg['n_prompt'])
    vs = ms.horizon_rows(io, eo, u) + cfg['hz_margin'] + cfg['seed_band'] / 2
    ok = np.isfinite(vs) & (vs < nv - 2 * cfg['neg_margin'])
    pos = np.column_stack([u[ok], vs[ok]])
    neg = np.column_stack([u, np.full_like(u, nv - cfg['neg_margin'])])
    return np.vstack([pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(int)


def detect_waterline_sam(path, io, eo, predictor, cfg=SAM_CFG, prompt=None, model=None, ccfg=mc.COLOR):
    """Waterline pixels (full-res distorted u, v) from the SAM sea mask, the mask score, and the agreement (IoU)
    of the mask with the color classes (NaN without a color model).
    prompt: optional hand-picked (points, labels, polygon) from read_prompts; the waterline is then limited
    to the polygon, otherwise to the region below the predicted horizon. model: optional color model, used
    with prompt to add per-image prompts and choose among SAM's candidate masks."""
    s = cfg['scale']
    img = cv2.resize(cv2.imread(path), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    nv, nu = img.shape[:2]
    wet = None
    if prompt is None:
        pts, lab = sam_prompts(io, eo, nu / s, nv / s, cfg)
        hz = ms.horizon_rows(io, eo, np.arange(nu) / s)
        top = np.nan_to_num(hz, nan=-cfg['hz_margin']) * s + cfg['hz_margin'] * s
        roi = np.arange(nv)[:, None] > top[None, :]
        multi = True
    else:
        pts, lab, poly = prompt
        roi = mc.poly_mask(poly, (nv, nu), s)
        multi = cfg['multimask']
        if model is not None:
            post, _ = mc.classify(img, prompt, model, ccfg | {'scale': s})
            wet = mc.wet_region(post, roi, pts[lab == 1], ccfg | {'scale': s})
            cp, cl = mc.color_prompts(wet, roi, pts[lab == 0], ccfg | {'scale': s})
            pts, lab, multi = np.vstack([pts, cp]), np.r_[lab, cl], True
    if lab.sum() == 0:
        return np.array([]), np.array([]), np.nan, np.nan
    predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(point_coords=pts * s, point_labels=lab, multimask_output=multi)
    if wet is None:
        k, iou = np.argmax(scores), np.nan
    else:
        ious = [((m & roi) & wet).sum() / max(((m & roi) | wet).sum(), 1) for m in masks]
        k, iou = np.argmax(ious), np.max(ious)
    sea = mc.pick_component(masks[k] & roi, pts[lab == 1], s)
    if not sea.any():
        return np.array([]), np.array([]), scores[k], iou
    u, v = mc.edge_points(sea, roi, s, cfg['edge'])
    return u, v, scores[k], iou


def detect_all_sam(scan, calib_dir, out_csv, predictor, cfg=SAM_CFG, paths=None, checkpoint=20, prompts=None,
                   model=None, ccfg=mc.COLOR):
    """SAM waterlines for images with quality > 1 (or only those in paths); appends (path, u, v) to out_csv
    and resumes if rerun. prompts: optional dict from read_prompts; model: optional color model. Returns
    per-image mask score, agreement with the color classes, point count, and whether hand-picked prompts
    were used."""
    done = set(pd.read_csv(out_csv, usecols=['path']).path) if Path(out_csv).exists() else set()
    todo = scan[(scan.quality > 1) & ~scan.path.isin(done)]
    if paths is not None:
        todo = todo[todo.path.isin(paths)]
    todo = todo.reset_index(drop=True)
    cal, rows, info = {}, [], []
    for i, r in todo.iterrows():
        if r.eo_file not in cal:
            cal[r.eo_file] = (ms.read_yaml(Path(calib_dir, r.io_file)), ms.read_yaml(Path(calib_dir, r.eo_file)))
        pr = (prompts or {}).get((r.camera, r.eo_file))
        u, v, sc, iou = detect_waterline_sam(r.path, *cal[r.eo_file], predictor, cfg, pr, model, ccfg)
        rows.append(pd.DataFrame({'path': r.path, 'u': u if len(u) else [np.nan], 'v': v if len(v) else [np.nan]}))
        info.append((r.path, sc, iou, len(u), pr is not None))
        if len(rows) == checkpoint or i == len(todo) - 1:
            pd.concat(rows).to_csv(out_csv, mode='a', header=not Path(out_csv).exists(), index=False, float_format='%.1f')
            print(f'{i + 1}/{len(todo)}  {r.time}')
            rows = []
    return pd.DataFrame(info, columns=['path', 'sam_score', 'color_iou', 'sam_npx', 'hand_prompts'])


def compare_waterlines(wl_a, wl_b, paths, names=('otsu', 'sam')):
    """Per-image distance (full-res px) from each waterline pixel of a to the nearest pixel of b, and b to a.
    wl_a, wl_b: grouped (path, u, v) tables; names label the columns. Returns median and 90th percentile of
    each and point counts."""
    out = []
    for p in paths:
        a = wl_a.get_group(p)[['u', 'v']].dropna().values if p in wl_a.groups else np.empty((0, 2))
        b = wl_b.get_group(p)[['u', 'v']].dropna().values if p in wl_b.groups else np.empty((0, 2))
        if len(a) == 0 or len(b) == 0:
            out.append((p, len(a), len(b)) + (np.nan,) * 4)
            continue
        dab, dba = cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]
        out.append((p, len(a), len(b), np.median(dab), np.percentile(dab, 90), np.median(dba), np.percentile(dba, 90)))
    a, b = names
    return pd.DataFrame(out, columns=['path', f'n_{a}', f'n_{b}', f'd_{a}_{b}_med', f'd_{a}_{b}_p90',
                                      f'd_{b}_{a}_med', f'd_{b}_{a}_p90'])
