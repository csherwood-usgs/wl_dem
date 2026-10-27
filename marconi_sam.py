# csherwood@usgs.gov, 2026-10-27, generated with Claude Opus 5.5
"""
Waterline detection with the Segment Anything Model (SAM; Kirillov et al., 2023), run in parallel with the
Otsu color-index detector in marconi_dem, plus a pixel-distance comparison of the two.

The idea follows ShoreScan (csherwood-usgs/ShoreScan, c_Workflow.py): prompt SAM with points in the sea and
keep the best-scoring mask. Differences from ShoreScan: the model is loaded once, the prompts come from the
calibration geometry (positive points in a band just below the predicted sea horizon, negative points along
the bottom of the image, which is beach), and the waterline is the edge of the sea mask (as for the Otsu
detector) rather than the lowest mask row in each column, so it works for alongshore views.

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

SAM_CFG = md.CFG | {'n_prompt': 12,        # positive (sea) and negative (beach) prompt points per image
                    'neg_margin': 40,      # px (full res) above the image bottom for the negative points
                    'u_margin': 0.05}      # fraction of image width excluded at each side for prompts


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


def detect_waterline_sam(path, io, eo, predictor, cfg=SAM_CFG):
    """Waterline pixels (full-res distorted u, v) from the SAM sea mask, and the mask score."""
    s = cfg['scale']
    img = cv2.resize(cv2.imread(path), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    nv, nu = img.shape[:2]
    pts, lab = sam_prompts(io, eo, nu / s, nv / s, cfg)
    if lab.sum() == 0:
        return np.array([]), np.array([]), np.nan
    predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(point_coords=pts * s, point_labels=lab, multimask_output=True)
    k = np.argmax(scores)
    hz = ms.horizon_rows(io, eo, np.arange(nu) / s)
    top = np.nan_to_num(hz, nan=-cfg['hz_margin']) * s + cfg['hz_margin'] * s
    roi = np.arange(nv)[:, None] > top[None, :]
    n, cc = cv2.connectedComponents((masks[k] & roi).astype(np.uint8), connectivity=8)
    pp = np.round(pts[lab == 1] * s).astype(int)
    counts = np.bincount(cc[np.clip(pp[:, 1], 0, nv - 1), np.clip(pp[:, 0], 0, nu - 1)], minlength=n)
    counts[0] = 0
    if counts.max() == 0:
        return np.array([]), np.array([]), scores[k]
    sea = (cc == np.argmax(counts)).astype(np.uint8)
    c = max(cv2.findContours(sea, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)[:, 0, :]
    e = cfg['edge']
    inner = cv2.erode(roi.astype(np.uint8), np.ones((2 * e + 1,) * 2, np.uint8))
    inner[:, :e] = inner[:, -e:] = 0
    inner[-e:, :] = 0
    c = c[inner[c[:, 1], c[:, 0]] > 0]
    return (c[:, 0] + 0.5) / s - 0.5, (c[:, 1] + 0.5) / s - 0.5, scores[k]


def detect_all_sam(scan, calib_dir, out_csv, predictor, cfg=SAM_CFG, paths=None, checkpoint=20):
    """SAM waterlines for images with quality > 1 (or only those in paths); appends (path, u, v) to out_csv
    and resumes if rerun. Returns per-image mask scores and point counts of this run."""
    done = set(pd.read_csv(out_csv, usecols=['path']).path) if Path(out_csv).exists() else set()
    todo = scan[(scan.quality > 1) & ~scan.path.isin(done)]
    if paths is not None:
        todo = todo[todo.path.isin(paths)]
    todo = todo.reset_index(drop=True)
    cal, rows, info = {}, [], []
    for i, r in todo.iterrows():
        if r.eo_file not in cal:
            cal[r.eo_file] = (ms.read_yaml(Path(calib_dir, r.io_file)), ms.read_yaml(Path(calib_dir, r.eo_file)))
        u, v, sc = detect_waterline_sam(r.path, *cal[r.eo_file], predictor, cfg)
        rows.append(pd.DataFrame({'path': r.path, 'u': u if len(u) else [np.nan], 'v': v if len(v) else [np.nan]}))
        info.append((r.path, sc, len(u)))
        if len(rows) == checkpoint or i == len(todo) - 1:
            pd.concat(rows).to_csv(out_csv, mode='a', header=not Path(out_csv).exists(), index=False, float_format='%.1f')
            print(f'{i + 1}/{len(todo)}  {r.time}')
            rows = []
    return pd.DataFrame(info, columns=['path', 'sam_score', 'sam_npx'])


def compare_waterlines(wl_a, wl_b, paths):
    """Per-image distance (full-res px) from each waterline pixel of a to the nearest pixel of b, and b to a.
    wl_a, wl_b: grouped (path, u, v) tables. Returns median and 90th percentile of each and point counts."""
    out = []
    for p in paths:
        a = wl_a.get_group(p)[['u', 'v']].dropna().values if p in wl_a.groups else np.empty((0, 2))
        b = wl_b.get_group(p)[['u', 'v']].dropna().values if p in wl_b.groups else np.empty((0, 2))
        if len(a) == 0 or len(b) == 0:
            out.append((p, len(a), len(b)) + (np.nan,) * 4)
            continue
        dab, dba = cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]
        out.append((p, len(a), len(b), np.median(dab), np.percentile(dab, 90), np.median(dba), np.percentile(dba, 90)))
    return pd.DataFrame(out, columns=['path', 'n_otsu', 'n_sam', 'd_otsu_sam_med', 'd_otsu_sam_p90',
                                      'd_sam_otsu_med', 'd_sam_otsu_p90'])
