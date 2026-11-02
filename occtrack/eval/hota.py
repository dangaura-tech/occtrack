"""HOTA: Higher Order Tracking Accuracy (Luiten et al., IJCV 2021).

A faithful NumPy port of the TrackEval implementation for box tracking:

* similarity = IoU; a global alignment score between every (gt id, tracker id) pair is used
  to break ties so that per-frame matching prefers associations that are good over time;
* for each localisation threshold alpha in {0.05, 0.10, ..., 0.95}:
  ``DetA = TP / (TP + FN + FP)``, ``AssA = mean over TPs of TPA / (TPA + FNA + FPA)``,
  ``HOTA = sqrt(DetA * AssA)``; LocA is the mean IoU of the matches;
* final scores are averages over alpha.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..boxes import iou_matrix

ALPHAS = np.arange(0.05, 0.99, 0.05)
EPS = np.finfo(float).eps


def hota(gt: Dict[int, np.ndarray], pred: Dict[int, np.ndarray]) -> Dict[str, float]:
    """``gt`` / ``pred``: ``{frame: array[K, >=5]}`` with columns ``id, x1, y1, x2, y2``."""
    frames = sorted(set(gt) | set(pred))
    empty = np.zeros((0, 5))
    gt_ids = sorted({int(i) for f in gt for i in gt[f][:, 0]})
    tr_ids = sorted({int(i) for f in pred for i in pred[f][:, 0]})
    gmap = {g: k for k, g in enumerate(gt_ids)}
    tmap = {t: k for k, t in enumerate(tr_ids)}
    n_g, n_t = len(gt_ids), len(tr_ids)

    seq = []
    for f in frames:
        g = gt.get(f, empty)
        p = pred.get(f, empty)
        gi = np.array([gmap[int(i)] for i in g[:, 0]], dtype=int)
        ti = np.array([tmap[int(i)] for i in p[:, 0]], dtype=int)
        seq.append((gi, ti, iou_matrix(g[:, 1:5], p[:, 1:5])))

    # global alignment score
    potential = np.zeros((n_g, n_t))
    gt_count = np.zeros((n_g, 1))
    tr_count = np.zeros((1, n_t))
    for gi, ti, sim in seq:
        if len(gi) and len(ti):
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.zeros_like(sim)
            ok = denom > EPS
            sim_iou[ok] = sim[ok] / denom[ok]
            potential[gi[:, None], ti[None, :]] += sim_iou
        gt_count[gi] += 1
        tr_count[0, ti] += 1
    gas = potential / np.maximum(gt_count + tr_count - potential, EPS)

    n_a = len(ALPHAS)
    tp, fn, fp, loc = np.zeros(n_a), np.zeros(n_a), np.zeros(n_a), np.zeros(n_a)
    matches = np.zeros((n_a, n_g, n_t))
    for gi, ti, sim in seq:
        if not len(gi):
            fp += len(ti)
            continue
        if not len(ti):
            fn += len(gi)
            continue
        score = gas[gi[:, None], ti[None, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = sim[rows, cols] >= alpha - EPS
            r, c = rows[ok], cols[ok]
            n = len(r)
            tp[a] += n
            fn[a] += len(gi) - n
            fp[a] += len(ti) - n
            if n:
                loc[a] += sim[r, c].sum()
                matches[a, gi[r], ti[c]] += 1

    det_a = tp / np.maximum(tp + fn + fp, 1)
    ass_a = np.zeros(n_a)
    for a in range(n_a):
        m = matches[a]
        ass = m / np.maximum(gt_count + tr_count - m, 1)
        ass_a[a] = (m * ass).sum() / max(tp[a], 1)
    hota_a = np.sqrt(det_a * ass_a)
    loc_a = np.where(tp > 0, loc / np.maximum(tp, 1), 1.0)
    return {
        "HOTA": float(hota_a.mean()),
        "DetA": float(det_a.mean()),
        "AssA": float(ass_a.mean()),
        "LocA": float(loc_a.mean()),
        "HOTA(0)": float(hota_a[0]),
    }
