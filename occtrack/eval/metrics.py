"""CLEAR-MOT (MOTA, MOTP, IDSW, FP, FN, MT/ML) and identity (IDF1, IDP, IDR) metrics.

This is a compact re-implementation intended for fast development loops and unit tests.
For leaderboard numbers use the official TrackEval toolkit (`scripts/run_trackeval.sh`).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..boxes import iou_matrix


class MOTAccumulator:
    def __init__(self, iou_thresh: float = 0.5) -> None:
        self.iou_thresh = iou_thresh
        self.num_gt = 0
        self.num_pred = 0
        self.fp = 0
        self.fn = 0
        self.idsw = 0
        self.frag = 0
        self.iou_sum = 0.0
        self.num_matches = 0
        self._last_match: Dict[int, int] = {}  # gt id -> last matched pred id (ever)
        self._prev_frame: Dict[int, int] = {}  # gt id -> pred id in previous frame
        self._gt_frames: Dict[int, int] = defaultdict(int)
        self._gt_tracked: Dict[int, int] = defaultdict(int)
        self._gt_was_tracked: Dict[int, bool] = {}
        # identity bookkeeping: frames where (gt, pred) are matched by IoU (independent of CLEAR matching)
        self._pair_overlap: Dict[tuple, int] = defaultdict(int)
        self._gt_count: Dict[int, int] = defaultdict(int)
        self._pred_count: Dict[int, int] = defaultdict(int)

    def update(self, gt_ids: np.ndarray, gt_boxes: np.ndarray, pred_ids: np.ndarray, pred_boxes: np.ndarray) -> None:
        gt_ids = np.asarray(gt_ids, dtype=int).reshape(-1)
        pred_ids = np.asarray(pred_ids, dtype=int).reshape(-1)
        iou = iou_matrix(gt_boxes, pred_boxes)
        self.num_gt += len(gt_ids)
        self.num_pred += len(pred_ids)

        # identity counts
        for g in gt_ids:
            self._gt_count[g] += 1
            self._gt_frames[g] += 1
        for p in pred_ids:
            self._pred_count[p] += 1
        gi, pi = np.nonzero(iou >= self.iou_thresh)
        for a, b in zip(gi, pi):
            self._pair_overlap[(gt_ids[a], pred_ids[b])] += 1

        # CLEAR-MOT matching: keep previous correspondences when still valid
        matches = []
        used_g, used_p = set(), set()
        pred_index = {p: j for j, p in enumerate(pred_ids)}
        for i, g in enumerate(gt_ids):
            p = self._prev_frame.get(g)
            if p is not None and p in pred_index:
                j = pred_index[p]
                if j not in used_p and iou[i, j] >= self.iou_thresh:
                    matches.append((i, j))
                    used_g.add(i)
                    used_p.add(j)
        free_g = [i for i in range(len(gt_ids)) if i not in used_g]
        free_p = [j for j in range(len(pred_ids)) if j not in used_p]
        if free_g and free_p:
            sub = iou[np.ix_(free_g, free_p)]
            cost = np.where(sub >= self.iou_thresh, 1.0 - sub, 1e6)
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if cost[a, b] < 1e6:
                    matches.append((free_g[a], free_p[b]))

        current: Dict[int, int] = {}
        for i, j in matches:
            g, p = gt_ids[i], pred_ids[j]
            last = self._last_match.get(g)
            if last is not None and last != p:
                self.idsw += 1
            if self._gt_was_tracked.get(g) is False:
                self.frag += 1
            self._last_match[g] = p
            current[g] = p
            self._gt_tracked[g] += 1
            self.iou_sum += iou[i, j]
            self.num_matches += 1
        for g in gt_ids:
            self._gt_was_tracked[g] = g in current
        self.fn += len(gt_ids) - len(matches)
        self.fp += len(pred_ids) - len(matches)
        self._prev_frame = current

    def compute(self) -> Dict[str, float]:
        gts = sorted(self._gt_count)
        preds = sorted(self._pred_count)
        idtp = 0
        if gts and preds:
            gi = {g: k for k, g in enumerate(gts)}
            pj = {p: k for k, p in enumerate(preds)}
            overlap = np.zeros((len(gts), len(preds)))
            for (g, p), n in self._pair_overlap.items():
                overlap[gi[g], pj[p]] = n
            r, c = linear_sum_assignment(-overlap)
            idtp = int(overlap[r, c].sum())
        idfn = self.num_gt - idtp
        idfp = self.num_pred - idtp
        ratios = [self._gt_tracked[g] / self._gt_frames[g] for g in self._gt_frames]
        return {
            "MOTA": 1.0 - (self.fn + self.fp + self.idsw) / max(self.num_gt, 1),
            "MOTP": self.iou_sum / max(self.num_matches, 1),
            "IDF1": 2 * idtp / max(self.num_gt + self.num_pred, 1),
            "IDP": idtp / max(idtp + idfp, 1),
            "IDR": idtp / max(idtp + idfn, 1),
            "IDSW": self.idsw,
            "Frag": self.frag,
            "FP": self.fp,
            "FN": self.fn,
            "MT": int(sum(r >= 0.8 for r in ratios)),
            "ML": int(sum(r < 0.2 for r in ratios)),
            "GT": self.num_gt,
            "num_gt_ids": len(gts),
        }


def evaluate_sequence(
    gt: Dict[int, np.ndarray], pred: Dict[int, np.ndarray], iou_thresh: float = 0.5
) -> Dict[str, float]:
    """Evaluate dicts ``{frame: array[K, >=5]}`` with columns ``id, x1, y1, x2, y2, ...``."""
    acc = MOTAccumulator(iou_thresh)
    empty = np.zeros((0, 5))
    for f in sorted(set(gt) | set(pred)):
        g = gt.get(f, empty)
        p = pred.get(f, empty)
        acc.update(g[:, 0], g[:, 1:5], p[:, 0], p[:, 1:5])
    return acc.compute()


def summarize(per_seq: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Combine per-sequence results: counts are summed, MOTA is recomputed, IDF1/MOTP/IDP/IDR are GT-weighted."""
    keys_sum = ["IDSW", "Frag", "FP", "FN", "MT", "ML", "GT", "num_gt_ids"]
    total = {k: sum(r[k] for r in per_seq.values()) for k in keys_sum}
    total["MOTA"] = 1.0 - (total["FN"] + total["FP"] + total["IDSW"]) / max(total["GT"], 1)
    w = np.array([r["GT"] for r in per_seq.values()], dtype=float)
    for k in ("IDF1", "MOTP", "IDP", "IDR"):
        v = np.array([r[k] for r in per_seq.values()])
        total[k] = float((v * w).sum() / max(w.sum(), 1))
    return total


def format_table(rows: Iterable[tuple]) -> str:
    cols = ["MOTA", "IDF1", "MOTP", "IDSW", "FP", "FN", "MT", "ML"]
    lines = ["| Sequence | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in rows:
        cells = [f"{r[c] * 100:.1f}" if c in ("MOTA", "IDF1", "MOTP") else str(int(r[c])) for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
