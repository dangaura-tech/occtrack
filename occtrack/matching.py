"""Cost matrices and linear assignment."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .boxes import iou_matrix, xyxy_to_xyah
from .kalman import CHI2INV95
from .track import Track

INF_COST = 1e5


def linear_assignment(cost: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian matching that rejects pairs with cost above ``thresh``."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    return (
        matches,
        [r for r in range(cost.shape[0]) if r not in matched_r],
        [c for c in range(cost.shape[1]) if c not in matched_c],
    )


def iou_distance(tracks: Sequence[Track], boxes: np.ndarray) -> np.ndarray:
    if not len(tracks) or not len(boxes):
        return np.zeros((len(tracks), len(boxes)))
    return 1.0 - iou_matrix(np.stack([t.xyxy for t in tracks]), boxes)


def fuse_score(cost: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Down-weight the IoU similarity of low-confidence detections (ByteTrack fuse_score)."""
    if cost.size == 0:
        return cost
    return 1.0 - (1.0 - cost) * scores[None, :]


def embedding_distance(tracks: Sequence[Track], feats: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if feats is None or not len(tracks) or not len(feats):
        return None
    if any(t.feature is None for t in tracks):
        return None
    t = np.stack([tr.feature for tr in tracks])
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    return np.clip(1.0 - t @ f.T, 0.0, 2.0) / 2.0


def gate_by_motion(
    cost: np.ndarray, tracks: Sequence[Track], boxes: np.ndarray, only_position: bool = False
) -> np.ndarray:
    """Set the cost to infinity where the Mahalanobis distance exceeds the chi-square 95% gate."""
    if cost.size == 0:
        return cost
    cost = cost.copy()
    z = xyxy_to_xyah(boxes)
    thresh = CHI2INV95[2 if only_position else 4]
    for i, t in enumerate(tracks):
        d = t.kf.gating_distance(t.mean, t.cov, z, only_position)
        cost[i, d > thresh] = INF_COST
    return cost


def direction_cost(tracks: Sequence[Track], boxes: np.ndarray, scores: np.ndarray, delta: int = 3) -> np.ndarray:
    """Observation-centric momentum (OC-SORT): angular difference between each track's observed
    direction of motion and the direction from its last observation to each detection,
    normalised to [0, 1] and weighted by detection confidence."""
    cost = np.zeros((len(tracks), len(boxes)))
    if not len(tracks) or not len(boxes):
        return cost
    centres = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
    for i, t in enumerate(tracks):
        d_track = t.observed_direction(delta)
        if d_track is None:
            continue
        v = centres - t.last_observation[:2]
        n = np.linalg.norm(v, axis=1)
        valid = n > 1e-6
        cos = np.ones(len(boxes))
        cos[valid] = (v[valid] @ d_track) / n[valid]
        cost[i] = np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi * scores
    return cost


def reid_cost(tracks: Sequence[Track], boxes: np.ndarray, feats: np.ndarray, spatial_sigma: float = 3.0) -> np.ndarray:
    """Long-term re-identification cost: prototype-memory appearance distance, gated by a
    spatial window that widens with the track's positional uncertainty."""
    if not len(tracks) or not len(boxes):
        return np.zeros((len(tracks), len(boxes)))
    centres = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
    cost = np.full((len(tracks), len(boxes)), INF_COST)
    for i, t in enumerate(tracks):
        dist = np.linalg.norm(centres - t.mean[:2], axis=1)
        radius = spatial_sigma * t.position_std() + t.mean[3]
        ok = dist <= radius
        if ok.any():
            cost[i, ok] = t.appearance_distance(feats[ok])
    return cost
