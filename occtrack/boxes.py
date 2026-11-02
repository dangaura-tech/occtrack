"""Bounding-box conversions and vectorised IoU."""

from __future__ import annotations

import numpy as np


def tlwh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).copy()
    b[..., 2:4] += b[..., 0:2]
    return b


def xyxy_to_tlwh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).copy()
    b[..., 2:4] -= b[..., 0:2]
    return b


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    """(x1, y1, x2, y2) -> (cx, cy, aspect = w / h, h), the Kalman measurement space."""
    b = np.asarray(b, dtype=np.float64)
    w = b[..., 2] - b[..., 0]
    h = b[..., 3] - b[..., 1]
    return np.stack([b[..., 0] + w / 2, b[..., 1] + h / 2, w / np.maximum(h, 1e-6), h], axis=-1)


def xyah_to_xyxy(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    h = s[..., 3]
    w = s[..., 2] * h
    return np.stack([s[..., 0] - w / 2, s[..., 1] - h / 2, s[..., 0] + w / 2, s[..., 1] + h / 2], axis=-1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``a`` (N, 4) and ``b`` (M, 4) boxes in xyxy format."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def occlusion_ratio(boxes: np.ndarray) -> np.ndarray:
    """Fraction of each box covered by any *other* box lying in front of it.

    Following the common pedestrian-scene heuristic, a box whose bottom edge is lower in
    the image is assumed to be closer to the camera.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    n = len(boxes)
    ratio = np.zeros(n)
    if n < 2:
        return ratio
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    lt = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    in_front = boxes[None, :, 3] > boxes[:, None, 3]
    np.fill_diagonal(in_front, False)
    covered = np.where(in_front, inter, 0.0).max(axis=1)
    ratio = covered / np.maximum(area, 1e-9)
    return np.clip(ratio, 0.0, 1.0)
