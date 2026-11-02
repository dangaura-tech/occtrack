"""Synthetic crowd scenes with controllable occlusion, for testing and ablations.

Pedestrians walk along straight lines with small random accelerations. A simulated
detector observes them with box jitter, confidence that decreases with occlusion,
occlusion-dependent miss rate and random false positives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .boxes import occlusion_ratio


@dataclass
class SceneConfig:
    num_frames: int = 300
    num_targets: int = 15
    width: int = 1920
    height: int = 1080
    speed: float = 6.0
    box_jitter: float = 0.04
    miss_rate: float = 0.05
    occluded_miss_rate: float = 0.6
    false_positives_per_frame: float = 0.5
    seed: int = 0


def simulate_scene(cfg: SceneConfig) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return ``(gt, detections)``.

    * ``gt[frame]``   -> array[K, 7]: ``id, x1, y1, x2, y2, 1, visibility``
    * ``dets[frame]`` -> array[K, 7]: ``-1, x1, y1, x2, y2, score, 1``
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.num_targets
    h = rng.uniform(120, 260, n)
    w = h * rng.uniform(0.35, 0.45, n)
    pos = np.c_[rng.uniform(0, cfg.width, n), rng.uniform(h, cfg.height, n)]  # bottom-centre
    angle = rng.uniform(0, 2 * np.pi, n)
    vel = np.c_[np.cos(angle), 0.3 * np.sin(angle)] * rng.uniform(0.5, 1.0, (n, 1)) * cfg.speed
    gt: Dict[int, np.ndarray] = {}
    dets: Dict[int, np.ndarray] = {}
    for f in range(1, cfg.num_frames + 1):
        vel += rng.normal(0, 0.15, vel.shape)
        pos += vel
        # bounce on image borders
        for d, lim in ((0, cfg.width), (1, cfg.height)):
            lo = h if d == 1 else 0
            out = (pos[:, d] < lo) | (pos[:, d] > lim)
            vel[out, d] *= -1
            pos[:, d] = np.clip(pos[:, d], lo, lim)
        boxes = np.c_[pos[:, 0] - w / 2, pos[:, 1] - h, pos[:, 0] + w / 2, pos[:, 1]]
        occ = occlusion_ratio(boxes)
        gt[f] = np.c_[np.arange(1, n + 1), boxes, np.ones(n), 1 - occ]

        miss_p = cfg.miss_rate + (cfg.occluded_miss_rate - cfg.miss_rate) * occ
        seen = rng.random(n) > miss_p
        jitter = rng.normal(0, cfg.box_jitter, (n, 4)) * np.c_[w, h, w, h]
        db = boxes[seen] + jitter[seen]
        score = np.clip(0.95 - 0.8 * occ[seen] + rng.normal(0, 0.05, seen.sum()), 0.05, 1.0)
        k = rng.poisson(cfg.false_positives_per_frame)
        if k:
            fh = rng.uniform(100, 250, k)
            fx = rng.uniform(0, cfg.width, k)
            fy = rng.uniform(fh, cfg.height, k)
            fp = np.c_[fx - 0.2 * fh, fy - fh, fx + 0.2 * fh, fy]
            db = np.r_[db, fp]
            score = np.r_[score, rng.uniform(0.1, 0.75, k)]
        dets[f] = np.c_[-np.ones(len(db)), db, score, np.ones(len(db))]
    return gt, dets
