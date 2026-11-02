"""Synthetic crowd scenes with controllable occlusion, camera motion and appearance.

Pedestrians walk along straight lines with small random accelerations in a world larger
than the image. The camera pans (with jitter), so objects move in image space even when
standing still. A simulated detector observes the visible targets with box jitter, a
confidence that decreases with occlusion, an occlusion-dependent miss rate and random false
positives, and a simulated Re-ID network returns embeddings that are corrupted by occluders.
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
    camera_pan: float = 0.0  # mean horizontal camera speed (px / frame)
    camera_jitter: float = 0.0  # std of per-frame camera shake (px)
    feature_dim: int = 0  # >0 also simulates Re-ID embeddings
    feature_noise: float = 0.3
    seed: int = 0


@dataclass
class Scene:
    gt: Dict[int, np.ndarray]  # frame -> [id, x1, y1, x2, y2, 1, visibility]
    dets: Dict[int, np.ndarray]  # frame -> [-1, x1, y1, x2, y2, score, 1]
    features: Dict[int, np.ndarray]  # frame -> (K, D) embeddings aligned with dets (empty if D == 0)
    warps: Dict[int, np.ndarray]  # frame -> 2x3 affine from the previous frame (identity for frame 1)


def simulate(cfg: SceneConfig) -> Scene:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.num_targets
    world_w = cfg.width + abs(cfg.camera_pan) * cfg.num_frames + 400
    h = rng.uniform(120, 260, n)
    w = h * rng.uniform(0.35, 0.45, n)
    x0 = 0.0 if cfg.camera_pan >= 0 else world_w - cfg.width
    pos = np.c_[rng.uniform(x0, x0 + cfg.width, n), rng.uniform(h, cfg.height, n)]  # bottom-centre, world
    angle = rng.uniform(0, 2 * np.pi, n)
    vel = np.c_[np.cos(angle), 0.3 * np.sin(angle)] * rng.uniform(0.5, 1.0, (n, 1)) * cfg.speed
    ident = rng.normal(size=(n, max(cfg.feature_dim, 1)))
    ident /= np.linalg.norm(ident, axis=1, keepdims=True)

    cam = np.array([x0, 0.0])
    gt, dets, feats, warps = {}, {}, {}, {}
    for f in range(1, cfg.num_frames + 1):
        prev_cam = cam.copy()
        if f > 1:
            cam = cam + np.array([cfg.camera_pan, 0.0]) + rng.normal(0, cfg.camera_jitter, 2)
        # image = world - cam  =>  x_t = x_{t-1} - (cam_t - cam_{t-1})
        shift = -(cam - prev_cam)
        warps[f] = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])

        vel += rng.normal(0, 0.15, vel.shape)
        pos += vel
        for d, lo, hi in ((0, 0.0, world_w), (1, None, cfg.height)):
            lo_d = h if d == 1 else lo
            out = (pos[:, d] < lo_d) | (pos[:, d] > hi)
            vel[out, d] *= -1
            pos[:, d] = np.clip(pos[:, d], lo_d, hi)
        img = pos - cam
        boxes = np.c_[img[:, 0] - w / 2, img[:, 1] - h, img[:, 0] + w / 2, img[:, 1]]
        visible = (boxes[:, 2] > 0) & (boxes[:, 0] < cfg.width) & (boxes[:, 3] > 0) & (boxes[:, 1] < cfg.height)
        occ_all = occlusion_ratio(boxes)
        vis_idx = np.flatnonzero(visible)
        gt[f] = np.c_[vis_idx + 1, boxes[vis_idx], np.ones(len(vis_idx)), 1 - occ_all[vis_idx]]

        occ = occ_all[vis_idx]
        miss_p = cfg.miss_rate + (cfg.occluded_miss_rate - cfg.miss_rate) * occ
        seen = vis_idx[rng.random(len(vis_idx)) > miss_p]
        occ_seen = occ_all[seen]
        jitter = rng.normal(0, cfg.box_jitter, (len(seen), 4)) * np.c_[w[seen], h[seen], w[seen], h[seen]]
        db = boxes[seen] + jitter
        score = np.clip(0.95 - 0.8 * occ_seen + rng.normal(0, 0.05, len(seen)), 0.05, 1.0)
        fe = ident[seen] * (1 - occ_seen[:, None]) + rng.normal(size=(len(seen), ident.shape[1])) * (
            cfg.feature_noise + occ_seen[:, None]
        ) / np.sqrt(ident.shape[1])
        k = rng.poisson(cfg.false_positives_per_frame)
        if k:
            fh = rng.uniform(100, 250, k)
            fx = rng.uniform(0, cfg.width, k)
            fy = rng.uniform(fh, cfg.height, k)
            db = np.r_[db, np.c_[fx - 0.2 * fh, fy - fh, fx + 0.2 * fh, fy]]
            score = np.r_[score, rng.uniform(0.1, 0.75, k)]
            fe = np.r_[fe, rng.normal(size=(k, ident.shape[1]))]
        dets[f] = np.c_[-np.ones(len(db)), db, score, np.ones(len(db))]
        fe = fe / (np.linalg.norm(fe, axis=1, keepdims=True) + 1e-12)
        feats[f] = fe if cfg.feature_dim > 0 else np.zeros((len(db), 0))
    return Scene(gt, dets, feats, warps)


def simulate_scene(cfg: SceneConfig) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Backward-compatible helper returning only ``(gt, detections)``."""
    s = simulate(cfg)
    return s.gt, s.dets
