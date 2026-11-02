"""OccTracker: occlusion-aware multi-object tracking by detection.

Per frame:

0. **Camera motion compensation** - warp every track by the estimated global affine motion.
1. **Predict** every active/lost track with the Kalman filter.
2. **Stage 1** - *high-confidence* detections vs. confirmed + lost tracks. The cost fuses
   IoU distance, observation-centric momentum (direction consistency) and, when embeddings
   are available, appearance; it is gated by the Mahalanobis distance.
3. **Stage 2** - *low-confidence* detections (often partially occluded people) vs. the
   remaining confirmed tracks, by IoU only.
4. **Stage 3** - leftover high-confidence detections vs. tentative tracks.
5. **Stage 4 (long-term Re-ID)** - leftover high-confidence detections vs. lost tracks via
   the occlusion-aware prototype memory, inside a spatial window that grows with the
   track's positional uncertainty.
6. Re-associated tracks get an **observation-centric re-update** (ORU) that removes the
   drift accumulated while they were invisible.
7. Lost tracks are kept for an **occlusion-adaptive** time; unmatched confident detections
   start new tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .boxes import occlusion_ratio
from .kalman import KalmanFilter
from .matching import (
    INF_COST,
    direction_cost,
    embedding_distance,
    fuse_score,
    gate_by_motion,
    iou_distance,
    linear_assignment,
    reid_cost,
)
from .track import Track, TrackState


@dataclass
class TrackerConfig:
    high_thresh: float = 0.6
    low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    match_thresh: float = 0.8
    second_match_thresh: float = 0.5
    tentative_match_thresh: float = 0.7
    track_buffer: int = 30
    occlusion_buffer_gain: float = 1.0
    occlusion_thresh: float = 0.3
    min_hits: int = 2
    appearance_weight: float = 0.3
    adaptive_noise: bool = True
    motion_gating: bool = True
    min_box_area: float = 10.0
    frame_rate: int = 30
    # v2 components
    cmc: bool = True  # apply camera-motion warps passed to update()
    re_update: bool = True  # observation-centric re-update (ORU)
    momentum_weight: float = 0.2  # observation-centric momentum (OCM); 0 disables
    momentum_delta: int = 3
    reid_stage: bool = True  # stage 4: long-term re-identification (needs embeddings)
    reid_thresh: float = 0.25
    reid_spatial_sigma: float = 3.0
    memory_size: int = 10


class OccTracker:
    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.cfg = config or TrackerConfig()
        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        self.frame_id = 0
        self._next_id = 1
        self.max_time_lost = int(round(self.cfg.frame_rate / 30.0 * self.cfg.track_buffer))

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_id = 0
        self._next_id = 1

    # --------------------------------------------------------------- helpers
    def _new_track(self, box: np.ndarray, score: float, feat: Optional[np.ndarray]) -> None:
        self.tracks.append(
            Track(
                self._next_id,
                box,
                score,
                self.frame_id,
                self.kf,
                feat,
                min_hits=self.cfg.min_hits,
                memory_size=self.cfg.memory_size,
                memory_occlusion_thresh=self.cfg.occlusion_thresh,
            )
        )
        self._next_id += 1

    def _cost(
        self, tracks: List[Track], boxes: np.ndarray, scores: np.ndarray, feats: Optional[np.ndarray]
    ) -> np.ndarray:
        cost = fuse_score(iou_distance(tracks, boxes), scores)
        overlap = cost < 1.0
        if self.cfg.momentum_weight > 0:
            cost = cost + self.cfg.momentum_weight * direction_cost(tracks, boxes, scores, self.cfg.momentum_delta)
        emb = embedding_distance(tracks, feats)
        if emb is not None:
            w = self.cfg.appearance_weight
            cost = np.where(overlap, (1 - w) * cost + w * emb, cost)
        cost = np.where(overlap, cost, np.maximum(cost, 1.0))  # never match boxes without overlap here
        if self.cfg.motion_gating:
            cost = gate_by_motion(cost, tracks, boxes)
        return cost

    def _max_lost(self, track: Track) -> int:
        bonus = self.cfg.occlusion_buffer_gain if track.occlusion >= self.cfg.occlusion_thresh else 0.0
        return int(self.max_time_lost * (1.0 + bonus))

    # ------------------------------------------------------------------ main
    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        features: Optional[np.ndarray] = None,
        warp: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Advance the tracker by one frame.

        Args:
            boxes: (N, 4) detections in ``x1, y1, x2, y2`` pixel coordinates.
            scores: (N,) detector confidences.
            features: optional (N, D) appearance embeddings (e.g. from a Re-ID network).
            warp: optional 2x3 affine camera motion from the previous frame to this one
                (e.g. from :class:`occtrack.cmc.SparseFlowCMC`).

        Returns:
            The confirmed tracks updated in this frame.
        """
        self.frame_id += 1
        cfg = self.cfg
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if features is not None:
            features = np.asarray(features, dtype=np.float64)
            dim = features.shape[-1] if features.ndim == 2 else (features.size // max(len(boxes), 1))
            features = features.reshape(len(boxes), dim)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = (scores >= cfg.low_thresh) & (area >= cfg.min_box_area)
        boxes, scores = boxes[keep], scores[keep]
        features = features[keep] if features is not None else None
        occl = occlusion_ratio(boxes)

        high = scores >= cfg.high_thresh
        hi_idx = np.flatnonzero(high)
        lo_idx = np.flatnonzero(~high)

        if cfg.cmc and warp is not None:
            for t in self.tracks:
                t.apply_warp(np.asarray(warp, dtype=np.float64))
        for t in self.tracks:
            t.predict()

        pool = [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t for t in self.tracks if t.state == TrackState.TENTATIVE]

        def feats_of(idx: np.ndarray) -> Optional[np.ndarray]:
            return features[idx] if features is not None else None

        def apply(track: Track, det: int) -> None:
            feat = features[det] if features is not None else None
            track.update(boxes[det], scores[det], self.frame_id, feat, occl[det], cfg.adaptive_noise, cfg.re_update)

        # Stage 1: high-score detections vs confirmed + lost tracks.
        cost = self._cost(pool, boxes[hi_idx], scores[hi_idx], feats_of(hi_idx))
        m1, u_pool, u_hi = linear_assignment(cost, cfg.match_thresh)
        for ti, di in m1:
            apply(pool[ti], hi_idx[di])

        # Stage 2: low-score detections vs remaining recently-seen tracks (IoU only).
        remaining = [pool[i] for i in u_pool if pool[i].state == TrackState.CONFIRMED]
        cost = iou_distance(remaining, boxes[lo_idx])
        m2, u_rem, _ = linear_assignment(cost, cfg.second_match_thresh)
        for ti, di in m2:
            apply(remaining[ti], lo_idx[di])
        for i in u_rem:
            remaining[i].mark_lost()

        # Stage 3: leftover high-score detections vs tentative tracks.
        hi_left = hi_idx[u_hi]
        cost = fuse_score(iou_distance(tentative, boxes[hi_left]), scores[hi_left])
        m3, u_tent, u_new = linear_assignment(cost, cfg.tentative_match_thresh)
        for ti, di in m3:
            apply(tentative[ti], hi_left[di])
        for i in u_tent:
            tentative[i].mark_removed()
        hi_left = hi_left[u_new]

        # Stage 4: long-term re-identification of lost tracks through the prototype memory.
        if cfg.reid_stage and features is not None and len(hi_left):
            lost = [t for t in self.tracks if t.state == TrackState.LOST and t.time_since_update > 1]
            cost = reid_cost(lost, boxes[hi_left], features[hi_left], cfg.reid_spatial_sigma)
            m4, _, u_new4 = linear_assignment(cost, cfg.reid_thresh)
            for ti, di in m4:
                apply(lost[ti], hi_left[di])
            hi_left = hi_left[u_new4]

        # Spawn new tracks.
        for di in hi_left:
            if scores[di] >= cfg.new_track_thresh:
                self._new_track(boxes[di], scores[di], features[di] if features is not None else None)

        # Retire lost tracks whose (occlusion-adaptive) buffer expired.
        for t in self.tracks:
            if t.state == TrackState.LOST and self.frame_id - t.last_frame > self._max_lost(t):
                t.mark_removed()
        self.tracks = [t for t in self.tracks if t.state != TrackState.REMOVED]

        return [t for t in self.tracks if t.is_confirmed and t.last_frame == self.frame_id]


__all__ = ["OccTracker", "TrackerConfig", "INF_COST"]
