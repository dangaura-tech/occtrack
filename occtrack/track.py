from __future__ import annotations

from collections import deque
from enum import IntEnum
from typing import Deque, Optional

import numpy as np

from .boxes import xyah_to_xyxy, xyxy_to_xyah
from .kalman import KalmanFilter


class TrackState(IntEnum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


class Track:
    """A single target: Kalman state, lifecycle bookkeeping and an appearance memory."""

    def __init__(
        self,
        track_id: int,
        box_xyxy: np.ndarray,
        score: float,
        frame_id: int,
        kf: KalmanFilter,
        feature: Optional[np.ndarray] = None,
        feature_momentum: float = 0.9,
        min_hits: int = 2,
    ) -> None:
        self.track_id = track_id
        self.kf = kf
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(box_xyxy))
        self.score = float(score)
        self.state = TrackState.CONFIRMED if frame_id == 1 else TrackState.TENTATIVE
        self.hits = 1
        self.min_hits = min_hits
        self.start_frame = frame_id
        self.last_frame = frame_id
        self.time_since_update = 0
        self.occlusion = 0.0
        self.feature_momentum = feature_momentum
        self.feature: Optional[np.ndarray] = None
        self.history: Deque[np.ndarray] = deque(maxlen=30)
        if feature is not None:
            self._update_feature(feature)

    # ------------------------------------------------------------------ props
    @property
    def xyxy(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def velocity(self) -> np.ndarray:
        return self.mean[4:6].copy()

    # ---------------------------------------------------------------- updates
    def _update_feature(self, feat: np.ndarray) -> None:
        feat = np.asarray(feat, dtype=np.float64)
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        if self.feature is None:
            self.feature = feat
        else:
            # Skip the appearance update while heavily occluded: the crop mostly shows the occluder.
            m = self.feature_momentum if self.occlusion < 0.5 else 1.0
            self.feature = m * self.feature + (1 - m) * feat
            self.feature /= np.linalg.norm(self.feature) + 1e-12

    def predict(self) -> None:
        if self.state != TrackState.CONFIRMED and self.state != TrackState.TENTATIVE:
            # Freeze the height velocity for lost tracks so boxes don't collapse/explode.
            self.mean[7] = 0.0
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.time_since_update += 1

    def update(
        self,
        box_xyxy: np.ndarray,
        score: float,
        frame_id: int,
        feature: Optional[np.ndarray] = None,
        occlusion: float = 0.0,
        adaptive_noise: bool = True,
    ) -> None:
        self.occlusion = occlusion
        noise = 1.0
        if adaptive_noise:
            # Confidence/occlusion-adaptive measurement noise: trust weak or occluded boxes less.
            noise = (1.0 + 2.0 * occlusion) / max(score, 0.1)
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(box_xyxy), noise)
        self.history.append(self.xyxy)
        self.score = float(score)
        self.hits += 1
        self.last_frame = frame_id
        self.time_since_update = 0
        if feature is not None:
            self._update_feature(feature)
        if self.state == TrackState.TENTATIVE and self.hits >= self.min_hits:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED

    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, state={self.state.name}, box={np.round(self.xyxy, 1).tolist()})"
