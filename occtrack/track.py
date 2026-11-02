from __future__ import annotations

from collections import deque
from enum import IntEnum
from typing import Deque, List, Optional, Tuple

import numpy as np

from .boxes import xyah_to_xyxy, xyxy_to_xyah
from .kalman import KalmanFilter, warp_state


class TrackState(IntEnum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


class Track:
    """A single target.

    Besides the Kalman state the track keeps

    * its recent **observations** (for observation-centric momentum and re-updates),
    * the filter state right after the last observation (for ORU),
    * an exponential-moving-average appearance embedding, and
    * an **occlusion-aware prototype memory**: a small bank of diverse embeddings that is only
      written when the target is mostly visible, used for long-term re-identification.
    """

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
        memory_size: int = 10,
        memory_occlusion_thresh: float = 0.3,
    ) -> None:
        self.track_id = track_id
        self.kf = kf
        z = xyxy_to_xyah(box_xyxy)
        self.mean, self.cov = kf.initiate(z)
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
        self.prototypes: Deque[np.ndarray] = deque(maxlen=memory_size)
        self.memory_occlusion_thresh = memory_occlusion_thresh
        self.observations: Deque[Tuple[int, np.ndarray]] = deque(maxlen=30)
        self.observations.append((frame_id, z.copy()))
        self._saved = (self.mean.copy(), self.cov.copy())
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

    @property
    def last_observation(self) -> np.ndarray:
        return self.observations[-1][1]

    def observed_direction(self, delta: int = 3) -> Optional[np.ndarray]:
        """Unit direction of motion estimated from observations ~``delta`` frames apart."""
        if len(self.observations) < 2:
            return None
        last_f, last_z = self.observations[-1]
        prev = self.observations[0]
        for f, z in reversed(list(self.observations)[:-1]):
            prev = (f, z)
            if last_f - f >= delta:
                break
        d = last_z[:2] - prev[1][:2]
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else None

    def position_std(self) -> float:
        """Std. deviation of the predicted centre (grows while the track is lost)."""
        return float(np.sqrt(self.cov[0, 0] + self.cov[1, 1]))

    def appearance_distance(self, feats: np.ndarray) -> np.ndarray:
        """Min cosine distance (in [0, 1]) between each feature and the prototype memory."""
        bank: List[np.ndarray] = list(self.prototypes) or ([self.feature] if self.feature is not None else [])
        if not bank:
            return np.ones(len(feats))
        b = np.stack(bank)
        f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        return np.clip(1.0 - (f @ b.T).max(axis=1), 0.0, 2.0) / 2.0

    # ---------------------------------------------------------------- updates
    def _update_feature(self, feat: np.ndarray) -> None:
        feat = np.asarray(feat, dtype=np.float64)
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        visible = self.occlusion < self.memory_occlusion_thresh
        if self.feature is None:
            self.feature = feat
        elif visible:
            # Skip appearance updates while occluded: the crop mostly shows the occluder.
            m = self.feature_momentum
            self.feature = m * self.feature + (1 - m) * feat
            self.feature /= np.linalg.norm(self.feature) + 1e-12
        if visible and (not self.prototypes or max(float(p @ feat) for p in self.prototypes) < 0.95):
            self.prototypes.append(feat)

    def apply_warp(self, warp: np.ndarray) -> None:
        """Camera-motion compensation of the state, the saved state and past observations."""
        self.mean, self.cov = warp_state(self.mean, self.cov, warp)
        self._saved = warp_state(*self._saved, warp)
        a, t = warp[:, :2], warp[:, 2]
        s = float(np.sqrt(abs(np.linalg.det(a))))
        for i, (f, z) in enumerate(self.observations):
            z = z.copy()
            z[:2] = a @ z[:2] + t
            z[3] *= s
            self.observations[i] = (f, z)

    def predict(self) -> None:
        if self.state == TrackState.LOST:
            # Freeze the height velocity for lost tracks so boxes don't collapse/explode.
            self.mean[7] = 0.0
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.time_since_update += 1

    def _re_update(self, z_new: np.ndarray, frame_id: int, noise: float) -> None:
        """Observation-centric re-update (OC-SORT): after a gap, rebuild the filter state along a
        virtual trajectory linearly interpolated between the last and the new observation, so the
        error accumulated while extrapolating without observations is removed."""
        last_f, z_last = self.observations[-1]
        gap = frame_id - last_f
        mean, cov = self._saved
        for i in range(1, gap):
            mean, cov = self.kf.predict(mean, cov)
            mean, cov = self.kf.update(mean, cov, z_last + (z_new - z_last) * i / gap, noise)
        self.mean, self.cov = self.kf.predict(mean, cov)

    def update(
        self,
        box_xyxy: np.ndarray,
        score: float,
        frame_id: int,
        feature: Optional[np.ndarray] = None,
        occlusion: float = 0.0,
        adaptive_noise: bool = True,
        re_update: bool = True,
    ) -> None:
        self.occlusion = occlusion
        noise = 1.0
        if adaptive_noise:
            # Confidence/occlusion-adaptive measurement noise: trust weak or occluded boxes less.
            noise = (1.0 + 2.0 * occlusion) / max(score, 0.1)
        z = xyxy_to_xyah(box_xyxy)
        if re_update and frame_id - self.last_frame > 1:
            self._re_update(z, frame_id, noise)
        self.mean, self.cov = self.kf.update(self.mean, self.cov, z, noise)
        self._saved = (self.mean.copy(), self.cov.copy())
        self.observations.append((frame_id, z.copy()))
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
