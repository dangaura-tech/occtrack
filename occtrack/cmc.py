"""Camera Motion Compensation (CMC).

Moving or zooming cameras make every object move in image space even when it stands still,
which breaks the constant-velocity assumption of the Kalman filter. CMC estimates the global
affine image motion between consecutive frames and warps every track state into the new
frame before association (cf. BoT-SORT).

``SparseFlowCMC`` tracks Shi-Tomasi corners with pyramidal Lucas-Kanade optical flow and fits a
partial (similarity) affine model with RANSAC. Foreground boxes can be masked out so that
moving people do not bias the estimate. OpenCV is an optional dependency; precomputed warps
can also be passed directly to :meth:`OccTracker.update`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class SparseFlowCMC:
    def __init__(self, max_corners: int = 1000, downscale: int = 2, ransac_thresh: float = 3.0) -> None:
        try:
            import cv2
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("SparseFlowCMC requires `pip install opencv-python`") from e
        self.cv2 = cv2
        self.max_corners = max_corners
        self.downscale = max(1, int(downscale))
        self.ransac_thresh = ransac_thresh
        self._prev: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None
        self._prev_pts = None

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        cv2 = self.cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.downscale > 1:
            h, w = gray.shape
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale), interpolation=cv2.INTER_AREA)
        return gray

    def _mask(self, shape, boxes: Optional[np.ndarray]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), np.uint8)
        mask[int(0.02 * h) : int(0.98 * h), int(0.02 * w) : int(0.98 * w)] = 255
        if boxes is not None:
            for x1, y1, x2, y2 in (np.asarray(boxes) / self.downscale).astype(int):
                mask[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = 0
        return mask

    def apply(self, frame: np.ndarray, boxes: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the 2x3 affine warp mapping the previous frame onto ``frame`` (full resolution)."""
        cv2 = self.cv2
        gray = self._prepare(frame)
        pts = cv2.goodFeaturesToTrack(
            gray, self.max_corners, qualityLevel=0.01, minDistance=1, blockSize=3, mask=self._mask(gray.shape, boxes)
        )
        warp = IDENTITY.copy()
        if self._prev is not None and self._prev_pts is not None and len(self._prev_pts) >= 4:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self._prev, gray, self._prev_pts, None)
            ok = status.ravel() == 1
            if ok.sum() >= 4:
                est, _ = cv2.estimateAffinePartial2D(
                    self._prev_pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh
                )
                if est is not None:
                    warp = est
                    warp[:, 2] *= self.downscale
        self._prev, self._prev_pts = gray, pts
        return warp
