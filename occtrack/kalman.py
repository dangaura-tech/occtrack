"""Constant-velocity Kalman filter in (cx, cy, a, h) space.

State: ``[cx, cy, a, h, vcx, vcy, va, vh]``. Process and measurement noise are scaled by
the box height (as in SORT/DeepSORT) and, optionally, the measurement noise is inflated
for low-confidence or heavily occluded detections (*confidence-adaptive* update).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg

# 0.95 quantile of the chi-square distribution with N degrees of freedom.
CHI2INV95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}


class KalmanFilter:
    ndim = 4

    def __init__(self, std_position: float = 1.0 / 20, std_velocity: float = 1.0 / 160) -> None:
        self._F = np.eye(2 * self.ndim)
        for i in range(self.ndim):
            self._F[i, self.ndim + i] = 1.0
        self._H = np.eye(self.ndim, 2 * self.ndim)
        self.std_pos = std_position
        self.std_vel = std_velocity

    def initiate(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.r_[z, np.zeros(self.ndim)]
        h = z[3]
        std = [
            2 * self.std_pos * h,
            2 * self.std_pos * h,
            1e-2,
            2 * self.std_pos * h,
            10 * self.std_vel * h,
            10 * self.std_vel * h,
            1e-5,
            10 * self.std_vel * h,
        ]
        return mean, np.diag(np.square(std))

    def _process_noise(self, h: float) -> np.ndarray:
        std = [
            self.std_pos * h,
            self.std_pos * h,
            1e-2,
            self.std_pos * h,
            self.std_vel * h,
            self.std_vel * h,
            1e-5,
            self.std_vel * h,
        ]
        return np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + self._process_noise(mean[3])
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray, noise_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = np.array([self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]) * noise_scale
        return self._H @ mean, self._H @ cov @ self._H.T + np.diag(np.square(std))

    def update(
        self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray, noise_scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov, noise_scale)
        chol = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        gain = scipy.linalg.cho_solve(chol, (cov @ self._H.T).T, check_finite=False).T
        innovation = z - proj_mean
        mean = mean + gain @ innovation
        cov = cov - gain @ proj_cov @ gain.T
        return mean, cov

    def gating_distance(
        self, mean: np.ndarray, cov: np.ndarray, zs: np.ndarray, only_position: bool = False
    ) -> np.ndarray:
        """Squared Mahalanobis distance between the state distribution and measurements."""
        proj_mean, proj_cov = self.project(mean, cov)
        zs = np.asarray(zs, dtype=np.float64).reshape(-1, 4)
        if only_position:
            proj_mean, proj_cov, zs = proj_mean[:2], proj_cov[:2, :2], zs[:, :2]
        chol = np.linalg.cholesky(proj_cov)
        d = zs - proj_mean
        sol = scipy.linalg.solve_triangular(chol, d.T, lower=True, check_finite=False)
        return np.sum(sol * sol, axis=0)
