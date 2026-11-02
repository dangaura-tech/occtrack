import numpy as np
import pytest

from occtrack.boxes import iou_matrix, occlusion_ratio, tlwh_to_xyxy, xyah_to_xyxy, xyxy_to_tlwh, xyxy_to_xyah
from occtrack.kalman import KalmanFilter


def test_box_roundtrips():
    b = np.array([[10.0, 20.0, 50.0, 120.0]])
    np.testing.assert_allclose(xyah_to_xyxy(xyxy_to_xyah(b)), b)
    np.testing.assert_allclose(tlwh_to_xyxy(xyxy_to_tlwh(b)), b)


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_occlusion_ratio_front_box_is_not_occluded():
    back = [0, 0, 10, 20]  # bottom at y=20 -> farther away
    front = [5, 5, 15, 30]  # bottom at y=30 -> closer, occludes the back box
    r = occlusion_ratio(np.array([back, front]))
    assert r[0] == pytest.approx(5 * 15 / 200)
    assert r[1] == 0.0


def test_kalman_tracks_constant_velocity():
    kf = KalmanFilter()
    z = np.array([100.0, 100.0, 0.5, 100.0])
    mean, cov = kf.initiate(z)
    for t in range(1, 30):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, z + np.array([5.0 * t, 0, 0, 0]))
    assert mean[4] == pytest.approx(5.0, abs=0.3)
    pred, _ = kf.predict(mean, cov)
    assert pred[0] == pytest.approx(100 + 5 * 30, abs=2.0)


def test_noise_scale_reduces_update_strength():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.0, 0.0, 0.5, 100.0]))
    mean, cov = kf.predict(mean, cov)
    z = np.array([20.0, 0.0, 0.5, 100.0])
    trusting, _ = kf.update(mean, cov, z, noise_scale=1.0)
    sceptical, _ = kf.update(mean, cov, z, noise_scale=5.0)
    assert trusting[0] > sceptical[0] > 0


def test_gating_distance_grows_with_offset():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.0, 0.0, 0.5, 100.0]))
    d = kf.gating_distance(mean, cov, np.array([[0, 0, 0.5, 100], [50, 0, 0.5, 100]]))
    assert d[0] < d[1]
