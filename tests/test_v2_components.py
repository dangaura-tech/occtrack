import numpy as np
import pytest

from occtrack import OccTracker, TrackerConfig
from occtrack.boxes import xyxy_to_xyah
from occtrack.eval import evaluate_sequence, hota
from occtrack.kalman import KalmanFilter, warp_state
from occtrack.matching import direction_cost
from occtrack.postprocess import interpolate_tracks
from occtrack.simulation import SceneConfig, simulate
from occtrack.track import Track


def box_at(x, y=100.0, w=40.0, h=100.0):
    return np.array([x, y, x + w, y + h])


# ------------------------------------------------------------------------------- CMC
def test_warp_state_translation_and_scale():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 50.0, 0.5, 80.0]))
    mean[4:6] = [2.0, 0.0]
    warp = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, -5.0]])
    m2, c2 = warp_state(mean, cov, warp)
    np.testing.assert_allclose(m2[:4], [210.0, 95.0, 0.5, 160.0])
    np.testing.assert_allclose(m2[4:6], [4.0, 0.0])
    assert c2[0, 0] == pytest.approx(4 * cov[0, 0])


def test_cmc_keeps_identity_under_camera_shake():
    """Static target, camera shakes by +-35 px per frame: a constant-velocity model cannot
    predict this motion, so without CMC the target is lost in most frames."""
    shakes = np.random.default_rng(0).choice([-35.0, 35.0], size=40)

    def run(cmc: bool):
        tracker = OccTracker(TrackerConfig(cmc=cmc))
        ids, x = [], 800.0
        for t in range(40):
            warp = None
            if t > 0:
                x += shakes[t]
                warp = np.array([[1.0, 0, shakes[t]], [0, 1.0, 0]])
            ids += [tr.track_id for tr in tracker.update(box_at(x)[None], np.array([0.9]), warp=warp)]
        return ids

    with_cmc, without = run(cmc=True), run(cmc=False)
    assert with_cmc == [1] * 40  # tracked in every frame with a single identity
    assert len(without) < 20


def test_sparse_flow_cmc_recovers_translation():
    cv2 = pytest.importorskip("cv2")
    from occtrack.cmc import SparseFlowCMC

    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((300, 400)) * 255).astype(np.uint8), (5, 5), 0)
    shifted = np.roll(base, shift=(4, -6), axis=(0, 1))
    cmc = SparseFlowCMC(downscale=1)
    cmc.apply(base)
    warp = cmc.apply(shifted)
    np.testing.assert_allclose(warp[:, 2], [-6, 4], atol=0.5)
    np.testing.assert_allclose(warp[:, :2], np.eye(2), atol=0.02)


# ------------------------------------------------------------------------- ORU / OCM
def test_observation_centric_re_update_reduces_drift():
    kf = KalmanFilter()

    def final_error(re_update: bool) -> float:
        tr = Track(1, box_at(0), 0.9, 1, kf)
        for f in range(2, 11):  # moving right at 10 px/frame
            tr.predict()
            tr.update(box_at(10 * (f - 1)), 0.9, f, re_update=re_update)
        for _ in range(10):  # unobserved: pretend the filter drifts (velocity flipped)
            tr.predict()
        tr.mean[4] = -10.0
        tr.predict()
        tr.update(box_at(10 * 20), 0.9, 21, re_update=re_update)
        tr.predict()
        return abs(tr.mean[4] - 10.0)  # velocity error after re-association

    assert final_error(True) < final_error(False)


def test_direction_cost_penalises_reversal():
    kf = KalmanFilter()
    tr = Track(1, box_at(0), 0.9, 1, kf)
    for f in range(2, 6):
        tr.predict()
        tr.update(box_at(10 * (f - 1)), 0.9, f)
    dets = np.stack([box_at(60), box_at(20)])  # ahead vs behind the last observation (x=40)
    cost = direction_cost([tr], dets, np.array([1.0, 1.0]))
    assert cost[0, 0] < 0.05 and cost[0, 1] > 0.9


def test_observation_history_is_warped():
    tr = Track(1, box_at(0), 0.9, 1, KalmanFilter())
    tr.apply_warp(np.array([[1.0, 0, 5.0], [0, 1.0, 0]]))
    assert tr.last_observation[0] == pytest.approx(xyxy_to_xyah(box_at(5))[0])


# --------------------------------------------------------------------------- Re-ID
def test_prototype_memory_ignores_occluded_appearance():
    tr = Track(1, box_at(0), 0.9, 1, KalmanFilter(), feature=np.array([1.0, 0.0]))
    tr.update(box_at(1), 0.9, 2, feature=np.array([0.0, 1.0]), occlusion=0.8)
    assert len(tr.prototypes) == 1
    np.testing.assert_allclose(tr.feature, [1.0, 0.0])
    assert tr.appearance_distance(np.array([[1.0, 0.0], [-1.0, 0.0]])).tolist() == pytest.approx([0.0, 1.0], abs=1e-9)


def test_reid_stage_recovers_long_occlusion():
    """Target disappears for 25 frames and reappears far from the extrapolated box."""
    feat = np.array([[1.0, 0.0, 0.0]])

    def run(reid: bool):
        tracker = OccTracker(TrackerConfig(reid_stage=reid, track_buffer=60, reid_spatial_sigma=50.0))
        ids = []
        for t in range(60):
            if 20 <= t < 45:
                tracker.update(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 3)))
                continue
            x = 100 + 4 * t if t < 20 else 400  # reappears somewhere else
            ids += [tr.track_id for tr in tracker.update(box_at(x)[None], np.array([0.9]), feat)]
        return set(ids)

    assert run(True) == {1}
    assert run(False) == {1, 2}


# ------------------------------------------------------------------------ HOTA / post
def seq(rows):
    out = {}
    for f, tid, x in rows:
        out.setdefault(f, []).append([tid, x, 0, x + 10, 10])
    return {f: np.array(v, float) for f, v in out.items()}


def test_hota_perfect_and_split_identity():
    gt = seq([(f, 1, f) for f in range(1, 21)])
    assert hota(gt, gt)["HOTA"] == pytest.approx(1.0)
    pred = seq([(f, 7 if f <= 10 else 8, f) for f in range(1, 21)])
    r = hota(gt, pred)
    assert r["DetA"] == pytest.approx(1.0)
    assert r["AssA"] == pytest.approx(0.5)
    assert r["HOTA"] == pytest.approx(np.sqrt(0.5))


def test_hota_detection_errors():
    gt = seq([(f, 1, 0) for f in range(1, 11)])
    pred = seq([(f, 1, 0) for f in range(1, 6)])  # second half missed
    r = hota(gt, pred)
    assert r["DetA"] == pytest.approx(0.5)
    assert r["AssA"] == pytest.approx(0.5)


def test_interpolation_fills_gaps():
    res = {1: np.array([[3, 0, 0, 10, 10, 0.9]]), 5: np.array([[3, 40, 0, 50, 10, 0.8]])}
    out = interpolate_tracks(res, max_gap=10)
    assert sorted(out) == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(out[3][0, 1:5], [20, 0, 30, 10])
    assert sorted(interpolate_tracks(res, max_gap=2)) == [1, 5]


def test_simulator_camera_motion_and_features():
    s = simulate(SceneConfig(num_frames=20, num_targets=5, camera_pan=5.0, feature_dim=8, seed=1))
    assert s.warps[2][0, 2] == pytest.approx(-5.0, abs=1e-9)
    assert all(len(s.features[f]) == len(s.dets[f]) for f in s.dets)
    assert s.features[3].shape[1] == 8


def test_full_v2_on_panning_scene_with_reid():
    s = simulate(SceneConfig(num_frames=150, num_targets=10, camera_pan=6.0, camera_jitter=1.0, feature_dim=32, seed=4))
    tracker = OccTracker()
    pred = {}
    for f in sorted(s.dets):
        d = s.dets[f]
        tr = tracker.update(d[:, 1:5], d[:, 5], s.features[f], warp=s.warps[f])
        pred[f] = np.array([[t.track_id, *t.xyxy] for t in tr]).reshape(-1, 5)
    res = evaluate_sequence(s.gt, pred)
    assert res["HOTA"] > 0.6 and res["MOTA"] > 0.6
