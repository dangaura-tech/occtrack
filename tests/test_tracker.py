import numpy as np

from occtrack import OccTracker, TrackerConfig, TrackState
from occtrack.eval import evaluate_sequence
from occtrack.simulation import SceneConfig, simulate_scene


def moving_box(t, x0=100.0, vx=5.0):
    x = x0 + vx * t
    return [x, 100.0, x + 40.0, 200.0]


def test_single_target_keeps_id():
    tracker = OccTracker()
    ids = set()
    for t in range(50):
        out = tracker.update(np.array([moving_box(t)]), np.array([0.9]))
        ids.update(tr.track_id for tr in out)
    assert ids == {1}


def test_two_crossing_targets_keep_identity():
    tracker = OccTracker()
    history = {}
    for t in range(60):
        boxes = np.array([moving_box(t, 100, 6), moving_box(t, 460, -6)])
        boxes[1, [1, 3]] += 150  # different rows -> they pass each other without merging
        for tr in tracker.update(boxes, np.array([0.9, 0.9])):
            history.setdefault(tr.track_id, []).append(tr.xyxy[0])
    assert set(history) == {1, 2}
    assert history[1][-1] > history[1][0]  # moving right
    assert history[2][-1] < history[2][0]  # moving left


def test_track_survives_short_occlusion_gap():
    tracker = OccTracker(TrackerConfig(track_buffer=10))
    seen = []
    for t in range(40):
        if 15 <= t < 22:
            out = tracker.update(np.zeros((0, 4)), np.zeros(0))
        else:
            out = tracker.update(np.array([moving_box(t)]), np.array([0.9]))
        seen += [tr.track_id for tr in out]
    assert set(seen) == {1}


def test_lost_track_is_removed_after_buffer():
    tracker = OccTracker(TrackerConfig(track_buffer=5, occlusion_buffer_gain=0.0))
    tracker.update(np.array([moving_box(0)]), np.array([0.9]))
    for _ in range(10):
        tracker.update(np.zeros((0, 4)), np.zeros(0))
    assert tracker.tracks == []


def test_low_score_detection_extends_track():
    tracker = OccTracker()
    for t in range(10):
        tracker.update(np.array([moving_box(t)]), np.array([0.9]))
    out = tracker.update(np.array([moving_box(10)]), np.array([0.3]))  # occluded -> weak score
    assert [tr.track_id for tr in out] == [1]
    assert tracker.tracks[0].state == TrackState.CONFIRMED


def test_low_score_detection_does_not_spawn_tracks():
    tracker = OccTracker()
    tracker.update(np.array([[0, 0, 50, 100]]), np.array([0.4]))
    assert tracker.tracks == []


def test_appearance_features_are_accepted():
    tracker = OccTracker()
    rng = np.random.default_rng(0)
    for t in range(5):
        out = tracker.update(np.array([moving_box(t)]), np.array([0.9]), rng.normal(size=(1, 16)))
    assert out[0].feature.shape == (16,)


def test_synthetic_scene_quality():
    gt, dets = simulate_scene(SceneConfig(num_frames=150, num_targets=8, seed=3))
    tracker = OccTracker()
    pred = {}
    for f in sorted(dets):
        d = dets[f]
        tr = tracker.update(d[:, 1:5], d[:, 5])
        pred[f] = np.array([[t.track_id, *t.xyxy] for t in tr]).reshape(-1, 5)
    res = evaluate_sequence(gt, pred)
    assert res["MOTA"] > 0.6
    assert res["IDF1"] > 0.6
