import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from occtrack.io import write_mot  # noqa: E402
from occtrack.visualize import TrailBuffer, draw_ground_truth, draw_tracks, id_color, render_sequence  # noqa: E402


def test_id_colors_are_deterministic_and_distinct():
    assert id_color(7) == id_color(7)
    colors = {id_color(i) for i in range(1, 30)}
    assert len(colors) == 29


def test_trail_buffer_keeps_last_points():
    tb = TrailBuffer(length=3)
    for x in range(5):
        pts = tb.update(1, np.array([x, 0, x + 2, 10.0]))
    assert pts == [(3, 10), (4, 10), (5, 10)]
    assert len(tb) == 1


def test_draw_tracks_marks_border_not_interior():
    frame = np.zeros((100, 100, 3), np.uint8)
    draw_tracks(frame, np.array([[3, 20, 30, 70, 80]]), show_ids=False)
    assert tuple(frame[50, 20]) == id_color(3)  # left edge
    assert frame[55, 45].sum() == 0  # interior untouched


def test_ground_truth_is_dashed():
    frame = np.zeros((60, 120, 3), np.uint8)
    draw_ground_truth(frame, np.array([[1, 10, 10, 110, 50]]), dash=5)
    top = frame[10, 10:110].sum(axis=1) > 0
    assert 0.3 < top.mean() < 0.8  # roughly half of the top edge is drawn


@pytest.mark.parametrize("ext", [".avi", ""])
def test_render_sequence(tmp_path, ext):
    img_dir = tmp_path / "img1"
    img_dir.mkdir()
    for f in range(1, 6):
        cv2.imwrite(str(img_dir / f"{f:06d}.jpg"), np.full((64, 96, 3), 40, np.uint8))
    res_file = tmp_path / "res.txt"
    write_mot(str(res_file), [(f, 2, np.array([5 + 3 * f, 10, 30 + 3 * f, 50]), 0.9) for f in range(1, 6)])
    from occtrack.io import load_mot

    out = tmp_path / f"vis{ext}"
    n = render_sequence(str(img_dir), load_mot(str(res_file)), str(out), fps=10)
    assert n == 5
    if ext:
        cap = cv2.VideoCapture(str(out))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    else:
        assert len(list(out.glob("*.png"))) == 5
