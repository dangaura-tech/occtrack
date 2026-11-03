"""Rendering of tracking results onto video frames (OpenCV).

Used by ``tools/visualize.py`` to turn a MOTChallenge result file into an annotated video:
ID-coloured boxes, ID labels, motion trails and an optional ground-truth overlay.
"""

from __future__ import annotations

import colorsys
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

Color = Tuple[int, int, int]


def id_color(track_id: int) -> Color:
    """Deterministic, well-separated BGR colour per ID (golden-ratio hue stepping)."""
    h = (int(track_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


class TrailBuffer:
    """Keeps the last ``length`` bottom-centre points of every track."""

    def __init__(self, length: int = 30) -> None:
        self.length = length
        self._trails: Dict[int, Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=self.length))

    def update(self, track_id: int, box_xyxy: np.ndarray) -> List[Tuple[int, int]]:
        x1, _, x2, y2 = box_xyxy
        self._trails[int(track_id)].append((int(round((x1 + x2) / 2)), int(round(y2))))
        return list(self._trails[int(track_id)])

    def __len__(self) -> int:
        return len(self._trails)


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("visualisation requires `pip install opencv-python-headless`") from e
    return cv2


def draw_tracks(
    frame: np.ndarray,
    rows: np.ndarray,
    trails: Optional[TrailBuffer] = None,
    thickness: int = 2,
    show_ids: bool = True,
) -> np.ndarray:
    """Draw tracks in place. ``rows``: (K, >=5) array of ``id, x1, y1, x2, y2``."""
    cv2 = _cv2()
    for r in np.asarray(rows).reshape(-1, max(np.asarray(rows).shape[-1], 5)):
        tid = int(r[0])
        box = r[1:5]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        c = id_color(tid)
        if trails is not None:
            pts = trails.update(tid, box)
            if len(pts) > 1:
                cv2.polylines(frame, [np.array(pts, np.int32)], False, c, max(1, thickness - 1), cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, thickness)
        if show_ids:
            label = str(tid)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), c, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_ground_truth(frame: np.ndarray, rows: np.ndarray, color: Color = (255, 255, 255), dash: int = 8) -> np.ndarray:
    """Dashed white boxes for ground-truth annotations (drawn under the tracks)."""
    cv2 = _cv2()
    for r in np.asarray(rows).reshape(-1, max(np.asarray(rows).shape[-1], 5)):
        x1, y1, x2, y2 = (int(round(v)) for v in r[1:5])
        for (ax, ay), (bx, by) in (
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ):
            length = max(abs(bx - ax), abs(by - ay))
            for s in range(0, length, 2 * dash):
                t0, t1 = s / max(length, 1), min(s + dash, length) / max(length, 1)
                p0 = (int(ax + (bx - ax) * t0), int(ay + (by - ay) * t0))
                p1 = (int(ax + (bx - ax) * t1), int(ay + (by - ay) * t1))
                cv2.line(frame, p0, p1, color, 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, text: str) -> np.ndarray:
    cv2 = _cv2()
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    cv2.putText(frame, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return frame


def iter_frames(img_dir: Path) -> Iterable[Tuple[int, Path]]:
    """MOTChallenge frames are named 000001.jpg, 000002.jpg, ... (1-based)."""
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.stem.isdigit():
            yield int(p.stem), p


def render_sequence(
    img_dir: str,
    results: Dict[int, np.ndarray],
    out_path: str,
    gt: Optional[Dict[int, np.ndarray]] = None,
    fps: float = 30.0,
    trail_length: int = 30,
    max_frames: int = 0,
    scale: float = 1.0,
) -> int:
    """Render a sequence to ``out_path`` (.mp4 / .avi video, or a directory of PNGs).

    Returns the number of frames written.
    """
    cv2 = _cv2()
    frames = list(iter_frames(Path(img_dir)))
    if max_frames:
        frames = frames[:max_frames]
    if not frames:
        raise FileNotFoundError(f"no frames found in {img_dir}")
    trails = TrailBuffer(trail_length)
    out = Path(out_path)
    to_dir = out.suffix == ""
    writer = None
    written = 0
    empty = np.zeros((0, 5))
    for fid, path in frames:
        img = cv2.imread(str(path))
        if gt is not None:
            draw_ground_truth(img, gt.get(fid, empty))
        rows = results.get(fid, empty)
        draw_tracks(img, rows, trails)
        draw_hud(img, f"frame {fid}  |  tracks {len(rows)}")
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if to_dir:
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{fid:06d}.png"), img)
        else:
            if writer is None:
                out.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out.suffix.lower() == ".mp4" else "MJPG"))
                writer = cv2.VideoWriter(str(out), fourcc, fps, (img.shape[1], img.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"could not open a video writer for {out}")
            writer.write(img)
        written += 1
    if writer is not None:
        writer.release()
    return written
