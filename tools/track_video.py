#!/usr/bin/env python
"""Detect + track people in a video file or webcam and write an annotated video.

Requires: pip install ultralytics opencv-python (and torch/torchvision for --reid).

Example:
    python tools/track_video.py --source street.mp4 --output street_tracked.mp4 --weights yolov8m.pt
"""

from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from occtrack import OccTracker  # noqa: E402
from occtrack.config import load_tracker_config  # noqa: E402
from occtrack.detectors.yolo import YOLODetector  # noqa: E402


def color_for(track_id: int) -> tuple[int, int, int]:
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def main() -> None:
    import cv2

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video path or webcam index")
    p.add_argument("--output", default="tracked.mp4")
    p.add_argument("--weights", default="yolov8m.pt")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--reid", action="store_true", help="use a torchvision appearance embedder")
    p.add_argument("--no-cmc", action="store_true", help="disable camera motion compensation")
    p.add_argument("--device", default=None)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector = YOLODetector(args.weights, device=args.device)
    embedder = None
    if args.reid:
        from occtrack.detectors.reid import TorchvisionEmbedder

        embedder = TorchvisionEmbedder(device=args.device or "cpu")
    tracker = OccTracker(load_tracker_config(args.config, frame_rate=int(round(fps))))
    cmc = None
    if not args.no_cmc:
        from occtrack.cmc import SparseFlowCMC

        cmc = SparseFlowCMC()

    trails: dict[int, list] = {}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        boxes, scores = detector(frame)
        feats = embedder(frame, boxes) if embedder is not None else None
        warp = cmc.apply(frame, boxes[scores > 0.5]) if cmc is not None else None
        for t in tracker.update(boxes, scores, feats, warp=warp):
            x1, y1, x2, y2 = t.xyxy.astype(int)
            c = color_for(t.track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"#{t.track_id}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            trail = trails.setdefault(t.track_id, [])
            trail.append(((x1 + x2) // 2, y2))
            del trail[:-40]
            if len(trail) > 1:
                cv2.polylines(frame, [np.array(trail, np.int32)], False, c, 2)
        cv2.putText(
            frame,
            f"frame {frame_idx}  tracks {len(tracker.tracks)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        writer.write(frame)
        if args.show:
            cv2.imshow("OccTrack", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    writer.release()
    print(f"wrote {frame_idx} frames to {args.output}")


if __name__ == "__main__":
    main()
