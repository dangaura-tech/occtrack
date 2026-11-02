"""Ultralytics YOLO detector wrapper (``pip install ultralytics``)."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


class YOLODetector:
    def __init__(
        self,
        weights: str = "yolov8m.pt",
        classes: Sequence[int] = (0,),
        conf: float = 0.05,
        imgsz: int = 1280,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("YOLODetector requires `pip install ultralytics`") from e
        self.model = YOLO(weights)
        self.classes = list(classes)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(boxes_xyxy [N, 4], scores [N])`` for one BGR frame."""
        res = self.model.predict(
            frame_bgr, conf=self.conf, classes=self.classes, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float64)
        scores = res.boxes.conf.cpu().numpy().astype(np.float64)
        return boxes, scores
