"""Class-aware tracking: one :class:`OccTracker` per category with globally unique IDs.

Running the association per class prevents identity switches between objects of different
categories (a cyclist's box is never matched to a pedestrian track), which matters for
open-vocabulary detectors that report several categories at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .track import Track
from .tracker import OccTracker, TrackerConfig


@dataclass
class LabelledTrack:
    track_id: int  # unique across classes
    label: int
    track: Track

    @property
    def xyxy(self) -> np.ndarray:
        return self.track.xyxy


class MultiClassTracker:
    def __init__(self, config: Optional[TrackerConfig] = None, factory: Optional[Callable[[], OccTracker]] = None):
        self._factory = factory or (lambda: OccTracker(config))
        self.trackers: Dict[int, OccTracker] = {}
        self._ids: Dict[Tuple[int, int], int] = {}
        self._next_id = 1

    def _global_id(self, label: int, local_id: int) -> int:
        key = (label, local_id)
        if key not in self._ids:
            self._ids[key] = self._next_id
            self._next_id += 1
        return self._ids[key]

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: Sequence[int],
        features: Optional[np.ndarray] = None,
        warp: Optional[np.ndarray] = None,
    ) -> List[LabelledTrack]:
        """Advance every class tracker by one frame (also those without detections, so tracks age)."""
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        labels = np.asarray(labels, int).reshape(-1)
        for c in np.unique(labels):
            if int(c) not in self.trackers:
                self.trackers[int(c)] = self._factory()
        out: List[LabelledTrack] = []
        for c, tracker in sorted(self.trackers.items()):
            m = labels == c
            f = features[m] if features is not None else None
            for t in tracker.update(boxes[m], scores[m], f, warp=warp):
                out.append(LabelledTrack(self._global_id(c, t.track_id), c, t))
        return out
