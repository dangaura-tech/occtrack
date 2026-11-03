"""Offline post-processing of tracking results.

``interpolate_tracks`` fills short gaps inside each trajectory by linear interpolation of
box coordinates, recovering frames where the target was occluded and not detected. An
optional ``smooth_sigma`` applies Gaussian smoothing along time (in the spirit of GSI,
Du et al. 2022) to reduce box jitter.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
from scipy.ndimage import gaussian_filter1d


def interpolate_tracks(
    results: Dict[int, np.ndarray], max_gap: int = 20, smooth_sigma: float = 0.0
) -> Dict[int, np.ndarray]:
    """``results``: ``{frame: array[K, >=5]}`` with columns ``id, x1, y1, x2, y2[, score]``."""
    per_id: Dict[int, list] = defaultdict(list)
    for f, rows in results.items():
        for r in rows:
            score = r[5] if len(r) > 5 else 1.0
            per_id[int(r[0])].append((f, *r[1:5], score))
    out: Dict[int, list] = defaultdict(list)
    for tid, rows in per_id.items():
        arr = np.array(sorted(rows))
        frames = arr[:, 0].astype(int)
        filled = [arr[0]]
        for prev, cur in zip(arr[:-1], arr[1:]):  # same length by construction (Python 3.9 has no strict=)
            gap = int(cur[0] - prev[0])
            if 1 < gap <= max_gap:
                for k in range(1, gap):
                    w = k / gap
                    box = (1 - w) * prev[1:5] + w * cur[1:5]
                    filled.append(np.r_[prev[0] + k, box, min(prev[5], cur[5])])
            filled.append(cur)
        track = np.array(filled)
        if smooth_sigma > 0 and len(track) > 2:
            # smooth only within contiguous segments
            seg_start = 0
            frames = track[:, 0].astype(int)
            for i in range(1, len(track) + 1):
                if i == len(track) or frames[i] - frames[i - 1] > 1:
                    if i - seg_start > 2:
                        track[seg_start:i, 1:5] = gaussian_filter1d(track[seg_start:i, 1:5], smooth_sigma, axis=0)
                    seg_start = i
        for row in track:
            out[int(row[0])].append(np.r_[tid, row[1:5], row[5]])
    return {f: np.stack(v) for f, v in sorted(out.items())}
