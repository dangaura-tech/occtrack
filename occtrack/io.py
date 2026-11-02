"""MOTChallenge text-format readers and writers.

Each line: ``frame, id, x, y, w, h, conf, class, visibility[, ...]`` (1-based frames).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def load_mot(path: str, min_conf: float = -np.inf, gt: bool = False) -> Dict[int, np.ndarray]:
    """Return ``{frame: array[K, 7]}`` with columns ``id, x1, y1, x2, y2, conf, visibility``.

    For ground truth (``gt=True``) only rows with ``conf == 1`` and pedestrian classes
    (1 = pedestrian, 7 = static person) are kept, following the MOT17 protocol.
    """
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    out: Dict[int, List[np.ndarray]] = defaultdict(list)
    for row in raw:
        frame, tid, x, y, w, h, conf = row[:7]
        if gt:
            cls = int(row[7]) if len(row) > 7 else 1
            if conf == 0 or cls not in (1, 7):
                continue
        elif conf < min_conf:
            continue
        vis = row[8] if len(row) > 8 else 1.0
        out[int(frame)].append(np.array([tid, x, y, x + w, y + h, conf, vis]))
    return {f: np.stack(v) for f, v in out.items()}


def write_mot(path: str, results: Iterable[Tuple[int, int, np.ndarray, float]]) -> None:
    """Write ``(frame, track_id, xyxy, score)`` tuples in MOTChallenge format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame, tid, b, s in results:
            x1, y1, x2, y2 = b
            f.write(f"{frame},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.3f},-1,-1,-1\n")


def read_seqinfo(seq_dir: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    p = Path(seq_dir) / "seqinfo.ini"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
    return info
