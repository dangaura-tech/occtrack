#!/usr/bin/env python
"""Run OccTracker on MOTChallenge sequences using public or private detections.

Example:
    python tools/track_mot.py --data data/MOT17/train --det-name det --config configs/mot17.yaml \
        --out results/mot17_train --filter FRCNN
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from occtrack import OccTracker  # noqa: E402
from occtrack.config import load_tracker_config  # noqa: E402
from occtrack.io import load_mot, read_seqinfo, write_mot  # noqa: E402


def track_sequence(seq_dir: Path, det_file: Path, cfg, out_file: Path) -> tuple[int, float]:
    info = read_seqinfo(str(seq_dir))
    n_frames = int(info.get("seqLength", 0))
    if "frameRate" in info:
        cfg.frame_rate = int(info["frameRate"])
    dets = load_mot(str(det_file))
    n_frames = max(n_frames, max(dets) if dets else 0)
    tracker = OccTracker(cfg)
    results = []
    tic = time.perf_counter()
    for f in range(1, n_frames + 1):
        d = dets.get(f, np.zeros((0, 7)))
        for t in tracker.update(d[:, 1:5], d[:, 5]):
            results.append((f, t.track_id, t.xyxy, t.score))
    elapsed = time.perf_counter() - tic
    write_mot(str(out_file), results)
    return n_frames, elapsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="split folder containing sequence sub-folders")
    p.add_argument("--det-name", default="det", help="detections file: <seq>/det/<det-name>.txt")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="results/tracks")
    p.add_argument("--filter", default="", help="only sequences whose name contains this string")
    args = p.parse_args()

    seqs = sorted(d for d in Path(args.data).iterdir() if d.is_dir() and args.filter in d.name)
    if not seqs:
        raise SystemExit(f"no sequences found in {args.data}")
    total_frames, total_time = 0, 0.0
    for seq in seqs:
        cfg = load_tracker_config(args.config)
        det_file = seq / "det" / f"{args.det_name}.txt"
        n, t = track_sequence(seq, det_file, cfg, Path(args.out) / f"{seq.name}.txt")
        total_frames += n
        total_time += t
        print(f"{seq.name:<20} {n:5d} frames  {n / max(t, 1e-9):7.1f} FPS")
    print(f"total: {total_frames} frames, {total_frames / max(total_time, 1e-9):.1f} FPS (tracking only)")


if __name__ == "__main__":
    main()
