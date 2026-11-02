#!/usr/bin/env python
"""Ablation study on synthetic crowded scenes (no dataset or GPU needed).

Compares the full OccTracker with variants that disable individual components, averaged
over several random scenes and occlusion levels.

Example:
    python tools/simulate_ablation.py --scenes 5 --targets 10 20 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from occtrack import OccTracker, TrackerConfig  # noqa: E402
from occtrack.eval import evaluate_sequence  # noqa: E402
from occtrack.simulation import SceneConfig, simulate_scene  # noqa: E402

VARIANTS = {
    "OccTrack (full)": {},
    "w/o occlusion buffer": {"occlusion_buffer_gain": 0.0},
    "w/o adaptive noise": {"adaptive_noise": False},
    "w/o motion gating": {"motion_gating": False},
    "w/o low-score stage": {"second_match_thresh": -1.0},
}


def run(cfg: TrackerConfig, dets) -> dict:
    tracker = OccTracker(cfg)
    out = {}
    for f in sorted(dets):
        d = dets[f]
        tracks = tracker.update(d[:, 1:5], d[:, 5])
        out[f] = np.array([[t.track_id, *t.xyxy] for t in tracks]).reshape(-1, 5)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", type=int, default=5)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--targets", type=int, nargs="+", default=[10, 20, 30])
    args = p.parse_args()

    for n in args.targets:
        print(f"\n### {n} targets per scene ({args.scenes} scenes x {args.frames} frames)\n")
        print("| Variant | MOTA | IDF1 | IDSW |")
        print("|---|---|---|---|")
        scenes = [
            simulate_scene(SceneConfig(num_frames=args.frames, num_targets=n, seed=s)) for s in range(args.scenes)
        ]
        for name, kw in VARIANTS.items():
            res = [evaluate_sequence(gt, run(TrackerConfig(**kw), dets)) for gt, dets in scenes]
            mota = np.mean([r["MOTA"] for r in res]) * 100
            idf1 = np.mean([r["IDF1"] for r in res]) * 100
            idsw = np.mean([r["IDSW"] for r in res])
            print(f"| {name} | {mota:.1f} | {idf1:.1f} | {idsw:.1f} |")


if __name__ == "__main__":
    main()
