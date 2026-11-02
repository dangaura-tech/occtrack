#!/usr/bin/env python
"""Ablation study on synthetic crowded scenes (no dataset or GPU needed).

Each scenario stresses a different failure mode (static camera, panning camera, crowded
scene with appearance cues); the full tracker is compared with variants that disable one
component at a time. Metrics: HOTA, AssA, MOTA, IDF1, IDSW (mean over scenes).

Example:
    python tools/simulate_ablation.py --scenes 5 --frames 300
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from occtrack import OccTracker, TrackerConfig  # noqa: E402
from occtrack.eval import evaluate_sequence  # noqa: E402
from occtrack.simulation import SceneConfig, simulate  # noqa: E402

VARIANTS = {
    "OccTrack v2 (full)": {},
    "w/o CMC": {"cmc": False},
    "w/o ORU": {"re_update": False},
    "w/o OCM": {"momentum_weight": 0.0},
    "w/o Re-ID memory stage": {"reid_stage": False},
    "w/o occlusion buffer": {"occlusion_buffer_gain": 0.0},
    "w/o adaptive noise": {"adaptive_noise": False},
    "w/o low-score stage": {"second_match_thresh": -1.0},
}

SCENARIOS = {
    "static camera": SceneConfig(num_targets=20),
    "panning camera": SceneConfig(num_targets=20, camera_pan=8.0, camera_jitter=2.0),
    "crowd + Re-ID": SceneConfig(num_targets=35, feature_dim=64, camera_pan=3.0, camera_jitter=1.0),
}


def run(cfg: TrackerConfig, scene, use_features: bool) -> dict:
    tracker = OccTracker(cfg)
    out = {}
    for f in sorted(scene.dets):
        d = scene.dets[f]
        feats = scene.features[f] if use_features else None
        tracks = tracker.update(d[:, 1:5], d[:, 5], feats, warp=scene.warps[f])
        out[f] = np.array([[t.track_id, *t.xyxy] for t in tracks]).reshape(-1, 5)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", type=int, default=5)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--scenario", choices=list(SCENARIOS), nargs="*", default=list(SCENARIOS))
    args = p.parse_args()

    for name in args.scenario:
        base = SCENARIOS[name]
        scenes = [simulate(replace(base, num_frames=args.frames, seed=s)) for s in range(args.scenes)]
        use_feats = base.feature_dim > 0
        print(f"\n### {name} ({args.scenes} scenes x {args.frames} frames, {base.num_targets} targets)\n")
        print("| Variant | HOTA | AssA | MOTA | IDF1 | IDSW |")
        print("|---|---|---|---|---|---|")
        for variant, kw in VARIANTS.items():
            res = [evaluate_sequence(s.gt, run(TrackerConfig(**kw), s, use_feats)) for s in scenes]
            m = {k: np.mean([r[k] for r in res]) for k in ("HOTA", "AssA", "MOTA", "IDF1", "IDSW")}
            print(
                f"| {variant} | {m['HOTA'] * 100:.1f} | {m['AssA'] * 100:.1f} | {m['MOTA'] * 100:.1f} "
                f"| {m['IDF1'] * 100:.1f} | {m['IDSW']:.1f} |"
            )


if __name__ == "__main__":
    main()
