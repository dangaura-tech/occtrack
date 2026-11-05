"""Throughput benchmarking of the tracker itself (detector and Re-ID excluded)."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional

import numpy as np

from .simulation import SceneConfig, simulate
from .tracker import OccTracker, TrackerConfig


@dataclass
class BenchmarkResult:
    targets: int
    frames: int
    mean_detections: float
    fps: float
    ms_per_frame_p50: float
    ms_per_frame_p95: float


def benchmark(
    targets: Iterable[int] = (10, 25, 50, 100),
    frames: int = 300,
    with_features: bool = False,
    config: Optional[TrackerConfig] = None,
    warmup: int = 10,
    seed: int = 0,
) -> List[BenchmarkResult]:
    """Time ``OccTracker.update`` on simulated scenes of increasing density.

    Scenes are generated up front, so only the tracking step is measured; frame times are
    collected with ``time.perf_counter`` and summarised as throughput and latency percentiles.
    """
    out = []
    for n in targets:
        scene = simulate(
            replace(
                SceneConfig(),
                num_frames=frames,
                num_targets=n,
                camera_pan=3.0,
                camera_jitter=1.0,
                feature_dim=128 if with_features else 0,
                seed=seed,
            )  # fmt: skip
        )
        tracker = OccTracker(config or TrackerConfig())
        times = []
        for f in sorted(scene.dets):
            d = scene.dets[f]
            feats = scene.features[f] if with_features else None
            t0 = time.perf_counter()
            tracker.update(d[:, 1:5], d[:, 5], feats, warp=scene.warps[f])
            if f > warmup:
                times.append(time.perf_counter() - t0)
        t = np.array(times)
        out.append(
            BenchmarkResult(
                targets=n,
                frames=len(t),
                mean_detections=float(np.mean([len(v) for v in scene.dets.values()])),
                fps=float(len(t) / max(t.sum(), 1e-12)),
                ms_per_frame_p50=float(np.percentile(t, 50) * 1e3),
                ms_per_frame_p95=float(np.percentile(t, 95) * 1e3),
            )
        )
    return out


def as_markdown(results: List[BenchmarkResult], title: str = "") -> str:
    lines = [f"### {title}", ""] if title else []
    lines += ["| Targets | Detections / frame | FPS | p50 (ms) | p95 (ms) |", "|---|---|---|---|---|"]
    for r in results:
        cells = [str(r.targets), f"{r.mean_detections:.1f}", f"{r.fps:.0f}", f"{r.ms_per_frame_p50:.2f}",
                 f"{r.ms_per_frame_p95:.2f}"]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def as_dict(results: List[BenchmarkResult]) -> List[Dict]:
    return [r.__dict__.copy() for r in results]
