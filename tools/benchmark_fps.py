#!/usr/bin/env python
"""Measure tracker-only throughput on simulated scenes of increasing crowd density.

    python tools/benchmark_fps.py --targets 10 25 50 100 --frames 300 [--features] [--json out.json]

Detection and Re-ID time are excluded (scenes are simulated up front), so the numbers
reflect the association and filtering cost of OccTrack on the current machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from occtrack.benchmark import as_dict, as_markdown, benchmark  # noqa: E402
from occtrack.config import load_tracker_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", type=int, nargs="+", default=[10, 25, 50, 100])
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--features", action="store_true", help="also pass 128-d appearance embeddings")
    p.add_argument("--config", default=None, help="tracker config YAML (default: built-in defaults)")
    p.add_argument("--json", default=None, help="write raw results to this file")
    args = p.parse_args()

    cfg = load_tracker_config(args.config) if args.config else None
    res = benchmark(args.targets, args.frames, args.features, cfg)
    cpu = platform.processor() or platform.machine()
    title = f"OccTrack tracking-only throughput ({cpu}, Python {platform.python_version()})"
    print(as_markdown(res, title))
    if args.json:
        Path(args.json).write_text(json.dumps(as_dict(res), indent=2))


if __name__ == "__main__":
    main()
