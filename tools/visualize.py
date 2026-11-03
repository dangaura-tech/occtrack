#!/usr/bin/env python
"""Render a MOTChallenge result file onto the frames of a sequence.

Examples:
    python tools/visualize.py --seq data/MOT17/train/MOT17-04-FRCNN \
        --results results/mot17_train/MOT17-04-FRCNN.txt --out vis/MOT17-04.mp4 --gt
    python tools/visualize.py --seq ... --results ... --out vis/frames/   # PNG frames instead of a video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from occtrack.io import load_mot, read_seqinfo  # noqa: E402
from occtrack.visualize import render_sequence  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", required=True, help="sequence folder containing img1/ (and gt/gt.txt)")
    p.add_argument("--results", required=True, help="tracker output in MOTChallenge format")
    p.add_argument("--out", required=True, help=".mp4 / .avi file, or a directory for PNG frames")
    p.add_argument("--gt", action="store_true", help="overlay ground truth (dashed white boxes)")
    p.add_argument("--trail", type=int, default=30, help="trail length in frames")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--scale", type=float, default=1.0)
    args = p.parse_args()

    seq = Path(args.seq)
    info = read_seqinfo(str(seq))
    fps = float(info.get("frameRate", 30))
    img_dir = seq / info.get("imDir", "img1")
    gt = load_mot(str(seq / "gt" / "gt.txt"), gt=True) if args.gt else None
    n = render_sequence(
        str(img_dir), load_mot(args.results), args.out, gt, fps, args.trail, args.max_frames, args.scale
    )
    print(f"wrote {n} frames to {args.out}")


if __name__ == "__main__":
    main()
