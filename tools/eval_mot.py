#!/usr/bin/env python
"""Evaluate tracker outputs against MOTChallenge ground truth with the built-in metrics.

Example:
    python tools/eval_mot.py --gt data/MOT17/train --pred results/mot17_train --filter FRCNN
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from occtrack.eval import evaluate_sequence, summarize  # noqa: E402
from occtrack.eval.metrics import format_table  # noqa: E402
from occtrack.io import load_mot  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--filter", default="")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    per_seq = {}
    for pred_file in sorted(Path(args.pred).glob("*.txt")):
        if args.filter not in pred_file.stem:
            continue
        gt_file = Path(args.gt) / pred_file.stem / "gt" / "gt.txt"
        if not gt_file.exists():
            print(f"skip {pred_file.stem}: no ground truth")
            continue
        per_seq[pred_file.stem] = evaluate_sequence(load_mot(str(gt_file), gt=True), load_mot(str(pred_file)), args.iou)
    if not per_seq:
        raise SystemExit("nothing to evaluate")
    rows = list(per_seq.items()) + [("**OVERALL**", summarize(per_seq))]
    print(format_table(rows))
    if args.json:
        Path(args.json).write_text(json.dumps({k: v for k, v in rows}, indent=2, default=float))


if __name__ == "__main__":
    main()
