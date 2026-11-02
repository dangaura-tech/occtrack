#!/usr/bin/env bash
# Evaluate tracker output with the official TrackEval toolkit (HOTA, CLEAR, Identity).
# Usage: bash scripts/run_trackeval.sh <MOT root, e.g. data/MOT17> <results dir> <benchmark: MOT17|MOT20>
set -euo pipefail

GT_ROOT=${1:?MOT root}
RESULTS=${2:?results dir}
BENCH=${3:-MOT17}
TE_DIR=${TRACKEVAL_DIR:-third_party/TrackEval}

if [ ! -d "$TE_DIR" ]; then
  git clone --depth 1 https://github.com/JonathonLuiten/TrackEval.git "$TE_DIR"
fi

TRACKER=occtrack
WORK=$(mktemp -d)
mkdir -p "$WORK/trackers/$BENCH-train/$TRACKER/data"
cp "$RESULTS"/*.txt "$WORK/trackers/$BENCH-train/$TRACKER/data/"

python "$TE_DIR/scripts/run_mot_challenge.py" \
  --BENCHMARK "$BENCH" --SPLIT_TO_EVAL train \
  --GT_FOLDER "$(dirname "$GT_ROOT")" \
  --TRACKERS_FOLDER "$WORK/trackers" --TRACKERS_TO_EVAL "$TRACKER" \
  --METRICS HOTA CLEAR Identity --USE_PARALLEL False --PRINT_CONFIG False
