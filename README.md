<div align="center">

# OccTrack

### Occlusion-Aware Multi-Object Tracking by Detection

[Dang Truong Minh Le](https://github.com/dangaura-tech)

Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)

[![CI](https://github.com/dangaura-tech/occtrack/actions/workflows/ci.yml/badge.svg)](https://github.com/dangaura-tech/occtrack/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Abstract

Tracking-by-detection methods break down in crowded scenes: when a pedestrian becomes
partially occluded, the detector's confidence drops, the box becomes noisy, and the
track is either dropped or hijacked by a neighbour. **OccTrack** is a simple,
detector-agnostic online tracker that treats occlusion as a first-class signal. A
geometric **occlusion ratio** is estimated for every detection from the depth ordering of
overlapping boxes and is used in three places:

1. **Confidence/occlusion-adaptive Kalman updates** - the measurement noise is inflated
   for weak or occluded boxes, so the motion model is not dragged by unreliable evidence.
2. **Occlusion-aware track lifetime** - tracks that disappeared while occluded are kept
   alive longer than tracks that simply left the scene.
3. **Occlusion-gated appearance memory** - appearance embeddings are not updated while a
   target is heavily occluded, preventing contamination by the occluder.

These are combined with ByteTrack-style two-stage association (high- and low-score
detections) and chi-square motion gating. The core tracker depends only on NumPy and
SciPy, runs in real time on a CPU, and can be paired with any detector.

## Pipeline

```
 frame t ─► Detector (YOLO / any) ─► boxes, scores ─┬─► occlusion ratio per box
                                    (Re-ID feats)    │
                                                     ▼
 tracks(t-1) ─► Kalman predict ─► Stage 1: high-score dets ↔ confirmed+lost tracks
                                           cost = IoU ⊕ appearance, Mahalanobis-gated
                                  Stage 2: low-score dets ↔ remaining tracks (IoU)
                                  Stage 3: leftover high-score dets ↔ tentative tracks
                                                     │
                       adaptive-noise Kalman update ◄┘
                       lost-track buffer  = T · (1 + γ · 1[occluded])
                       new tracks from unmatched dets with score > τ_new
```

## Installation

```bash
git clone https://github.com/dangaura-tech/occtrack.git
cd occtrack
pip install -e .                 # core tracker (numpy, scipy)
pip install -e ".[video,reid]"   # + YOLO detector, OpenCV, torchvision Re-ID
```

An editable install also provides the console commands `occtrack-mot`, `occtrack-eval`
and `occtrack-ablation` (equivalent to the scripts in `tools/`).

## Quick start

```python
import numpy as np
from occtrack import OccTracker, TrackerConfig

tracker = OccTracker(TrackerConfig(track_buffer=30))
for boxes_xyxy, scores in detections_per_frame:  # your detector output
    for track in tracker.update(boxes_xyxy, scores):
        print(track.track_id, track.xyxy)
```

**Track people in a video**

```bash
python tools/track_video.py --source street.mp4 --output street_tracked.mp4 --weights yolov8m.pt
```

## MOTChallenge benchmarks

Download [MOT17](https://motchallenge.net/data/MOT17/) / [MOT20](https://motchallenge.net/data/MOT20/)
to `data/` (no data is included in this repository), then:

```bash
# track with the public detections
python tools/track_mot.py --data data/MOT17/train --config configs/mot17.yaml \
    --out results/mot17_train --filter FRCNN

# quick evaluation with the built-in CLEAR-MOT / IDF1 implementation
python tools/eval_mot.py --gt data/MOT17/train --pred results/mot17_train --filter FRCNN

# official HOTA / CLEAR / Identity metrics via TrackEval
bash scripts/run_trackeval.sh data/MOT17 results/mot17_train MOT17
```

## Ablations on synthetic crowds

`tools/simulate_ablation.py` generates crowded scenes with a simulated detector whose miss
rate and confidence depend on occlusion, and compares the full tracker with variants that
remove one component at a time. It needs no dataset or GPU:

```bash
python tools/simulate_ablation.py --scenes 5 --targets 10 20 30
```

| Variant | Config |
|---|---|
| OccTrack (full) | `configs/default.yaml` |
| w/o occlusion-aware buffer | `occlusion_buffer_gain: 0` |
| w/o adaptive measurement noise | `adaptive_noise: false` |
| w/o motion gating | `motion_gating: false` |
| w/o low-score association | `second_match_thresh: -1` |
| ByteTrack-like baseline | `configs/bytetrack_like.yaml` |

## Repository structure

```
occtrack/
├── occtrack/
│   ├── tracker.py       # OccTracker: 3-stage association + lifecycle management
│   ├── track.py         # Track state, adaptive-noise updates, appearance memory
│   ├── kalman.py        # constant-velocity Kalman filter (cx, cy, a, h)
│   ├── matching.py      # IoU / appearance costs, gating, Hungarian assignment
│   ├── boxes.py         # box formats, IoU, occlusion ratio
│   ├── simulation.py    # synthetic crowded scenes + detector model
│   ├── eval/            # CLEAR-MOT + IDF1 metrics
│   ├── detectors/       # YOLO wrapper, torchvision Re-ID embedder
│   └── io.py            # MOTChallenge format
├── configs/             # default, MOT17, MOT20, ablation baseline
├── tools/               # track_mot, eval_mot, track_video, simulate_ablation
├── scripts/             # TrackEval wrapper
└── tests/               # unit tests (pytest)
```

## Citation

```bibtex
@misc{le2026occtrack,
  title        = {{OccTrack}: Occlusion-Aware Multi-Object Tracking by Detection},
  author       = {Le, Dang Truong Minh},
  year         = {2026},
  howpublished = {\url{https://github.com/dangaura-tech/occtrack}}
}
```

## Acknowledgements

OccTrack builds on SORT (Bewley et al., 2016), DeepSORT (Wojke et al., 2017) and
ByteTrack (Zhang et al., 2022). Evaluation follows the MOTChallenge protocol and the
TrackEval toolkit (Luiten et al.).

## License

MIT
