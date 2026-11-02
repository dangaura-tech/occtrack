#!/usr/bin/env python
"""Render an animated GIF of OccTrack on a simulated crowded scene (no data needed).

Grey boxes are raw detections (opacity = confidence); coloured boxes are confirmed tracks
with their ID and a motion trail. The scene uses a panning, shaking camera, so camera motion
compensation is active.

    python tools/demo_gif.py --out assets/occtrack_demo.gif
"""

from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from occtrack import OccTracker, TrackerConfig  # noqa: E402
from occtrack.simulation import SceneConfig, simulate  # noqa: E402


def color(track_id: int) -> tuple:
    r, g, b = colorsys.hsv_to_rgb((track_id * 0.61803398875) % 1.0, 0.7, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="assets/occtrack_demo.gif")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--targets", type=int, default=18)
    p.add_argument("--scale", type=float, default=1 / 3)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()

    cfg = SceneConfig(num_frames=args.frames, num_targets=args.targets, camera_pan=4.0, camera_jitter=1.5,
                      feature_dim=32, seed=args.seed)  # fmt: skip
    scene = simulate(cfg)
    tracker = OccTracker(TrackerConfig())
    W, H = int(cfg.width * args.scale), int(cfg.height * args.scale)
    trails: dict = {}
    frames = []
    for f in sorted(scene.dets):
        d = scene.dets[f]
        tracks = tracker.update(d[:, 1:5], d[:, 5], scene.features[f], warp=scene.warps[f])
        # move existing trails with the camera so they stay attached to the ground
        shift = scene.warps[f][:, 2]
        for tid in trails:
            trails[tid] = [(x + shift[0], y + shift[1]) for x, y in trails[tid]]

        img = Image.new("RGB", (W, H), (16, 20, 28))
        draw = ImageDraw.Draw(img, "RGBA")
        for x1, y1, x2, y2, s in d[:, 1:6]:
            a = int(60 + 120 * s)
            draw.rectangle([x1 * args.scale, y1 * args.scale, x2 * args.scale, y2 * args.scale],
                           outline=(170, 170, 170, a), width=1)  # fmt: skip
        for t in tracks:
            x1, y1, x2, y2 = t.xyxy * args.scale
            c = color(t.track_id)
            trails.setdefault(t.track_id, []).append(((x1 + x2) / 2 / args.scale, y2 / args.scale))
            trails[t.track_id] = trails[t.track_id][-25:]
            pts = [(x * args.scale, y * args.scale) for x, y in trails[t.track_id]]
            if len(pts) > 1:
                draw.line(pts, fill=c + (200,), width=2)
            draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
            draw.rectangle([x1, y1 - 14, x1 + 30, y1], fill=c)
            draw.text((x1 + 3, y1 - 13), f"{t.track_id}", fill=(0, 0, 0))
        draw.rectangle([0, 0, W, 22], fill=(0, 0, 0, 150))
        draw.text((8, 5), f"OccTrack v2 | frame {f:03d} | tracks {len(tracks)} | CMC + ORU + OCM + Re-ID",
                  fill=(230, 230, 230))  # fmt: skip
        frames.append(img)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {len(frames)} frames to {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
