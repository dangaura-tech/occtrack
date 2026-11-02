# Method details

## Occlusion ratio
For detection *i* the occlusion ratio is

    o_i = max_{j : y2_j > y2_i} |B_i ∩ B_j| / |B_i|

i.e. the largest fraction of box *i* covered by a box whose bottom edge is lower in the
image (assumed closer to a camera looking down at a ground plane).

## Adaptive measurement noise
The Kalman measurement covariance is scaled by

    s_i = (1 + 2 o_i) / max(score_i, 0.1)

so that a fully visible, confident detection is trusted as usual, while a weak or
occluded detection mostly updates the *existence* of the track rather than its state.

## Occlusion-aware lifetime
A lost track is removed after `T (1 + γ)` frames if it was occluded (o >= τ_o) when last
seen, and after `T` frames otherwise. `T = track_buffer · frame_rate / 30`.

## Association
| Stage | Detections | Tracks | Cost | Threshold |
|---|---|---|---|---|
| 1 | score ≥ τ_high | confirmed + lost | (1-w)·(1-IoU·score) + w·cos-dist, Mahalanobis-gated | `match_thresh` |
| 2 | τ_low ≤ score < τ_high | unmatched confirmed | 1-IoU | `second_match_thresh` |
| 3 | unmatched high | tentative | 1-IoU·score | `tentative_match_thresh` |

## v2 components

### Camera motion compensation
`SparseFlowCMC` detects Shi-Tomasi corners (outside detection boxes and image borders) on a
downscaled grey image, tracks them with pyramidal Lucas-Kanade flow, and fits a similarity
transform with RANSAC. For a warp `[A | t]`, the state `(cx, cy, a, h, vx, vy, va, vh)` is
mapped with `(cx, cy) <- A (cx, cy) + t`, `(vx, vy) <- A (vx, vy)`, `h, vh <- s h, s vh`
where `s = sqrt|det A|`. The covariance uses the same linear map. Saved states and stored
observations are warped as well, so ORU and OCM stay consistent across frames.

### Observation-centric re-update (ORU)
A track re-associated after `g > 1` frames restores the state saved at its last
observation. It then alternates predict / update with virtual observations
`z_last + (z_new - z_last) * i / g` for `i = 1..g-1`, before the regular update. This
replaces the purely extrapolated (and usually drifted) state.

### Observation-centric momentum (OCM)
The track direction is the unit vector between observations ~`momentum_delta` frames apart.
For each detection, the angle between that direction and the vector from the last
observation to the detection, divided by pi and multiplied by the detection score, is added
to the stage-1 cost with weight `momentum_weight`.

### Occlusion-aware Re-ID memory and stage 4
Each track stores up to `memory_size` L2-normalised embeddings. A new embedding is added
only if the target's occlusion ratio is below `occlusion_thresh` and its cosine similarity
to every stored prototype is below 0.95. Stage 4 matches remaining confident detections to
tracks that have been lost for more than one frame. The cost is the minimum cosine distance
to the memory, restricted to a window of radius `reid_spatial_sigma * std(centre) + h`
around the predicted centre. The window grows automatically while the track is lost.

### HOTA
`occtrack.eval.hota` mirrors TrackEval. A global alignment score between (gt, tracker)
identities, computed from IoU-normalised similarities over the whole sequence, biases the
per-frame Hungarian matching. DetA, AssA and LocA are computed for alpha = 0.05..0.95 and
averaged; HOTA = sqrt(DetA * AssA).
