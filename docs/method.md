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
