"""OccTrack: occlusion-aware multi-object tracking by detection."""

from .track import Track, TrackState
from .tracker import OccTracker, TrackerConfig

__version__ = "0.3.0"
__all__ = ["OccTracker", "TrackerConfig", "Track", "TrackState"]
