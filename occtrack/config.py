from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from .tracker import TrackerConfig


def load_tracker_config(path: str | None, **overrides) -> TrackerConfig:
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    data = {**(data or {}).get("tracker", {}), **{k: v for k, v in overrides.items() if v is not None}}
    valid = {f.name for f in fields(TrackerConfig)}
    unknown = set(data) - valid
    if unknown:
        raise KeyError(f"unknown tracker options: {sorted(unknown)}")
    return TrackerConfig(**data)
