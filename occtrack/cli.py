"""Console entry points so the tools are available after ``pip install``.

The implementations live in ``tools/`` (kept as plain scripts for readability); these
wrappers load them by path so both ``python tools/x.py`` and ``occtrack-x`` work.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _run(name: str) -> None:
    path = _TOOLS / f"{name}.py"
    if not path.exists():
        raise SystemExit(f"{path} not found - entry points require an editable install (pip install -e .)")
    spec = importlib.util.spec_from_file_location(f"occtrack_tools_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()


def track_mot() -> None:
    _run("track_mot")


def eval_mot() -> None:
    _run("eval_mot")


def simulate_ablation() -> None:
    _run("simulate_ablation")
