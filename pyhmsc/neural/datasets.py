"""Dataset configuration helpers for experimental Neural-HMSC benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    """Load a Neural-HMSC benchmark YAML or JSON config."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install PyYAML to read benchmark config files") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Benchmark config must be a mapping")
    return data


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    """Write stable, human-readable JSON metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

