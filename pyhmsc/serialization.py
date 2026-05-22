"""Python-native compiled model serialization.

This module defines the replacement boundary for the RDS bridge: JSON metadata
plus HDF5 arrays. The first schema is intentionally fixed-effect-only so it can
be validated and compared against R golden files before broader HMSC features
are added.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "0.1"


def write_compiled_model(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output: str | Path,
    arrays_filename: str = "init_arrays.h5",
) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / arrays_filename
    metadata_path = output / "init.json"

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to write Python-native compiled models") from exc

    with h5py.File(arrays_path, "w") as handle:
        for name, value in arrays.items():
            handle.create_dataset(name, data=np.asarray(value))

    meta = dict(metadata)
    meta["schema_version"] = SCHEMA_VERSION
    meta["arrays"] = {
        name: f"{arrays_path.name}:/{name}"
        for name in arrays
    }
    metadata_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def read_compiled_model(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    arrays = {}

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to read Python-native compiled models") from exc

    for name, ref in metadata.get("arrays", {}).items():
        file_part, dataset = _split_hdf5_ref(ref)
        with h5py.File(path.parent / file_part, "r") as handle:
            arrays[name] = handle[dataset][()]
    return metadata, arrays


def _split_hdf5_ref(ref: str) -> tuple[str, str]:
    if ":/" not in ref:
        raise ValueError(f"Invalid HDF5 array reference {ref!r}")
    file_part, dataset = ref.split(":", 1)
    return file_part, dataset
