"""Merge Python-native HDF5 posterior shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def merge_hdf5_posteriors(inputs: list[str | Path], output: str | Path) -> Path:
    """Merge per-chain HDF5 posteriors into one posterior file.

    All datasets are concatenated along axis 0, the chain dimension. Root
    metadata must match across inputs when present.
    """
    if not inputs:
        raise ValueError("merge requires at least one input posterior")
    input_paths = [Path(path) for path in inputs]
    output_path = Path(output)
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to merge HDF5 posterior files") from exc

    metadata = None
    metadata_seen = False
    expected_keys = None
    arrays_by_key: dict[str, list[np.ndarray]] = {}
    elapsed = 0.0
    for path in input_paths:
        with h5py.File(path, "r") as handle:
            candidate = _read_metadata_attr(handle)
            if not metadata_seen:
                metadata = candidate
                metadata_seen = True
            elif candidate != metadata:
                raise ValueError(f"Posterior metadata does not match for {path}")
            elapsed += float(handle.attrs.get("time", 0.0))
            datasets = dict(_iter_datasets(handle))
            keys = set(datasets)
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                missing = sorted(expected_keys.difference(keys))
                extra = sorted(keys.difference(expected_keys))
                raise ValueError(
                    f"Posterior datasets do not match for {path}; "
                    f"missing={missing}, extra={extra}"
                )
            for key, value in datasets.items():
                arrays_by_key.setdefault(key, []).append(np.asarray(value))

    if not arrays_by_key:
        raise ValueError("Input posteriors contain no datasets")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.attrs["time"] = elapsed
        if metadata is not None:
            handle.attrs["pyhmsc_metadata"] = json.dumps(metadata)
        n_chains = None
        for key, arrays in arrays_by_key.items():
            _validate_compatible_arrays(key, arrays)
            data = np.concatenate(arrays, axis=0)
            if n_chains is None:
                n_chains = int(data.shape[0])
            _create_dataset(handle, key, data)
        handle.attrs["nChains"] = int(n_chains or 0)
    return output_path


def _read_metadata_attr(handle: Any) -> Any:
    if "pyhmsc_metadata" not in handle.attrs:
        return None
    value = handle.attrs["pyhmsc_metadata"]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def _iter_datasets(group: Any, prefix: str = ""):
    for name, value in group.items():
        key = f"{prefix}/{name}" if prefix else name
        if hasattr(value, "keys"):
            yield from _iter_datasets(value, key)
        else:
            yield key, value[()]


def _validate_compatible_arrays(key: str, arrays: list[np.ndarray]) -> None:
    if not arrays:
        raise ValueError(f"No arrays for {key}")
    tail = arrays[0].shape[1:]
    dtype = arrays[0].dtype
    for array in arrays[1:]:
        if array.shape[1:] != tail:
            raise ValueError(f"Dataset {key} has incompatible shapes")
        if array.dtype != dtype:
            raise ValueError(f"Dataset {key} has incompatible dtypes")


def _create_dataset(handle: Any, key: str, data: np.ndarray) -> None:
    parts = key.split("/")
    group = handle
    for part in parts[:-1]:
        group = group.require_group(part)
    group.create_dataset(parts[-1], data=data)
