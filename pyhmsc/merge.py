"""Merge Python-native HDF5 posterior shards."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChainStatus:
    chain: int
    path: Path
    status: str
    message: str


def merge_hdf5_posteriors(
    inputs: list[str | Path],
    output: str | Path,
    expected_chains: list[int] | None = None,
) -> Path:
    """Merge per-chain HDF5 posteriors into one posterior file.

    All datasets are concatenated along axis 0, the chain dimension. Root
    metadata must match across inputs when present.
    """
    if not inputs:
        raise ValueError("merge requires at least one input posterior")
    input_paths = [Path(path) for path in inputs]
    if expected_chains is not None:
        missing = missing_expected_chains(input_paths, expected_chains)
        if missing:
            raise ValueError(f"Missing expected chain outputs: {missing}")
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


def inspect_chain_directory(
    directory: str | Path,
    expected_chains: list[int],
    expected_draws: int | None = None,
) -> list[ChainStatus]:
    directory = Path(directory)
    files = {chain: path for chain, path in chain_files(directory).items()}
    baseline_metadata = None
    statuses = []
    for chain in expected_chains:
        path = files.get(chain, directory / f"posterior_chain_{chain}.h5")
        status, message, metadata = inspect_chain_file(path, expected_draws=expected_draws)
        if status == "passed":
            if baseline_metadata is None:
                baseline_metadata = metadata
            elif metadata != baseline_metadata:
                status = "failed"
                message = "metadata mismatch"
        statuses.append(ChainStatus(chain=chain, path=path, status=status, message=message))
    return statuses


def inspect_chain_file(
    path: str | Path,
    expected_draws: int | None = None,
) -> tuple[str, str, Any]:
    path = Path(path)
    if not path.exists():
        return "missing", "file not found", None
    if path.stat().st_size == 0:
        return "failed", "empty file", None
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to inspect HDF5 posterior files") from exc
    try:
        with h5py.File(path, "r") as handle:
            if "Beta" not in handle:
                return "failed", "missing Beta dataset", _read_metadata_attr(handle)
            beta = handle["Beta"]
            if beta.ndim < 2:
                return "failed", "Beta dataset has invalid rank", _read_metadata_attr(handle)
            if int(beta.shape[0]) < 1:
                return "failed", "no chain draws in Beta dataset", _read_metadata_attr(handle)
            if expected_draws is not None and int(beta.shape[1]) != expected_draws:
                return (
                    "failed",
                    f"expected {expected_draws} draws, found {int(beta.shape[1])}",
                    _read_metadata_attr(handle),
                )
            return "passed", "ok", _read_metadata_attr(handle)
    except OSError as exc:
        return "failed", f"unreadable HDF5: {exc}", None


def chain_files(directory: str | Path) -> dict[int, Path]:
    directory = Path(directory)
    files = {}
    for path in directory.glob("posterior_chain_*.h5"):
        match = re.search(r"posterior_chain_(\d+)\.h5$", path.name)
        if match:
            files[int(match.group(1))] = path
    return files


def missing_expected_chains(inputs: list[Path], expected_chains: list[int]) -> list[int]:
    present = set()
    for path in inputs:
        match = re.search(r"posterior_chain_(\d+)\.h5$", path.name)
        if match:
            present.add(int(match.group(1)))
    return [chain for chain in expected_chains if chain not in present]


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
