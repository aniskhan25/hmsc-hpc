"""Posterior storage inspection helpers for HDF5 and optional Zarr outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    chunks: tuple[int, ...] | None


@dataclass(frozen=True)
class PosteriorStorageInfo:
    path: Path
    format: str
    datasets: list[DatasetInfo]
    metadata_present: bool
    n_chains: int | None
    n_draws: int | None
    total_nbytes: int
    attrs: dict[str, Any]

    def to_text(self) -> str:
        lines = [
            "posterior storage",
            f"path: {self.path}",
            f"format: {self.format}",
            f"metadata_present: {self.metadata_present}",
            f"n_chains: {self.n_chains}",
            f"n_draws: {self.n_draws}",
            f"total_nbytes: {self.total_nbytes}",
            "datasets:",
        ]
        for dataset in self.datasets:
            chunks = dataset.chunks if dataset.chunks is not None else "None"
            lines.append(
                f"  {dataset.name}: shape={dataset.shape} dtype={dataset.dtype} "
                f"nbytes={dataset.nbytes} chunks={chunks}"
            )
        return "\n".join(lines) + "\n"


def inspect_posterior_storage(path: str | Path) -> PosteriorStorageInfo:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return _inspect_hdf5(path)
    if suffix == ".zarr":
        return _inspect_zarr(path)
    raise ValueError(f"Unsupported posterior storage format for {path}")


def _inspect_hdf5(path: Path) -> PosteriorStorageInfo:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to inspect HDF5 posterior files") from exc

    datasets = []
    with h5py.File(path, "r") as handle:
        attrs = _json_safe_attrs(dict(handle.attrs))
        _collect_hdf5_datasets(handle, datasets)
    return _info_from_datasets(
        path=path,
        format_name="hdf5",
        datasets=datasets,
        attrs=attrs,
        metadata_present="pyhmsc_metadata" in attrs,
    )


def _inspect_zarr(path: Path) -> PosteriorStorageInfo:
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install zarr to inspect Zarr posterior stores") from exc

    root = zarr.open_group(str(path), mode="r")
    datasets = []
    _collect_zarr_datasets(root, datasets)
    attrs = _json_safe_attrs(dict(root.attrs))
    return _info_from_datasets(
        path=path,
        format_name="zarr",
        datasets=datasets,
        attrs=attrs,
        metadata_present="pyhmsc_metadata" in attrs,
    )


def _collect_hdf5_datasets(group: Any, datasets: list[DatasetInfo], prefix: str = "") -> None:
    for name, value in group.items():
        key = f"{prefix}/{name}" if prefix else name
        if hasattr(value, "keys"):
            _collect_hdf5_datasets(value, datasets, key)
        else:
            datasets.append(
                DatasetInfo(
                    name=key,
                    shape=tuple(int(dim) for dim in value.shape),
                    dtype=str(value.dtype),
                    nbytes=int(value.size * value.dtype.itemsize),
                    chunks=tuple(int(dim) for dim in value.chunks) if value.chunks else None,
                )
            )


def _collect_zarr_datasets(group: Any, datasets: list[DatasetInfo], prefix: str = "") -> None:
    for name, value in group.items():
        key = f"{prefix}/{name}" if prefix else name
        if hasattr(value, "items"):
            _collect_zarr_datasets(value, datasets, key)
        else:
            dtype = value.dtype
            datasets.append(
                DatasetInfo(
                    name=key,
                    shape=tuple(int(dim) for dim in value.shape),
                    dtype=str(dtype),
                    nbytes=int(value.size * dtype.itemsize),
                    chunks=tuple(int(dim) for dim in value.chunks) if value.chunks else None,
                )
            )


def _info_from_datasets(
    path: Path,
    format_name: str,
    datasets: list[DatasetInfo],
    attrs: dict[str, Any],
    metadata_present: bool,
) -> PosteriorStorageInfo:
    datasets = sorted(datasets, key=lambda dataset: dataset.name)
    beta = next((dataset for dataset in datasets if dataset.name == "Beta"), None)
    n_chains = beta.shape[0] if beta is not None and len(beta.shape) >= 1 else None
    n_draws = beta.shape[1] if beta is not None and len(beta.shape) >= 2 else None
    return PosteriorStorageInfo(
        path=path,
        format=format_name,
        datasets=datasets,
        metadata_present=metadata_present,
        n_chains=n_chains,
        n_draws=n_draws,
        total_nbytes=sum(dataset.nbytes for dataset in datasets),
        attrs=attrs,
    )


def _json_safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        try:
            json.dumps(value)
        except TypeError:
            value = str(value)
        safe[str(key)] = value
    return safe
