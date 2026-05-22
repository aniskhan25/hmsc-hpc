"""Raw data loading helpers for Python-native workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def read_table(path: str | Path, index_col: int | str | None = 0):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, index_col=index_col)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", index_col=index_col)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        try:
            return sparse.load_npz(path)
        except ValueError:
            return np.load(path)
    raise ValueError(f"Unsupported raw data format: {path.suffix}")
