"""Generate a spatially blocked real-data plant hold-out project."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


SOURCE = Path("examples/projects/big_spatial_plants_validation")
DEFAULT_OUTPUT = Path("examples/projects/big_spatial_plants_holdout_validation")
FORMULA = "~ Hillshading270_40 + HA_All_rivers_normalised + Thorium_mosaic_GWR2 + Max_temp_smooth"


def generate_project(source: Path, output: Path, grid_size: int = 8, holdout_modulus: int = 5) -> None:
    if grid_size < 2 or holdout_modulus < 2:
        raise ValueError("grid_size and holdout_modulus must be at least 2")
    data = source / "data"
    Y = pd.read_csv(data / "Y_presence.csv", index_col=0)
    X = pd.read_csv(data / "X.csv", index_col=0)
    design = pd.read_csv(data / "study_design.csv", index_col=0)
    if not Y.index.equals(X.index) or not Y.index.equals(design.index):
        raise ValueError("source response, covariates, and study design are not aligned")

    x_bin = np.minimum((design["xcoord"] * grid_size).astype(int), grid_size - 1)
    y_bin = np.minimum((design["ycoord"] * grid_size).astype(int), grid_size - 1)
    block = x_bin + grid_size * y_bin
    holdout = (x_bin + y_bin) % holdout_modulus == 0
    if not holdout.any() or holdout.all():
        raise ValueError("spatial block split must contain train and test sites")
    train_index = Y.index[~holdout]
    test_index = Y.index[holdout]
    train_Y = Y.loc[train_index]
    if not ((train_Y.sum() > 0) & (train_Y.sum() < len(train_Y))).all():
        raise ValueError("every species must contain presences and absences in training data")

    tables = {
        "data/train/Y.csv": train_Y,
        "data/train/X.csv": X.loc[train_index],
        "data/train/study_design.csv": design.loc[train_index],
        "data/test/Y.csv": Y.loc[test_index],
        "data/test/X.csv": X.loc[test_index],
        "data/test/study_design.csv": design.loc[test_index, ["site"]],
        "data/test/coords.csv": design.loc[test_index, ["xcoord", "ycoord"]],
        "data/split.csv": pd.DataFrame(
            {
                "split": np.where(holdout, "test", "train"),
                "block": block,
                "x_bin": x_bin,
                "y_bin": y_bin,
            },
            index=Y.index,
        ),
    }
    for relative, table in tables.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path)
    shutil.copy2(data / "taxonomy.csv", output / "data/taxonomy.csv")
    _write_configs(output)


def _write_configs(output: Path) -> None:
    common = "\n".join(
        [
            "response: data/train/Y.csv",
            "covariates: data/train/X.csv",
            f'formula:\n  X: "{FORMULA}"',
            "distribution: probit",
            "chains: 2",
            "samples: 1000",
            "transient: 500",
            "thin: 10",
        ]
    )
    (output / "model_fixed.yaml").write_text(common + "\n", encoding="utf-8")
    spatial = {
        "spatial_full": [],
        "spatial_gpp": ["    n_knots: 20"],
        "spatial_nngp": ["    n_neighbors: 15"],
    }
    for method, extra in spatial.items():
        lines = [
            common,
            "study_design: data/train/study_design.csv",
            "random_levels:",
            "  site:",
            "    column: site",
            f"    type: {method}",
            '    coords: ["xcoord", "ycoord"]',
            "    nf: 2",
            *extra,
        ]
        (output / f"model_{method}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--holdout-modulus", type=int, default=5)
    args = parser.parse_args()
    generate_project(args.source, args.output, args.grid_size, args.holdout_modulus)
    print(args.output)


if __name__ == "__main__":
    main()
