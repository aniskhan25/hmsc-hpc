"""Generate a deterministic Whittaker plant held-out-site project."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


SOURCE = Path("examples/projects/whittaker_plants_hmsc_book")
DEFAULT_OUTPUT = Path("examples/projects/whittaker_plants_holdout_validation")


def generate_project(source: Path, output: Path, n_test: int = 12) -> None:
    data = source / "data"
    Y = pd.read_csv(data / "Y_presence.csv", index_col=0)
    X = pd.read_csv(data / "X.csv", index_col=0)
    study = pd.read_csv(data / "study_design_site.csv", index_col=0)
    traits = pd.read_csv(data / "traits.csv", index_col=0)
    phylo = pd.read_csv(data / "phylo_cov.csv", index_col=0)
    if not Y.index.equals(X.index) or not Y.index.equals(study.index):
        raise ValueError("response, covariates, and study design must have aligned sites")
    if not Y.columns.equals(traits.index) or not Y.columns.equals(phylo.index):
        raise ValueError("traits and phylogeny must follow response species order")
    if not phylo.index.equals(phylo.columns):
        raise ValueError("phylogenetic covariance rows and columns must match")

    species_totals = Y.sum(axis=0)
    singleton_species = species_totals == 1
    critical_sites = Y.loc[:, singleton_species].sum(axis=1) > 0
    candidates = X.loc[~critical_sites].sort_values("TMG")
    if n_test < 2 or n_test > len(candidates):
        raise ValueError(f"n_test must be between 2 and {len(candidates)}")
    targets = np.linspace(0, len(candidates) - 1, n_test)
    selected = []
    heldout_counts = pd.Series(0, index=Y.columns, dtype=int)
    for target in targets:
        positions = sorted(
            range(len(candidates)),
            key=lambda position: (abs(position - target), position),
        )
        for position in positions:
            site = candidates.index[position]
            if site in selected:
                continue
            candidate_counts = heldout_counts + Y.loc[site]
            if (candidate_counts < species_totals).all():
                selected.append(site)
                heldout_counts = candidate_counts
                break
        else:
            raise ValueError("unable to select hold-out sites while preserving every species")
    test_index = pd.Index(selected)
    train_index = Y.index[~Y.index.isin(test_index)]
    train_Y = Y.loc[train_index]
    if not (train_Y.sum(axis=0) > 0).all():
        raise ValueError("every species must remain present in training data")

    tables = {
        "data/train/Y.csv": train_Y,
        "data/train/X.csv": X.loc[train_index],
        "data/train/study_design.csv": study.loc[train_index],
        "data/test/Y.csv": Y.loc[test_index],
        "data/test/X.csv": X.loc[test_index],
        "data/test/study_design.csv": study.loc[test_index],
        "data/split.csv": pd.DataFrame(
            {"split": np.where(Y.index.isin(test_index), "test", "train")},
            index=Y.index,
        ),
    }
    for relative, table in tables.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path)
    for filename in ["traits.csv", "phylo_cov.csv", "taxonomy.csv"]:
        shutil.copy2(data / filename, output / "data" / filename)
    _write_configs(output)


def _write_configs(output: Path) -> None:
    fixed = [
        "response: data/train/Y.csv",
        "covariates: data/train/X.csv",
        "formula:",
        '  X: "~ TMG"',
        "distribution: probit",
        "traits: data/traits.csv",
        'trait_formula: "~ CN"',
        "phylo_cov: data/phylo_cov.csv",
        "chains: 2",
        "samples: 1000",
        "transient: 500",
        "thin: 10",
    ]
    (output / "model_fixed.yaml").write_text("\n".join(fixed) + "\n", encoding="utf-8")
    iid = [
        "response: data/train/Y.csv",
        "covariates: data/train/X.csv",
        "formula:",
        '  X: "~ TMG"',
        "distribution: probit",
        "study_design: data/train/study_design.csv",
        "random_levels:",
        "  plot:",
        "    column: plot",
        "    type: iid",
        "    nf: 1",
        "chains: 2",
        "samples: 1000",
        "transient: 500",
        "thin: 10",
    ]
    (output / "model_iid.yaml").write_text("\n".join(iid) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-sites", type=int, default=12)
    args = parser.parse_args()
    generate_project(args.source, args.output, args.test_sites)
    print(args.output)


if __name__ == "__main__":
    main()
