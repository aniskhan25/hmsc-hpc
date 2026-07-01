"""Build a fixed-shape Big Spatial Plant transfer project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_MATRIX = Path("examples/big_spatial/data")
SOURCE_PROJECT = Path("examples/projects/big_spatial_plants_validation")
DEFAULT_OUTPUT = Path("examples/projects/neural_hmsc_big_spatial_transfer")
SOURCE_COVARIATE = "Max_temp_smooth"


def generate_project(
    source_matrix: Path,
    source_project: Path,
    output: Path,
    *,
    n_train: int = 40,
    n_species: int = 75,
) -> None:
    """Create a deterministic training/holdout projection for frozen inference."""
    project_data = source_project / "data"
    covariates = pd.read_csv(project_data / "X.csv", index_col=0)
    study = pd.read_csv(project_data / "study_design.csv", index_col=0)
    if not covariates.index.equals(study.index):
        raise ValueError("source project covariates and study design must be aligned")
    if SOURCE_COVARIATE not in covariates:
        raise ValueError(f"source project is missing {SOURCE_COVARIATE!r}")
    if n_train < 2 or n_train >= len(study):
        raise ValueError(
            "n_train must leave at least one training and one held-out site"
        )

    train_positions = _farthest_point_positions(
        study[["xcoord", "ycoord"]].to_numpy(dtype=float),
        n_train,
    )
    train_sites = study.index[train_positions]
    test_sites = study.index[~np.isin(np.arange(len(study)), train_positions)]
    source_positions = np.asarray(
        [_source_position(site) for site in study.index], dtype=int
    )

    taxonomy = pd.read_csv(source_matrix / "taxa_used_tree.csv")
    candidate_species = set(taxonomy["name"].astype(str))
    response = pd.read_csv(
        source_matrix / "Y.csv",
        usecols=lambda column: column in candidate_species,
    )
    if source_positions.max(initial=-1) >= len(response):
        raise ValueError(
            "source project site identifiers exceed the source response matrix"
        )
    response = response.iloc[source_positions].set_axis(study.index)
    response = response.gt(0).astype(np.int8)

    training_response = response.loc[train_sites]
    prevalence = training_response.mean(axis=0)
    eligible = prevalence[(prevalence > 0.0) & (prevalence < 1.0)]
    ranked = sorted(
        eligible.index, key=lambda name: (-float(eligible[name]), str(name))
    )
    if len(ranked) < n_species:
        raise ValueError(
            f"only {len(ranked)} species have presences and absences in training; "
            f"need {n_species}"
        )
    species = ranked[:n_species]
    train_Y = response.loc[train_sites, species]
    test_Y = response.loc[test_sites, species]

    projected_X = pd.DataFrame(
        {"TMG": covariates[SOURCE_COVARIATE].to_numpy(dtype=float)},
        index=covariates.index,
    )
    split = pd.DataFrame("test", index=study.index, columns=["split"])
    split.loc[train_sites, "split"] = "train"
    split["source_row"] = source_positions
    split["selection"] = np.where(
        split["split"] == "train", "spatial_maximin", "held_out"
    )

    selected_taxonomy = taxonomy.set_index("name").loc[species]
    tables = {
        "data/train/Y.csv": train_Y,
        "data/train/X.csv": projected_X.loc[train_sites],
        "data/train/study_design.csv": study.loc[train_sites],
        "data/test/Y.csv": test_Y,
        "data/test/X.csv": projected_X.loc[test_sites],
        "data/test/study_design.csv": study.loc[test_sites],
        "data/split.csv": split,
        "data/taxonomy.csv": selected_taxonomy,
    }
    for relative, table in tables.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path)

    metadata = {
        "dataset": "Big Spatial Plant community",
        "source_matrix": str(source_matrix),
        "source_project": str(source_project),
        "source_covariate": SOURCE_COVARIATE,
        "projected_covariate": "TMG",
        "site_selection": "coordinate maximin; starts nearest coordinate centroid",
        "species_selection": "training prevalence descending, species name tie-break",
        "holdout_used_for_selection": False,
        "training_sites": int(len(train_sites)),
        "heldout_sites": int(len(test_sites)),
        "species": int(len(species)),
    }
    (output / "projection_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "model_fixed.yaml").write_text(
        "\n".join(
            [
                "response: data/train/Y.csv",
                "covariates: data/train/X.csv",
                "formula:",
                '  X: "~ TMG"',
                "distribution: probit",
                "chains: 2",
                "samples: 1000",
                "transient: 500",
                "thin: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _farthest_point_positions(coords: np.ndarray, count: int) -> np.ndarray:
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("coordinates must be a finite n_sites x 2 array")
    center = coords.mean(axis=0)
    selected = [int(np.argmin(np.sum(np.square(coords - center), axis=1)))]
    minimum_distance = np.sum(np.square(coords - coords[selected[0]]), axis=1)
    while len(selected) < count:
        minimum_distance[selected] = -1.0
        next_position = int(np.argmax(minimum_distance))
        selected.append(next_position)
        distance = np.sum(np.square(coords - coords[next_position]), axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return np.asarray(selected, dtype=int)


def _source_position(site: object) -> int:
    prefix = "site_"
    text = str(site)
    if not text.startswith(prefix) or not text[len(prefix) :].isdigit():
        raise ValueError(
            f"source project site identifier has unsupported form: {text!r}"
        )
    return int(text[len(prefix) :])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-matrix", type=Path, default=SOURCE_MATRIX)
    parser.add_argument("--source-project", type=Path, default=SOURCE_PROJECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-sites", type=int, default=40)
    parser.add_argument("--species", type=int, default=75)
    args = parser.parse_args()
    generate_project(
        args.source_matrix,
        args.source_project,
        args.output,
        n_train=args.train_sites,
        n_species=args.species,
    )
    print(args.output)


if __name__ == "__main__":
    main()
