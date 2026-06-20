"""Generate replicated spatial hold-out projects and an array-task manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.simulate import apply_spatial_holdout_group_order, simulate_spatial_holdout_data


DEFAULT_OUTPUT = Path("examples/projects/replicated_spatial_holdout_validation")
TEMPLATE_PROJECT = ROOT / "examples/projects/simulated_spatial_holdout_validation"
DATA_PATHS = {
    "train_Y": "data/train/Y.csv",
    "train_X": "data/train/X.csv",
    "train_study_design": "data/train/study_design.csv",
    "test_Y": "data/test/Y.csv",
    "test_X": "data/test/X.csv",
    "test_study_design": "data/test/study_design.csv",
    "test_coords": "data/test/coords.csv",
    "truth_linear_predictor": "data/test/truth_linear_predictor.csv",
    "truth_beta": "data/truth_beta.csv",
    "truth_lambda": "data/truth_lambda.csv",
    "split": "data/split.csv",
}
MODELS = ["fixed", "spatial_full", "spatial_gpp", "spatial_nngp"]
ORDERINGS = ["canonical", "reverse", "random"]


def generate_projects(output: Path, seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    task_id = 0
    for seed in seeds:
        base = simulate_spatial_holdout_data(seed=seed)
        for ordering in ORDERINGS:
            project = output / f"seed_{seed}" / ordering
            ordered = apply_spatial_holdout_group_order(base, ordering=ordering, seed=seed + 10000)
            _write_project(project, ordered)
            task_models = MODELS if ordering == "canonical" else ["spatial_nngp"]
            for model in task_models:
                rows.append(
                    {
                        "task_id": task_id,
                        "seed": seed,
                        "ordering": ordering,
                        "model": model,
                        "project": str(project.resolve()),
                    }
                )
                task_id += 1
    manifest = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "tasks.csv", index=False)
    return manifest


def _write_project(project: Path, data: dict[str, pd.DataFrame]) -> None:
    for key, relative in DATA_PATHS.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data[key].to_csv(path)
    for model in MODELS:
        shutil.copy2(TEMPLATE_PROJECT / f"model_{model}.yaml", project / f"model_{model}.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[321, 654, 987])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    manifest = generate_projects(args.output, args.seeds)
    print(args.output)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
