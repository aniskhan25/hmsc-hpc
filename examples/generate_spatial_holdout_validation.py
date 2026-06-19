"""Generate the deterministic spatial held-out prediction project."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.simulate import simulate_spatial_holdout_data


DEFAULT_OUTPUT = Path("examples/projects/simulated_spatial_holdout_validation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = simulate_spatial_holdout_data(
        n_sites=100,
        n_species=6,
        holdout_stride=5,
        seed=321,
    )
    paths = {
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
    for key, relative in paths.items():
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data[key].to_csv(path)
    print(args.output)


if __name__ == "__main__":
    main()
