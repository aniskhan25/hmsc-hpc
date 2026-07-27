#!/usr/bin/env python3
"""Run a qualified variable-shape probit checkpoint by stable identifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc import load_variable_shape_baseline  # noqa: E402
from pyhmsc.neural import simulate_fixed_effect_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--n-sites", type=int, default=30)
    parser.add_argument("--n-species", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    engine = load_variable_shape_baseline(args.registry_root)
    dataset = simulate_fixed_effect_dataset(
        n_sites=args.n_sites,
        n_species=args.n_species,
        distribution="probit",
        seed=20260730,
    )
    compatibility = engine.check_compatibility(dataset)
    fit = engine.infer(
        dataset,
        draws=64,
        chains=1,
        seed=20260730,
        output=output / "variable_shape_posterior.h5",
    )
    result = {
        "baseline_id": "neural_hmsc_variable_probit_v1",
        "shape_range": engine.shape_range,
        "input_dimensions": compatibility["dimensions"],
        "coefficient_calibration": engine.calibration.method,
        "beta_samples_shape": list(fit.beta_samples().shape),
        "beta_mean_shape": list(fit.beta_mean().shape),
        "mcmc_role": "statistical_reference_only",
    }
    (output / "variable_shape_example.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
