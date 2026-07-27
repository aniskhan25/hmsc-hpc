#!/usr/bin/env python3
"""Run calibrated coefficient inference and predictive deployment from v0.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc import compile_hmsc_model, load_neural_hmsc_release  # noqa: E402
from pyhmsc.neural.release import NEURAL_HMSC_RELEASE_ID  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--release-id", default=NEURAL_HMSC_RELEASE_ID)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    release = load_neural_hmsc_release(args.registry_root, release_id=args.release_id)
    engine = release.load_checkpoint(args.seed)

    rng = np.random.default_rng(20260721)
    tmg = np.linspace(-1.5, 1.5, engine.model.n_sites)
    X = pd.DataFrame({"TMG": tmg})
    Y = pd.DataFrame(
        rng.binomial(1, 0.2, size=(engine.model.n_sites, engine.model.n_species)),
        columns=engine.species_names,
    )
    compiled = compile_hmsc_model(
        Y=Y,
        X=X,
        formula=engine.formula,
        distr="probit",
        chains=1,
        output=output / "compiled",
    )
    fit = engine.infer(
        compiled.init_json,
        draws=32,
        chains=1,
        seed=args.seed,
        output=output / "neural_posterior.h5",
    )

    ensemble = release.load_predictive_ensemble(dataset="whittaker")
    probability = ensemble.predict_mean(X)
    result = {
        "release_id": release.release_id,
        "release_content_sha256": release.manifest["content_sha256"],
        "checkpoint_seed": args.seed,
        "coefficient_calibration": engine.coefficient_calibration.method,
        "beta_samples_shape": list(fit.beta_samples().shape),
        "beta_mean_shape": list(fit.beta_mean().shape),
        "predictive_policy": ensemble.calibration_role,
        "predictive_member_seeds": list(ensemble.seeds),
        "predictive_probability_shape": list(probability.shape),
        "predictive_probability_range": [
            float(probability.to_numpy().min()),
            float(probability.to_numpy().max()),
        ],
        "response_semantics": "predictive_only",
        "mcmc_role": release.manifest["qualified_python_mcmc_role"],
    }
    result_path = output / "neural_hmsc_v0_1_example.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
