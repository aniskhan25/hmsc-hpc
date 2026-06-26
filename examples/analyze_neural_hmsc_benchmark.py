"""Analyze Neural-HMSC posterior output against an MCMC reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.benchmark import compare_beta_posterior_files, compare_gamma_posterior_files, write_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", choices=["Beta", "Gamma"], default="Beta")
    parser.add_argument("--neural-posterior", required=True, help="Neural posterior .h5 path")
    parser.add_argument("--mcmc-posterior", required=True, help="MCMC reference posterior .h5 path")
    parser.add_argument("--truth-beta", help="truth_beta.csv path")
    parser.add_argument("--truth-gamma", help="truth_gamma.csv path")
    parser.add_argument("--X", help="covariate CSV path for predictive metrics")
    parser.add_argument("--Y", help="response CSV path for predictive metrics")
    parser.add_argument("--formula", default="~ x1 + x2")
    parser.add_argument("--distribution", help="normal, probit, or poisson")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--neural-seconds", type=float)
    parser.add_argument("--mcmc-seconds", type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stem", default="neural_hmsc_mcmc_reference")
    args = parser.parse_args()

    if args.parameter == "Gamma":
        row = compare_gamma_posterior_files(
            neural_posterior=args.neural_posterior,
            mcmc_posterior=args.mcmc_posterior,
            truth_gamma=args.truth_gamma,
            dataset=args.dataset,
            distribution=args.distribution,
        )
    else:
        row = compare_beta_posterior_files(
            neural_posterior=args.neural_posterior,
            mcmc_posterior=args.mcmc_posterior,
            truth_beta=args.truth_beta,
            dataset=args.dataset,
            distribution=args.distribution,
            neural_seconds=args.neural_seconds,
            mcmc_seconds=args.mcmc_seconds,
            X=args.X,
            Y=args.Y,
            formula=args.formula,
        )
    paths = write_benchmark_report([row], args.output_dir, stem=args.stem)
    print(f"Wrote {paths.csv}")
    print(f"Wrote {paths.markdown}")


if __name__ == "__main__":
    main()
