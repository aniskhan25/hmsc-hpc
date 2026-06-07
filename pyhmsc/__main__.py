"""Command line helpers for pyhmsc."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyhmsc.compiler import compile_hmsc_model
from pyhmsc.config import model_from_config
from pyhmsc.data import read_table
from pyhmsc.merge import inspect_chain_directory, merge_hdf5_posteriors
from pyhmsc.posterior import HmscFit
from pyhmsc.runner import run_gibbs_sampler
from pyhmsc.validation import validate_compiled_native_model, validate_fit


def main() -> None:
    parser = argparse.ArgumentParser(prog="pyhmsc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="compile raw data to init.json + HDF5")
    compile_parser.add_argument("config", nargs="?", help="YAML/JSON model config path")
    compile_parser.add_argument("--Y", help="response table path")
    compile_parser.add_argument("--X", help="covariate table path")
    compile_parser.add_argument("--formula", help="one-sided X formula")
    compile_parser.add_argument("--distribution", default="poisson")
    compile_parser.add_argument("--chains", type=int, default=4)
    compile_parser.add_argument("--output", default="run")

    sample_parser = subparsers.add_parser("sample", help="run hmsc-hpc sampler on a compiled init file")
    sample_parser.add_argument("input", help="compiled init.json or compatibility RDS input")
    sample_parser.add_argument("--output", default="posterior.h5")
    sample_parser.add_argument("--samples", type=int, default=100)
    sample_parser.add_argument("--transient", type=int, default=100)
    sample_parser.add_argument("--thin", type=int, default=1)
    sample_parser.add_argument("--verbose", type=int, default=100)
    sample_parser.add_argument("--chains", type=int, nargs="*")

    validate_init_parser = subparsers.add_parser(
        "validate-init",
        help="validate a Python-native compiled init.json before sampling",
    )
    validate_init_parser.add_argument("input", help="compiled init.json")
    validate_init_parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any validation check fails",
    )

    summarize_parser = subparsers.add_parser("summarize", help="summarize a posterior file")
    summarize_parser.add_argument("posterior", help="posterior .h5, .json, or .rds file")
    summarize_parser.add_argument("--param", default="Beta")
    summarize_parser.add_argument("--level", type=float, default=0.95)
    summarize_parser.add_argument("--random-level", type=int, default=0)
    summarize_parser.add_argument("--x-index", type=int)

    merge_parser = subparsers.add_parser("merge", help="merge HDF5 posterior shards")
    merge_parser.add_argument("inputs", nargs="+", help="input posterior .h5 files")
    merge_parser.add_argument("--output", required=True, help="merged posterior .h5 path")
    merge_parser.add_argument("--expected-chains", type=int, nargs="*", help="expected chain ids")

    chain_status_parser = subparsers.add_parser("chain-status", help="inspect HDF5 chain shards")
    chain_status_parser.add_argument("directory", help="directory containing posterior_chain_<id>.h5 files")
    chain_status_parser.add_argument("--expected-chains", type=int, nargs="+", required=True)
    chain_status_parser.add_argument("--expected-draws", type=int)
    chain_status_parser.add_argument("--run-name", help="LUMI RUN_NAME to include in rerun commands")
    chain_status_parser.add_argument("--strict", action="store_true", help="exit non-zero if any chain failed")

    predict_parser = subparsers.add_parser("predict", help="predict from a posterior and covariate table")
    predict_parser.add_argument("posterior")
    predict_parser.add_argument("--X", required=True)
    predict_parser.add_argument("--formula")
    predict_parser.add_argument("--distribution")
    predict_parser.add_argument("--random-effects", choices=["none", "known", "marginal"], default="none")
    predict_parser.add_argument("--unseen-groups", choices=["error", "zero", "sample", "nearest"], default="error")
    predict_parser.add_argument("--output")

    ppc_parser = subparsers.add_parser("ppc", help="run posterior predictive checks")
    ppc_parser.add_argument("posterior")
    ppc_parser.add_argument("--X", required=True)
    ppc_parser.add_argument("--Y", required=True)
    ppc_parser.add_argument("--formula")
    ppc_parser.add_argument("--distribution")
    ppc_parser.add_argument("--random-effects", choices=["none", "known", "marginal"], default="none")
    ppc_parser.add_argument("--unseen-groups", choices=["error", "zero", "sample", "nearest"], default="error")
    ppc_parser.add_argument("--level", type=float, default=0.95)
    ppc_parser.add_argument("--seed", type=int)
    ppc_parser.add_argument("--scope", choices=["species", "site-richness"], default="species")
    ppc_parser.add_argument("--output")

    diagnostics_parser = subparsers.add_parser("diagnostics", help="compute basic diagnostics")
    diagnostics_parser.add_argument("posterior")
    diagnostics_parser.add_argument("--param", default="Beta")
    diagnostics_parser.add_argument("--random-level", type=int, default=0)
    diagnostics_parser.add_argument("--x-index", type=int)
    diagnostics_parser.add_argument("--rhat-threshold", type=float, default=1.01)
    diagnostics_parser.add_argument("--ess-threshold", type=float, default=400.0)
    diagnostics_parser.add_argument("--output")

    associations_parser = subparsers.add_parser(
        "associations",
        help="summarize residual species associations from Lambda samples",
    )
    associations_parser.add_argument("posterior")
    associations_parser.add_argument("--random-level", type=int, default=0)
    associations_parser.add_argument("--level", type=float, default=0.95)
    associations_parser.add_argument("--x-index", type=int)
    associations_parser.add_argument(
        "--covariance",
        action="store_true",
        help="summarize covariance instead of correlation",
    )
    associations_parser.add_argument(
        "--matrix",
        action="store_true",
        help="write/print the mean association matrix instead of a pair table",
    )
    associations_parser.add_argument("--include-self", action="store_true")
    associations_parser.add_argument("--output")

    validate_parser = subparsers.add_parser("validate", help="run simple validation checks")
    validate_parser.add_argument("posterior")
    validate_parser.add_argument("--X")
    validate_parser.add_argument("--Y")
    validate_parser.add_argument("--formula")
    validate_parser.add_argument("--distribution", default="poisson")

    args = parser.parse_args()
    if args.command == "compile":
        if args.config:
            model, config = model_from_config(args.config)
            compiled = model.compile(Path(args.output), chains=int(config.get("chains", args.chains)))
        else:
            missing = [name for name in ("Y", "X", "formula") if getattr(args, name) is None]
            if missing:
                parser.error(f"compile requires CONFIG or --Y/--X/--formula; missing {missing}")
            compiled = compile_hmsc_model(
                Y=read_table(args.Y),
                X=read_table(args.X),
                formula=args.formula,
                distr=args.distribution,
                chains=args.chains,
                output=Path(args.output),
            )
        print(compiled.init_json)
    elif args.command == "sample":
        if str(args.input).lower().endswith(".json"):
            try:
                results = validate_compiled_native_model(args.input)
            except ValueError as exc:
                parser.error(str(exc))
            failed = [result for result in results if not result.passed]
            if failed:
                details = "; ".join(f"{result.name}: {result.details}" for result in failed)
                raise SystemExit(f"Compiled native model is not sampler-ready: {details}")
        run_gibbs_sampler(
            init_file=args.input,
            output_file=args.output,
            samples=args.samples,
            transient=args.transient,
            thin=args.thin,
            verbose=args.verbose,
            chains=args.chains,
        )
        print(args.output)
    elif args.command == "validate-init":
        try:
            results = validate_compiled_native_model(args.input)
        except ValueError as exc:
            parser.error(str(exc))
        failed = False
        for result in results:
            status = "passed" if result.passed else "failed"
            print(f"{result.name}: {status} {result.details}")
            failed = failed or not result.passed
        if failed and args.strict:
            raise SystemExit(1)
    elif args.command == "summarize":
        fit = HmscFit.from_file(args.posterior)
        if args.param == "Eta":
            summary = fit.eta_summary(level=args.random_level, cred_level=args.level)
        elif args.param == "Lambda":
            summary = fit.lambda_summary(
                level=args.random_level,
                cred_level=args.level,
                x_index=args.x_index,
            )
        else:
            summary = fit.summary(args.param, level=args.level)
        print(summary.to_string(index=False))
    elif args.command == "merge":
        output = merge_hdf5_posteriors(args.inputs, args.output, expected_chains=args.expected_chains)
        print(output)
    elif args.command == "chain-status":
        statuses = inspect_chain_directory(
            args.directory,
            expected_chains=args.expected_chains,
            expected_draws=args.expected_draws,
        )
        failed = [status for status in statuses if status.status != "passed"]
        print("chain status")
        for status in statuses:
            print(f"{status.chain} {status.status} {status.path} {status.message}")
        if failed:
            failed_ids = " ".join(str(status.chain) for status in failed)
            print("rerun")
            if args.run_name:
                print(f"RUN_NAME={args.run_name} sbatch --array={','.join(str(status.chain) for status in failed)} docs/lumi_python_native_array_sbatch.sh")
            else:
                print(f"sbatch --array={','.join(str(status.chain) for status in failed)} docs/lumi_python_native_array_sbatch.sh")
            print(f"failed_chains {failed_ids}")
            if args.strict:
                raise SystemExit(1)
    elif args.command == "predict":
        from pyhmsc.model import HmscModel

        X = read_table(args.X)
        fit = HmscFit.from_file(args.posterior)
        if args.formula:
            dummy_y = X.iloc[:, :1].copy()
            dummy_y.columns = ["species_0"]
            fit.model = HmscModel(
                Y=dummy_y,
                X=X,
                x_formula=args.formula,
                distr=args.distribution or fit._distribution(),
            )
        elif fit._x_formula() is None:
            parser.error("predict requires --formula unless posterior metadata includes formula.X")
        pred = fit.predict(X, random_effects=args.random_effects, unseen_groups=args.unseen_groups)
        if args.output:
            pred.to_csv(args.output)
        else:
            print(pred.to_string())
    elif args.command == "ppc":
        from pyhmsc.model import HmscModel

        X = read_table(args.X)
        Y = read_table(args.Y)
        fit = HmscFit.from_file(args.posterior)
        if args.formula:
            fit.model = HmscModel(
                Y=Y,
                X=X,
                x_formula=args.formula,
                distr=args.distribution or fit._distribution(),
            )
        elif fit._x_formula() is None:
            parser.error("ppc requires --formula unless posterior metadata includes formula.X")
        if args.scope == "site-richness":
            summary = fit.richness_ppc_summary(
                Y=Y,
                X=X,
                level=args.level,
                random_effects=args.random_effects,
                unseen_groups=args.unseen_groups,
                rng_seed=args.seed,
            )
        else:
            summary = fit.ppc_summary(
                Y=Y,
                X=X,
                level=args.level,
                random_effects=args.random_effects,
                unseen_groups=args.unseen_groups,
                rng_seed=args.seed,
            )
        if args.output:
            summary.to_csv(args.output, index=False)
        else:
            print(summary.to_string(index=False))
    elif args.command == "diagnostics":
        fit = HmscFit.from_file(args.posterior)
        overview = fit.diagnostics_overview(
            args.param,
            rhat_threshold=args.rhat_threshold,
            ess_threshold=args.ess_threshold,
            level=args.random_level,
            x_index=args.x_index,
        )
        diagnostics = fit.diagnostics(args.param, level=args.random_level, x_index=args.x_index)
        lines = [
            "diagnostics",
            *(f"{key}: {value}" for key, value in overview.items()),
            "",
            diagnostics.to_string(index=False),
        ]
        text = "\n".join(lines) + "\n"
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    elif args.command == "associations":
        fit = HmscFit.from_file(args.posterior)
        if args.matrix:
            summary = fit.species_associations(
                level=args.random_level,
                correlation=not args.covariance,
                x_index=args.x_index,
            )
            if args.output:
                summary.to_csv(args.output)
            else:
                print(summary.to_string())
        else:
            summary = fit.species_association_summary(
                level=args.random_level,
                cred_level=args.level,
                correlation=not args.covariance,
                x_index=args.x_index,
                include_self=args.include_self,
            )
            if args.output:
                summary.to_csv(args.output, index=False)
            else:
                print(summary.to_string(index=False))
    elif args.command == "validate":
        from pyhmsc.model import HmscModel

        X = read_table(args.X) if args.X else None
        Y = read_table(args.Y) if args.Y else None
        model = None
        if X is not None and args.formula:
            dummy_y = Y if Y is not None else X.iloc[:, :1].rename(columns={X.columns[0]: "species_0"})
            model = HmscModel(Y=dummy_y, X=X, x_formula=args.formula, distr=args.distribution)
        fit = HmscFit.from_file(args.posterior, model=model)
        results = validate_fit(fit, X=X, Y=Y)
        for result in results:
            print(f"{result.name}: {'passed' if result.passed else 'failed'}")


if __name__ == "__main__":
    main()
