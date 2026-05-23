"""Command line helpers for pyhmsc."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyhmsc.compiler import compile_hmsc_model
from pyhmsc.config import model_from_config
from pyhmsc.data import read_table
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

    predict_parser = subparsers.add_parser("predict", help="predict from a posterior and covariate table")
    predict_parser.add_argument("posterior")
    predict_parser.add_argument("--X", required=True)
    predict_parser.add_argument("--formula", required=True)
    predict_parser.add_argument("--distribution", default="poisson")
    predict_parser.add_argument("--random-effects", choices=["none", "known", "marginal"], default="none")
    predict_parser.add_argument("--unseen-groups", choices=["error", "zero", "sample", "nearest"], default="error")
    predict_parser.add_argument("--output")

    diagnostics_parser = subparsers.add_parser("diagnostics", help="compute basic diagnostics")
    diagnostics_parser.add_argument("posterior")
    diagnostics_parser.add_argument("--param", default="Beta")

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
        print(fit.summary(args.param).to_string(index=False))
    elif args.command == "predict":
        from pyhmsc.model import HmscModel

        X = read_table(args.X)
        dummy_y = X.iloc[:, :1].copy()
        dummy_y.columns = ["species_0"]
        fit = HmscFit.from_file(
            args.posterior,
            model=HmscModel(Y=dummy_y, X=X, x_formula=args.formula, distr=args.distribution),
        )
        pred = fit.predict(X, random_effects=args.random_effects, unseen_groups=args.unseen_groups)
        if args.output:
            pred.to_csv(args.output)
        else:
            print(pred.to_string())
    elif args.command == "diagnostics":
        fit = HmscFit.from_file(args.posterior)
        print("rhat")
        print(fit.rhat(args.param))
        print("ess")
        print(fit.ess(args.param))
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
