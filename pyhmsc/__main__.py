"""Command line helpers for pyhmsc."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyhmsc.compiler import compile_hmsc_model
from pyhmsc.config import model_from_config
from pyhmsc.data import read_table


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


if __name__ == "__main__":
    main()
