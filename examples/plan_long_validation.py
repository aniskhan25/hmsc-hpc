"""Plan targeted longer or 4-chain validation from posterior diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.posterior import HmscFit


DEFAULT_PARAMS = ["Beta", "Gamma", "Associations"]


def build_plan(
    posteriors: list[Path],
    params: list[str] | None = None,
    rhat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
    include_latent: bool = False,
) -> pd.DataFrame:
    rows = []
    params = params or DEFAULT_PARAMS
    for posterior in posteriors:
        fit = HmscFit.from_file(posterior)
        for param in params:
            rows.extend(
                _diagnostic_rows(
                    fit=fit,
                    posterior=posterior,
                    param=param,
                    rhat_threshold=rhat_threshold,
                    ess_threshold=ess_threshold,
                    align=False,
                )
            )
        if include_latent:
            for param in ["Eta", "Lambda"]:
                rows.extend(
                    _diagnostic_rows(
                        fit=fit,
                        posterior=posterior,
                        param=param,
                        rhat_threshold=rhat_threshold,
                        ess_threshold=ess_threshold,
                        align=True,
                    )
                )
    return pd.DataFrame(rows)


def _diagnostic_rows(
    fit: HmscFit,
    posterior: Path,
    param: str,
    rhat_threshold: float,
    ess_threshold: float,
    align: bool,
) -> list[dict[str, Any]]:
    try:
        overview = fit.diagnostics_overview(
            param,
            rhat_threshold=rhat_threshold,
            ess_threshold=ess_threshold,
            align=align,
        )
    except (KeyError, ValueError):
        return []
    needed = overview["n_rhat_flagged"] > 0 or overview["n_ess_flagged"] > 0
    return [
        {
            "posterior": str(posterior),
            "param": param,
            "aligned": bool(align),
            "needs_longer_validation": bool(needed),
            "recommendation": _recommendation(param, needed, align),
            "n_parameters": overview["n_parameters"],
            "rhat_max": overview["rhat_max"],
            "rhat_median": overview["rhat_median"],
            "ess_min": overview["ess_min"],
            "ess_median": overview["ess_median"],
            "n_rhat_flagged": overview["n_rhat_flagged"],
            "n_ess_flagged": overview["n_ess_flagged"],
            "rhat_threshold": overview["rhat_threshold"],
            "ess_threshold": overview["ess_threshold"],
        }
    ]


def _recommendation(param: str, needed: bool, align: bool) -> str:
    if not needed:
        return "no longer run needed for this diagnostic"
    if param == "Associations":
        return "run targeted 4-chain association validation"
    if param in {"Beta", "Gamma"}:
        return f"run longer 4-chain validation if {param} inference is needed"
    if param in {"Eta", "Lambda"} and align:
        return "inspect aligned factors; prefer Associations for residual association inference"
    return "run targeted longer validation only if this parameter is an inference target"


def _report(plan: pd.DataFrame) -> str:
    if plan.empty:
        return "No supported diagnostics found for the requested posterior(s).\n"
    lines = [
        "# Targeted Longer Validation Plan",
        "",
        plan.to_string(index=False),
        "",
    ]
    needed = plan[plan["needs_longer_validation"]]
    if needed.empty:
        lines.append("No longer or 4-chain validation is recommended by the configured thresholds.")
    else:
        lines.append("Recommended targeted follow-up:")
        for _, row in needed.iterrows():
            lines.append(
                f"- {row['posterior']} {row['param']}: {row['recommendation']} "
                f"(R-hat flags {row['n_rhat_flagged']}, ESS flags {row['n_ess_flagged']})"
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("posteriors", nargs="+", type=Path)
    parser.add_argument("--param", action="append", dest="params")
    parser.add_argument("--include-latent", action="store_true")
    parser.add_argument("--rhat-threshold", type=float, default=1.01)
    parser.add_argument("--ess-threshold", type=float, default=400.0)
    parser.add_argument("--output")
    parser.add_argument("--csv-output")
    args = parser.parse_args()

    plan = build_plan(
        args.posteriors,
        params=args.params,
        rhat_threshold=args.rhat_threshold,
        ess_threshold=args.ess_threshold,
        include_latent=args.include_latent,
    )
    text = _report(plan)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.csv_output:
        plan.to_csv(args.csv_output, index=False)


if __name__ == "__main__":
    main()
