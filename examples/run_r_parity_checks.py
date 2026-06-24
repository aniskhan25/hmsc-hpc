"""Optional one-time R parity checks for Python-native compiled models.

This script intentionally is not part of the normal test suite dependency
chain. It requires R only when explicitly invoked and compares Python-native
compiled artifacts against base R formula and factor encodings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.config import model_from_config
from pyhmsc.serialization import read_compiled_model
from pyhmsc.validation import ValidationResult, validate_compiled_native_model


DEFAULT_CASES = [
    Path("tests/fixtures/fixed_effect/model.yaml"),
    Path("tests/fixtures/fixed_effect/model_traits_phylo.yaml"),
    Path("examples/projects/iid_random_intercept/model.yaml"),
]


@dataclass(frozen=True)
class RParityCase:
    config_path: Path
    output_name: str


def default_cases() -> list[RParityCase]:
    return [
        RParityCase(path, path.parent.name if path.name == "model.yaml" else path.stem)
        for path in DEFAULT_CASES
    ]


def run_parity_case(config_path: Path, workdir: Path, rscript: str = "Rscript") -> list[ValidationResult]:
    config_path = config_path.resolve()
    case_dir = workdir / _case_name(config_path)
    case_dir.mkdir(parents=True, exist_ok=True)

    model, config = model_from_config(config_path)
    chains = int(config.get("chains", 2))
    compiled = model.compile(case_dir / "compiled", chains=chains)
    metadata, arrays = read_compiled_model(compiled.init_json)

    r_out = case_dir / "r"
    r_out.mkdir(parents=True, exist_ok=True)
    script = case_dir / "parity.R"
    script.write_text(_r_script(config_path, config, r_out), encoding="utf-8")
    subprocess.run([rscript, str(script)], check=True, text=True)

    results = [
        ValidationResult(
            "native_compiled_validation",
            all(result.passed for result in validate_compiled_native_model(compiled.init_json)),
            {
                result.name: {"passed": result.passed, "details": result.details}
                for result in validate_compiled_native_model(compiled.init_json)
            },
        ),
        _compare_matrix("X_design", arrays["X"], r_out / "X_design.csv", metadata["names"]["covariates"]),
    ]
    if bool(metadata.get("capabilities", {}).get("traits")):
        results.append(
            _compare_matrix("trait_design", arrays["T"], r_out / "T_design.csv", metadata["names"]["traits"])
        )
    if bool(metadata.get("capabilities", {}).get("phylogeny")):
        results.append(_compare_matrix("phylo_cov", arrays["C"], r_out / "C_ordered.csv", metadata["names"]["species"]))
    if metadata.get("random_levels"):
        pi = arrays["Pi"]
        for idx, level in enumerate(metadata["random_levels"]):
            results.append(
                _compare_vector(
                    f"random_level_{level['name']}_codes",
                    pi[:, idx],
                    r_out / f"Pi_{idx}.csv",
                    expected_levels=level["levels"],
                )
            )
    return results


def _case_name(config_path: Path) -> str:
    parent = config_path.parent.name
    return parent if config_path.name == "model.yaml" else f"{parent}_{config_path.stem}"


def _compare_matrix(
    name: str,
    observed: np.ndarray,
    expected_path: Path,
    observed_names: list[str],
    atol: float = 1e-10,
) -> ValidationResult:
    expected = pd.read_csv(expected_path, index_col=0)
    expected_names = [_normalize_r_name(name) for name in expected.columns]
    observed = np.asarray(observed, dtype=float)
    expected_values = expected.to_numpy(dtype=float)
    shape_ok = observed.shape == expected_values.shape
    names_ok = observed_names == expected_names
    values_ok = shape_ok and bool(np.allclose(observed, expected_values, atol=atol, rtol=0.0))
    return ValidationResult(
        name,
        shape_ok and names_ok and values_ok,
        {
            "observed_shape": tuple(int(value) for value in observed.shape),
            "expected_shape": tuple(int(value) for value in expected_values.shape),
            "observed_names": observed_names,
            "expected_names": expected_names,
            "max_abs_diff": _max_abs_diff(observed, expected_values) if shape_ok else None,
        },
    )


def _compare_vector(
    name: str,
    observed: np.ndarray,
    expected_path: Path,
    expected_levels: list[str],
) -> ValidationResult:
    expected = pd.read_csv(expected_path, index_col=0)
    expected_codes = expected["code"].to_numpy(dtype=int)
    observed_codes = np.asarray(observed, dtype=int)
    observed_levels = sorted(str(value) for value in expected["level"].drop_duplicates())
    codes_ok = observed_codes.shape == expected_codes.shape and bool(np.array_equal(observed_codes, expected_codes))
    levels_ok = expected_levels == observed_levels
    return ValidationResult(
        name,
        codes_ok and levels_ok,
        {
            "observed_shape": tuple(int(value) for value in observed_codes.shape),
            "expected_shape": tuple(int(value) for value in expected_codes.shape),
            "observed_levels": expected_levels,
            "expected_levels": observed_levels,
        },
    )


def _max_abs_diff(observed: np.ndarray, expected: np.ndarray) -> float:
    if observed.size == 0 and expected.size == 0:
        return 0.0
    return float(np.max(np.abs(observed - expected)))


def _normalize_r_name(name: str) -> str:
    return "Intercept" if name == "(Intercept)" else name


def _r_script(config_path: Path, config: dict[str, Any], output: Path) -> str:
    base = config_path.parent.resolve()
    formula = config["formula"].get("X") if isinstance(config["formula"], dict) else config["formula"]
    lines = [
        "options(stringsAsFactors = FALSE)",
        f"base <- {_r_string(str(base))}",
        f"out <- {_r_string(str(output.resolve()))}",
        "dir.create(out, recursive = TRUE, showWarnings = FALSE)",
        f"X <- read.csv(file.path(base, {_r_string(config['covariates'])}), row.names = 1, check.names = FALSE)",
        f"X_design <- model.matrix(as.formula({_r_string(formula)}), data = X)",
        'write.csv(X_design, file.path(out, "X_design.csv"))',
    ]
    if config.get("traits"):
        trait_formula = config.get("trait_formula") or "~ ."
        lines.extend(
            [
                f"Y <- read.csv(file.path(base, {_r_string(config['response'])}), row.names = 1, check.names = FALSE)",
                f"Tr <- read.csv(file.path(base, {_r_string(config['traits'])}), row.names = 1, check.names = FALSE)",
                "Tr <- Tr[colnames(Y), , drop = FALSE]",
                f"T_design <- model.matrix(as.formula({_r_string(trait_formula)}), data = Tr)",
                'write.csv(T_design, file.path(out, "T_design.csv"))',
            ]
        )
    if config.get("phylo_cov"):
        lines.extend(
            [
                "if (!exists('Y')) Y <- read.csv(file.path(base, "
                f"{_r_string(config['response'])}), row.names = 1, check.names = FALSE)",
                f"C <- read.csv(file.path(base, {_r_string(config['phylo_cov'])}), row.names = 1, check.names = FALSE)",
                "C <- as.matrix(C[colnames(Y), colnames(Y), drop = FALSE])",
                'write.csv(C, file.path(out, "C_ordered.csv"))',
            ]
        )
    random_levels = config.get("random_levels") or {}
    if random_levels:
        lines.append(
            f"study <- read.csv(file.path(base, {_r_string(config['study_design'])}), row.names = 1, check.names = FALSE)"
        )
        for idx, (name, spec) in enumerate(random_levels.items()):
            column = spec.get("column", name)
            lines.extend(
                [
                    f"f <- factor(study[[{_r_string(column)}]])",
                    'codes <- as.integer(f) - 1L',
                    "pi <- data.frame(code = codes, level = as.character(f))",
                    f'write.csv(pi, file.path(out, "Pi_{idx}.csv"))',
                ]
            )
    return "\n".join(lines) + "\n"


def _r_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def run_cases(
    configs: list[Path],
    output: Path,
    rscript: str = "Rscript",
    skip_if_missing: bool = False,
) -> dict[str, list[ValidationResult]]:
    if shutil.which(rscript) is None:
        if skip_if_missing:
            return {}
        raise RuntimeError(f"{rscript!r} was not found")
    output.mkdir(parents=True, exist_ok=True)
    return {str(path): run_parity_case(path, output, rscript=rscript) for path in configs}


def _print_report(results: dict[str, list[ValidationResult]]) -> bool:
    if not results:
        print("R parity checks skipped: Rscript not found")
        return True
    failed = False
    for config, checks in results.items():
        print(config)
        for check in checks:
            status = "passed" if check.passed else "failed"
            print(f"  {check.name}: {status} {check.details}")
            failed = failed or not check.passed
    return not failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path, help="model YAML/JSON config paths")
    parser.add_argument("--output", type=Path, default=Path("run/r_parity_checks"))
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--skip-if-r-missing", action="store_true")
    args = parser.parse_args()

    configs = args.configs or DEFAULT_CASES
    try:
        results = run_cases(configs, args.output, rscript=args.rscript, skip_if_missing=args.skip_if_r_missing)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"R parity check failed while running R: {exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if not _print_report(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
