"""Inspect R/Hmsc spatial random-level boundary semantics.

This is intentionally an inspection workflow, not a parity claim. It creates an
R/Hmsc HPC init object for a spatial random-level config, loads the RDS boundary,
and compares R/Hmsc spatial arrays against Python-native compiled arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hmsc.utils.export_rds_utils import load_model_from_rds
from pyhmsc.config import model_from_config
from pyhmsc.compiler import _select_gpp_knots
from pyhmsc.serialization import read_compiled_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--transient", type=int, default=5)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--chains", type=int)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--prepare-r-init-script", action="store_true")
    parser.add_argument("--r-init-file", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    model, config = model_from_config(args.config)
    chains = int(args.chains or config.get("chains", 2))
    compiled = model.compile(args.output / "python_native" / "compiled", chains=chains)
    metadata, native_arrays = read_compiled_model(compiled.init_json)

    r_root = args.output / "r_bridge"
    r_root.mkdir(parents=True, exist_ok=True)
    init_file = Path(args.r_init_file) if args.r_init_file is not None else r_root / "init_file.rds"

    if args.prepare_r_init_script:
        script_path = _write_spatial_init_script(
            model=model,
            metadata=metadata,
            native_arrays=native_arrays,
            init_file=init_file,
            workdir=r_root,
            samples=args.samples,
            transient=args.transient,
            thin=args.thin,
            chains=chains,
            verbose=args.verbose,
        )
        manifest = {
            "config": str(args.config),
            "compiled_init": str(compiled.init_json),
            "r_script": str(script_path),
            "r_init_file": str(init_file),
        }
        (args.output / "r_init_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(script_path)
        return

    if not init_file.exists():
        raise FileNotFoundError(f"R init file not found: {init_file}")
    r_import, r_model = load_model_from_rds(init_file)
    result = _inspect_boundary(metadata, native_arrays, r_import, r_model)
    result["config"] = str(args.config)
    result["compiled_init"] = str(compiled.init_json)
    result["r_init_file"] = str(init_file)
    (args.output / "spatial_boundary_inspection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_report(args.output / "spatial_boundary_inspection_report.md", result)
    print((args.output / "spatial_boundary_inspection_report.md").read_text(encoding="utf-8"))


def _write_spatial_init_script(
    *,
    model: Any,
    metadata: dict[str, Any],
    native_arrays: dict[str, np.ndarray],
    init_file: Path,
    workdir: Path,
    samples: int,
    transient: int,
    thin: int,
    chains: int,
    verbose: int,
) -> Path:
    y_csv = workdir / "Y.csv"
    x_csv = workdir / "X.csv"
    study_csv = workdir / "studyDesign_raw.csv"
    sdata_csv = workdir / "sData.csv"
    model.Y.to_csv(y_csv)
    model.X.to_csv(x_csv)
    if model.study_design is None or not model.random_levels:
        raise ValueError("spatial boundary inspection requires study_design and random_levels")
    model.study_design.to_csv(study_csv)

    level_name, spec = next(iter(model.random_levels.items()))
    level_meta = metadata["random_levels"][0]
    level_type = str(spec.get("type", "iid"))
    if level_type not in {"spatial_full", "spatial_gpp", "gpp", "spatial_nngp", "nngp"}:
        raise ValueError(f"expected a spatial random level, got {level_type!r}")
    column = str(spec.get("column", level_name))
    coords = [str(value) for value in spec.get("coords", ["x", "y"])]
    sdata = (
        model.study_design.assign(__unit=model.study_design[column].astype(str))
        .groupby("__unit", sort=True)[coords]
        .mean()
    )
    sdata.to_csv(sdata_csv)

    sknot_csv = None
    if level_type in {"spatial_gpp", "gpp"}:
        n_knots = int(spec.get("n_knots", spec.get("nKnots", min(max(2, int(np.sqrt(len(sdata)))), len(sdata)))))
        knots = np.asarray(level_meta.get("knots"), dtype=float) if "knots" in level_meta else _select_gpp_knots(sdata.to_numpy(dtype=float), n_knots)
        sknot_csv = workdir / "sKnot.csv"
        pd.DataFrame(knots, columns=coords).to_csv(sknot_csv)

    script = _r_spatial_init_script(
        y_csv=y_csv,
        x_csv=x_csv,
        study_csv=study_csv,
        sdata_csv=sdata_csv,
        sknot_csv=sknot_csv,
        init_file=init_file,
        formula=model.x_formula,
        distr=model.distr,
        level_name=level_name,
        column=column,
        level_type=level_type,
        n_neighbors=int(spec.get("n_neighbors", spec.get("nNeighbors", 10))),
        nf_min=int(level_meta.get("nfMin", spec.get("nf", 1))),
        nf_max=int(level_meta.get("nfMax", max(int(spec.get("nf", 1)), 4))),
        samples=samples,
        transient=transient,
        thin=thin,
        chains=chains,
        verbose=verbose,
    )
    script_path = workdir / "make_spatial_init.R"
    script_path.write_text(script, encoding="utf-8")
    return script_path


def _r_spatial_init_script(
    *,
    y_csv: Path,
    x_csv: Path,
    study_csv: Path,
    sdata_csv: Path,
    sknot_csv: Path | None,
    init_file: Path,
    formula: str,
    distr: str,
    level_name: str,
    column: str,
    level_type: str,
    n_neighbors: int,
    nf_min: int,
    nf_max: int,
    samples: int,
    transient: int,
    thin: int,
    chains: int,
    verbose: int,
) -> str:
    if level_type == "spatial_full":
        random_level_call = "HmscRandomLevel(sData = sData, sMethod = \"Full\")"
        optional = ""
    elif level_type in {"spatial_gpp", "gpp"}:
        random_level_call = "HmscRandomLevel(sData = sData, sMethod = \"GPP\", sKnot = as.matrix(sKnot))"
        optional = f"sKnot <- read.csv({_r_string(sknot_csv)}, row.names = 1, check.names = FALSE)\n"
    elif level_type in {"spatial_nngp", "nngp"}:
        random_level_call = f"HmscRandomLevel(sData = sData, sMethod = \"NNGP\", nNeighbours = {int(n_neighbors)})"
        optional = ""
    else:
        raise ValueError(f"unsupported spatial type {level_type!r}")
    return f"""library(Hmsc)
library(jsonify)

Y <- read.csv({_r_string(y_csv)}, row.names = 1, check.names = FALSE)
XData <- read.csv({_r_string(x_csv)}, row.names = 1, check.names = FALSE)
studyDesign <- read.csv({_r_string(study_csv)}, row.names = 1, check.names = FALSE)
studyDesign <- studyDesign[, c({_r_string(column)}), drop = FALSE]
studyDesign[[{_r_string(column)}]] <- factor(studyDesign[[{_r_string(column)}]])
sData <- read.csv({_r_string(sdata_csv)}, row.names = 1, check.names = FALSE)
sData <- sData[levels(studyDesign[[{_r_string(column)}]]), , drop = FALSE]
{optional}

rL <- {random_level_call}
rL <- setPriors(rL, nfMin = {int(nf_min)}, nfMax = {int(nf_max)})

m <- Hmsc(
  Y = as.matrix(Y),
  XData = XData,
  XFormula = {formula},
  distr = {_r_string(distr)},
  studyDesign = studyDesign,
  ranLevels = list({_r_name(level_name)} = rL)
)

init_obj <- sampleMcmc(
  m,
  samples = {int(samples)},
  thin = {int(thin)},
  transient = {int(transient)},
  nChains = {int(chains)},
  verbose = {int(verbose)},
  engine = "HPC"
)

saveRDS(to_json(init_obj), file = {_r_string(init_file)})
"""


def _inspect_boundary(
    metadata: dict[str, Any],
    native_arrays: dict[str, np.ndarray],
    r_import: dict[str, Any],
    r_model: dict[str, Any],
) -> dict[str, Any]:
    level = metadata["random_levels"][0]
    level_type = level["type"]
    r_level_name = list(r_model.get("rL", {}).keys())[0]
    r_level = r_model["rL"][r_level_name]
    data_par = r_import["dataParList"]["rLPar"][0]
    result = {
        "level_type": level_type,
        "r_level_name": r_level_name,
        "level_metadata": {
            "n_levels": level.get("n_levels"),
            "levels": level.get("levels"),
            "nfMin": level.get("nfMin"),
            "nfMax": level.get("nfMax"),
            "alphapw": level.get("alphapw"),
        },
        "r_level": {
            "N": _scalar(r_level.get("N")),
            "sDim": _scalar(r_level.get("sDim")),
            "spatialMethod": _first(r_level.get("spatialMethod")),
            "nfMin": _scalar(r_level.get("nfMin")),
            "nfMax": _scalar(r_level.get("nfMax")),
            "alphapw": r_level.get("alphapw"),
        },
        "boundary_arrays": {
            "Y": _compare_array(native_arrays["Y"], r_model.get("YScaled")),
            "X": _compare_array(native_arrays["X"], r_model.get("XScaled")),
            "T": _compare_array(native_arrays["T"], r_model.get("TrScaled")),
            "Pi": _compare_array(native_arrays["Pi"], np.asarray(r_model.get("Pi"), dtype=int) - 1),
        },
        "spatial_arrays": {},
    }
    result["spatial_arrays"]["alphapw"] = _compare_array(level.get("alphapw"), r_level.get("alphapw"))
    prefix = level["array_prefix"]
    if level_type == "spatial_full":
        result["spatial_arrays"]["distMat"] = _compare_array(
            native_arrays[f"{prefix}_distMat"],
            np.asarray(data_par["distMat"], dtype=float).reshape(level["n_levels"], level["n_levels"]),
        )
    elif level_type in {"spatial_gpp", "gpp"}:
        n_knots = int(_scalar(data_par["nKnots"]))
        result["spatial_arrays"]["nKnots"] = {
            "native": int(level.get("nKnots", native_arrays[f"{prefix}_distMat22"].shape[0])),
            "r": n_knots,
            "passed": int(level.get("nKnots", native_arrays[f"{prefix}_distMat22"].shape[0])) == n_knots,
        }
        result["spatial_arrays"]["distMat12"] = _compare_array(
            native_arrays[f"{prefix}_distMat12"],
            np.asarray(data_par["distMat12"], dtype=float).reshape(level["n_levels"], n_knots),
        )
        result["spatial_arrays"]["distMat22"] = _compare_array(
            native_arrays[f"{prefix}_distMat22"],
            np.asarray(data_par["distMat22"], dtype=float).reshape(n_knots, n_knots),
        )
    elif level_type in {"spatial_nngp", "nngp"}:
        result["spatial_arrays"]["nNeighbours"] = {
            "native": int(level.get("nNeighbors", native_arrays[f"{prefix}_nngp_indices"].shape[1])),
            "r": int(_scalar(r_level.get("nNeighbours"))),
        }
        result["spatial_arrays"]["r_indices_summary"] = _list_summary(data_par.get("indices"))
        result["spatial_arrays"]["r_distList_summary"] = _list_summary(data_par.get("distList"))
        result["spatial_arrays"]["native_indices_shape"] = [int(value) for value in native_arrays[f"{prefix}_nngp_indices"].shape]
        result["spatial_arrays"]["native_distances_shape"] = [int(value) for value in native_arrays[f"{prefix}_nngp_distances"].shape]
    result["boundary_passed"] = bool(
        all(check["passed"] for check in result["boundary_arrays"].values())
        and all(
            check.get("passed", True)
            for check in result["spatial_arrays"].values()
            if isinstance(check, dict)
        )
    )
    return result


def _compare_array(native: Any, r_value: Any, *, atol: float = 1e-10) -> dict[str, Any]:
    native_array = np.asarray(native, dtype=float)
    r_array = np.asarray(r_value, dtype=float)
    shape_ok = native_array.shape == r_array.shape
    max_abs_diff = None
    passed = False
    if shape_ok:
        max_abs_diff = float(np.max(np.abs(native_array - r_array))) if native_array.size else 0.0
        passed = bool(np.allclose(native_array, r_array, atol=atol, rtol=0.0))
    return {
        "passed": passed,
        "native_shape": [int(value) for value in native_array.shape],
        "r_shape": [int(value) for value in r_array.shape],
        "max_abs_diff": max_abs_diff,
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    boundary_lines = [
        f"| {name} | {check['passed']} | {check['native_shape']} | {check['r_shape']} | {check['max_abs_diff']} |"
        for name, check in result["boundary_arrays"].items()
    ]
    spatial_lines = []
    for name, value in result["spatial_arrays"].items():
        if isinstance(value, dict) and "native_shape" in value:
            spatial_lines.append(
                f"| {name} | {value['passed']} | {value['native_shape']} | {value['r_shape']} | {value['max_abs_diff']} |"
            )
        else:
            spatial_lines.append(f"| {name} |  |  |  | `{json.dumps(value, default=_json_default)}` |")
    report = [
        "# R/Hmsc Spatial Boundary Inspection",
        "",
        f"Config: `{result['config']}`",
        f"Boundary passed: `{result['boundary_passed']}`",
        f"Spatial type: `{result['level_type']}`",
        f"R spatial method: `{result['r_level']['spatialMethod']}`",
        "",
        "## Boundary Arrays",
        "",
        "| array | passed | native shape | R shape | max abs diff |",
        "| --- | --- | --- | --- | --- |",
        *boundary_lines,
        "",
        "## Spatial Arrays",
        "",
        "| array | passed | native shape | R shape | max abs diff / summary |",
        "| --- | --- | --- | --- | --- |",
        *spatial_lines,
        "",
        "## Random-Level Metadata",
        "",
        "```json",
        json.dumps({"native": result["level_metadata"], "r": result["r_level"]}, indent=2, sort_keys=True, default=_json_default),
        "```",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def _r_string(value: str | Path | None) -> str:
    return json.dumps(str(value))


def _r_name(value: str) -> str:
    if value.isidentifier():
        return value
    return f"`{value}`"


def _scalar(value: Any) -> Any:
    if isinstance(value, list):
        return _scalar(value[0]) if value else None
    if isinstance(value, np.ndarray):
        return _scalar(value.tolist())
    return value


def _first(value: Any) -> Any:
    return _scalar(value)


def _list_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        return {"type": type(value).__name__}
    lengths = []
    shapes = []
    for item in value:
        array = np.asarray(item)
        lengths.append(int(array.size))
        shapes.append([int(dim) for dim in array.shape])
    return {
        "length": len(value),
        "item_lengths_head": lengths[:10],
        "item_shapes_head": shapes[:10],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


if __name__ == "__main__":
    main()
