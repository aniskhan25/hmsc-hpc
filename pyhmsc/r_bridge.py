"""R bridge for creating official Hmsc-HPC initialization objects."""

from __future__ import annotations

import json
import numpy as np
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from pyhmsc.compiler import _select_gpp_knots

if TYPE_CHECKING:
    from pyhmsc.model import HmscModel


def check_r_available(rscript: str = "Rscript") -> str:
    resolved = shutil.which(rscript)
    if not resolved:
        raise RuntimeError(
            f"{rscript!r} was not found. Phase 1 requires R with Hmsc and jsonify installed."
        )
    return resolved


def make_init_with_r(
    model: "HmscModel",
    init_file: str | Path,
    workdir: str | Path,
    samples: int,
    transient: int,
    thin: int,
    chains: int,
    verbose: int = 100,
    rscript: str = "Rscript",
) -> Path:
    """Write model data, generate an R script, and run official Hmsc initialization."""
    rscript_path = check_r_available(rscript)
    script_path = write_init_script_with_r(
        model,
        init_file=init_file,
        workdir=workdir,
        samples=samples,
        transient=transient,
        thin=thin,
        chains=chains,
        verbose=verbose,
    )
    subprocess.run([rscript_path, str(script_path)], check=True, cwd=Path(workdir))
    return Path(init_file)


def write_init_script_with_r(
    model: "HmscModel",
    init_file: str | Path,
    workdir: str | Path,
    samples: int,
    transient: int,
    thin: int,
    chains: int,
    verbose: int = 100,
) -> Path:
    """Write model CSV inputs and an R script that creates an Hmsc-HPC init RDS."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    init_file = Path(init_file)

    y_csv = workdir / "Y.csv"
    x_csv = workdir / "X.csv"
    model.Y.to_csv(y_csv)
    model.X.to_csv(x_csv)
    traits_csv = None
    if model.traits is not None:
        traits_csv = workdir / "Tr.csv"
        model.traits.to_csv(traits_csv)
    phylo_csv = None
    if model.phylo_cov is not None:
        phylo_csv = workdir / "C.csv"
        model.phylo_cov.to_csv(phylo_csv)
    study_design_csv = None
    if model.study_design is not None:
        study_design_csv = workdir / "studyDesign.csv"
        model.study_design.to_csv(study_design_csv)

    script_path = workdir / "make_init.R"
    script_path.write_text(
        _init_script(
            y_csv=y_csv,
            x_csv=x_csv,
            traits_csv=traits_csv,
            phylo_csv=phylo_csv,
            study_design_csv=study_design_csv,
            random_levels=model.random_levels,
            init_file=init_file,
            formula=model.x_formula,
            trait_formula=model.trait_formula,
            distr=model.distr,
            samples=samples,
            transient=transient,
            thin=thin,
            chains=chains,
            verbose=verbose,
        ),
        encoding="utf-8",
    )
    return script_path


def _r_string(value: str | Path) -> str:
    return json.dumps(str(value))


def _init_script(
    y_csv: Path,
    x_csv: Path,
    traits_csv: Path | None,
    phylo_csv: Path | None,
    init_file: Path,
    formula: str,
    trait_formula: str | None,
    distr: str,
    samples: int,
    transient: int,
    thin: int,
    chains: int,
    verbose: int,
    study_design_csv: Path | None = None,
    random_levels: dict | None = None,
) -> str:
    optional_reads = []
    optional_args = []
    if traits_csv is not None:
        optional_reads.extend(
            [
                f"Tr <- read.csv({_r_string(traits_csv)}, row.names = 1, check.names = FALSE)",
                "Tr <- Tr[colnames(Y), , drop = FALSE]",
            ]
        )
        optional_args.append("Tr = as.matrix(Tr)")
        optional_args.append(f"TrFormula = as.formula({_r_string(trait_formula or '~ .')})")
    if phylo_csv is not None:
        optional_reads.extend(
            [
                f"C <- read.csv({_r_string(phylo_csv)}, row.names = 1, check.names = FALSE)",
                "C <- as.matrix(C[colnames(Y), colnames(Y), drop = FALSE])",
            ]
        )
        optional_args.append("C = C")
    if random_levels:
        if study_design_csv is None:
            raise ValueError("study_design is required when random_levels are provided")
        random_columns = [str(spec.get("column", name)) for name, spec in random_levels.items()]
        has_spatial = any(
            str(spec.get("type", "iid")) in {"spatial_full", "spatial_gpp", "gpp", "spatial_nngp", "nngp"}
            for spec in random_levels.values()
        )
        spatial_inputs = (
            _write_spatial_random_level_inputs(
                study_design_csv.parent,
                pd.read_csv(study_design_csv, index_col=0),
                random_levels,
            )
            if has_spatial
            else {}
        )
        optional_reads.extend(
            [
                f"studyDesign <- read.csv({_r_string(study_design_csv)}, row.names = 1, check.names = FALSE)",
                f"studyDesign <- studyDesign[, c({', '.join(_r_string(column) for column in random_columns)}), drop = FALSE]",
                "ranLevels <- list()",
            ]
        )
        for name, spec in random_levels.items():
            level_type = str(spec.get("type", "iid"))
            column = str(spec.get("column", name))
            optional_reads.append(
                f"studyDesign[[{_r_string(column)}]] <- factor(studyDesign[[{_r_string(column)}]])"
            )
            if level_type == "iid":
                optional_reads.append(
                    f"ranLevels[[{_r_string(name)}]] <- HmscRandomLevel(units = studyDesign[[{_r_string(column)}]])"
                )
            elif level_type in {"spatial_full", "spatial_gpp", "gpp", "spatial_nngp", "nngp"}:
                spatial = spatial_inputs[name]
                optional_reads.extend(_spatial_random_level_r_block(name, spec, column, spatial))
            else:
                raise NotImplementedError(f"R bridge init writer does not support random level type {level_type!r}")
        optional_args.append("studyDesign = studyDesign")
        optional_args.append("ranLevels = ranLevels")
    optional_read_block = "\n".join(optional_reads)
    optional_arg_block = "".join(f",\n  {arg}" for arg in optional_args)
    return f"""library(Hmsc)
library(jsonify)

Y <- read.csv({_r_string(y_csv)}, row.names = 1, check.names = FALSE)
XData <- read.csv({_r_string(x_csv)}, row.names = 1, check.names = FALSE)
{optional_read_block}

m <- Hmsc(
  Y = as.matrix(Y),
  XData = XData,
  XFormula = {formula},
  distr = {_r_string(distr)}{optional_arg_block}
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


def _write_spatial_random_level_inputs(
    workdir: Path,
    study_design: pd.DataFrame,
    random_levels: dict,
) -> dict[str, dict[str, Path]]:
    spatial_inputs: dict[str, dict[str, Path]] = {}
    for index, (name, spec) in enumerate(random_levels.items()):
        level_type = str(spec.get("type", "iid"))
        if level_type not in {"spatial_full", "spatial_gpp", "gpp", "spatial_nngp", "nngp"}:
            continue
        column = str(spec.get("column", name))
        coords = [str(value) for value in spec.get("coords", ["x", "y"])]
        if column not in study_design:
            raise ValueError(f"study_design is missing random level column {column!r}")
        missing = [coord for coord in coords if coord not in study_design]
        if missing:
            raise ValueError(f"study_design is missing spatial coordinate columns: {missing}")
        sdata = (
            study_design.assign(__unit=study_design[column].astype(str))
            .groupby("__unit", sort=True)[coords]
            .mean()
        )
        sdata_csv = workdir / f"sData_random_level_{index}.csv"
        sdata.to_csv(sdata_csv)
        paths = {"sData": sdata_csv}
        if level_type in {"spatial_gpp", "gpp"}:
            n_knots = int(spec.get("n_knots", spec.get("nKnots", min(max(2, int(np.sqrt(len(sdata)))), len(sdata)))))
            knots = _select_gpp_knots(sdata.to_numpy(dtype=float), n_knots)
            sknot_csv = workdir / f"sKnot_random_level_{index}.csv"
            pd.DataFrame(knots, columns=coords).to_csv(sknot_csv)
            paths["sKnot"] = sknot_csv
        spatial_inputs[name] = paths
    return spatial_inputs


def _spatial_random_level_r_block(
    name: str,
    spec: dict,
    column: str,
    spatial: dict[str, Path],
) -> list[str]:
    level_type = str(spec.get("type", "iid"))
    n_neighbors = int(spec.get("n_neighbors", spec.get("nNeighbors", 10)))
    nf = int(spec.get("nf", 1))
    nf_min = int(spec.get("nfMin", nf))
    nf_max = int(spec.get("nfMax", max(nf, 4)))
    r_lvalue = f"ranLevels[[{_r_string(name)}]]"
    lines = [
        f"sData_{name} <- read.csv({_r_string(spatial['sData'])}, row.names = 1, check.names = FALSE)",
        f"sData_{name} <- sData_{name}[levels(studyDesign[[{_r_string(column)}]]), , drop = FALSE]",
    ]
    if level_type == "spatial_full":
        lines.append(f"rL_{name} <- HmscRandomLevel(sData = sData_{name}, sMethod = \"Full\")")
    elif level_type in {"spatial_gpp", "gpp"}:
        lines.append(f"sKnot_{name} <- read.csv({_r_string(spatial['sKnot'])}, row.names = 1, check.names = FALSE)")
        lines.append(
            f"rL_{name} <- HmscRandomLevel(sData = sData_{name}, sMethod = \"GPP\", sKnot = as.matrix(sKnot_{name}))"
        )
    elif level_type in {"spatial_nngp", "nngp"}:
        lines.append(
            f"rL_{name} <- HmscRandomLevel(sData = sData_{name}, sMethod = \"NNGP\", nNeighbours = {n_neighbors})"
        )
    else:
        raise NotImplementedError(f"R bridge init writer does not support random level type {level_type!r}")
    lines.append(f"rL_{name} <- setPriors(rL_{name}, nfMin = {nf_min}, nfMax = {nf_max})")
    lines.append(f"{r_lvalue} <- rL_{name}")
    return lines
