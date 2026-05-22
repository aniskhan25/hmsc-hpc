"""R bridge for creating official Hmsc-HPC initialization objects."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

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
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    init_file = Path(init_file)

    y_csv = workdir / "Y.csv"
    x_csv = workdir / "X.csv"
    model.Y.to_csv(y_csv)
    model.X.to_csv(x_csv)

    script_path = workdir / "make_init.R"
    script_path.write_text(
        _init_script(
            y_csv=y_csv,
            x_csv=x_csv,
            init_file=init_file,
            formula=model.x_formula,
            distr=model.distr,
            samples=samples,
            transient=transient,
            thin=thin,
            chains=chains,
            verbose=verbose,
        ),
        encoding="utf-8",
    )
    subprocess.run([rscript_path, str(script_path)], check=True, cwd=workdir)
    return init_file


def _r_string(value: str | Path) -> str:
    return json.dumps(str(value))


def _init_script(
    y_csv: Path,
    x_csv: Path,
    init_file: Path,
    formula: str,
    distr: str,
    samples: int,
    transient: int,
    thin: int,
    chains: int,
    verbose: int,
) -> str:
    return f"""library(Hmsc)
library(jsonify)

Y <- read.csv({_r_string(y_csv)}, row.names = 1, check.names = FALSE)
XData <- read.csv({_r_string(x_csv)}, row.names = 1, check.names = FALSE)

m <- Hmsc(
  Y = as.matrix(Y),
  XData = XData,
  XFormula = {formula},
  distr = {_r_string(distr)}
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
