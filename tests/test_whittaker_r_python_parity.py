from pathlib import Path

import numpy as np
import pandas as pd

from examples.run_whittaker_r_python_parity import (
    _acceptance_gates,
    _boundary_checks,
    _metric_deltas,
    _posterior_compare,
)
from pyhmsc.model import HmscModel
from pyhmsc.r_bridge import _init_script, write_init_script_with_r


def test_r_bridge_script_preserves_traits_and_phylogeny():
    script = _init_script(
        y_csv=Path("Y.csv"),
        x_csv=Path("X.csv"),
        traits_csv=Path("Tr.csv"),
        phylo_csv=Path("C.csv"),
        init_file=Path("init.rds"),
        formula="~ TMG",
        trait_formula="~ CN",
        distr="probit",
        samples=2,
        transient=1,
        thin=1,
        chains=1,
        verbose=1,
    )

    assert "Tr <- read.csv" in script
    assert "Tr <- Tr[colnames(Y), , drop = FALSE]" in script
    assert "C <- as.matrix(C[colnames(Y), colnames(Y), drop = FALSE])" in script
    assert "Tr = as.matrix(Tr)" in script
    assert 'TrFormula = as.formula("~ CN")' in script
    assert "C = C" in script


def test_r_bridge_script_preserves_iid_random_level():
    script = _init_script(
        y_csv=Path("Y.csv"),
        x_csv=Path("X.csv"),
        traits_csv=None,
        phylo_csv=None,
        init_file=Path("init.rds"),
        formula="~ x",
        trait_formula=None,
        distr="poisson",
        samples=2,
        transient=1,
        thin=1,
        chains=1,
        verbose=1,
        study_design_csv=Path("studyDesign.csv"),
        random_levels={"plot": {"column": "plot", "type": "iid"}},
    )

    assert "studyDesign <- read.csv" in script
    assert 'studyDesign <- studyDesign[, c("plot"), drop = FALSE]' in script
    assert "ranLevels <- list()" in script
    assert 'studyDesign[["plot"]] <- factor(studyDesign[["plot"]])' in script
    assert 'ranLevels[["plot"]] <- HmscRandomLevel(units = studyDesign[["plot"]])' in script
    assert "studyDesign = studyDesign" in script
    assert "ranLevels = ranLevels" in script


def test_r_bridge_script_writes_spatial_gpp_random_level(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 0, 1, 2], "sp2": [0, 1, 2, 1]}),
        X=pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        x_formula="~ x",
        distr="poisson",
        study_design=pd.DataFrame(
            {
                "plot": ["b", "a", "d", "c"],
                "xcoord": [1.0, 0.0, 1.0, 0.0],
                "ycoord": [0.0, 0.0, 1.0, 1.0],
            }
        ),
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_gpp",
                "coords": ["xcoord", "ycoord"],
                "n_knots": 2,
                "nf": 1,
                "nfMax": 4,
            }
        },
    )

    script_path = write_init_script_with_r(
        model,
        init_file=tmp_path / "init.rds",
        workdir=tmp_path,
        samples=2,
        transient=1,
        thin=1,
        chains=1,
        verbose=1,
    )
    script = script_path.read_text(encoding="utf-8")

    assert (tmp_path / "sData_random_level_0.csv").exists()
    assert (tmp_path / "sKnot_random_level_0.csv").exists()
    assert 'studyDesign <- studyDesign[, c("plot"), drop = FALSE]' in script
    assert 'studyDesign[["plot"]] <- factor(studyDesign[["plot"]])' in script
    assert "sData_plot <- read.csv" in script
    assert 'sData_plot <- sData_plot[levels(studyDesign[["plot"]]), , drop = FALSE]' in script
    assert 'HmscRandomLevel(sData = sData_plot, sMethod = "GPP", sKnot = as.matrix(sKnot_plot))' in script
    assert "rL_plot <- setPriors(rL_plot, nfMin = 1, nfMax = 4)" in script
    assert 'ranLevels[["plot"]] <- rL_plot' in script


def test_boundary_checks_compare_native_against_r_import():
    native = {
        "Y": np.array([[1.0, 0.0]]),
        "X": np.array([[1.0, 2.0]]),
        "T": np.array([[1.0], [1.0]]),
        "C": np.eye(2),
    }
    r_model = {
        "YScaled": [[1.0, 0.0]],
        "XScaled": [[1.0, 2.0]],
        "TrScaled": [[1.0], [1.0]],
        "C": np.eye(2).tolist(),
    }

    checks = _boundary_checks(native, r_model)

    assert all(check["passed"] for check in checks.values())
    assert checks["X"]["max_abs_diff"] == 0.0


def test_posterior_and_metric_gates_pass_for_close_runs():
    left = np.arange(24, dtype=float).reshape(2, 3, 2, 2)
    right = left + 0.01
    beta_compare = _posterior_compare(left, right)
    gamma_compare = _posterior_compare(left, right)
    heldout = pd.DataFrame(
        [
            {"model": "python_native", "brier_score": 0.11, "log_loss": 0.22, "macro_auc": 0.6, "prevalence_mae": 0.3, "richness_mae": 1.0},
            {"model": "r_bridge", "brier_score": 0.10, "log_loss": 0.20, "macro_auc": 0.61, "prevalence_mae": 0.31, "richness_mae": 1.1},
        ]
    )
    deltas = _metric_deltas(heldout, baseline="r_bridge", candidate="python_native")

    gates = _acceptance_gates(
        boundary={"X": {"passed": True}},
        beta_compare=beta_compare,
        gamma_compare=gamma_compare,
        deltas=deltas,
        min_beta_corr=0.95,
        min_gamma_corr=0.95,
        max_brier_delta=0.02,
        max_log_loss_delta=0.05,
    )

    assert beta_compare["mean_correlation"] > 0.99
    assert all(gate["passed"] for gate in gates.values())
