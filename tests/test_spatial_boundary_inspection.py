from pathlib import Path

import numpy as np

from examples.inspect_r_spatial_boundary import _compare_array, _r_spatial_init_script


def test_spatial_full_r_script_uses_sdata_and_factor_study_design():
    script = _r_spatial_init_script(
        y_csv=Path("Y.csv"),
        x_csv=Path("X.csv"),
        study_csv=Path("study.csv"),
        sdata_csv=Path("sData.csv"),
        sknot_csv=None,
        init_file=Path("init.rds"),
        formula="~ env",
        distr="probit",
        level_name="plot",
        column="plot",
        level_type="spatial_full",
        n_neighbors=10,
        nf_min=1,
        nf_max=4,
        samples=2,
        transient=1,
        thin=1,
        chains=1,
        verbose=1,
    )

    assert "studyDesign <- studyDesign[, c(\"plot\"), drop = FALSE]" in script
    assert "studyDesign[[\"plot\"]] <- factor(studyDesign[[\"plot\"]])" in script
    assert "sData <- sData[levels(studyDesign[[\"plot\"]]), , drop = FALSE]" in script
    assert 'HmscRandomLevel(sData = sData, sMethod = "Full")' in script
    assert "setPriors(rL, nfMin = 1, nfMax = 4)" in script


def test_spatial_gpp_r_script_uses_native_knots():
    script = _r_spatial_init_script(
        y_csv=Path("Y.csv"),
        x_csv=Path("X.csv"),
        study_csv=Path("study.csv"),
        sdata_csv=Path("sData.csv"),
        sknot_csv=Path("sKnot.csv"),
        init_file=Path("init.rds"),
        formula="~ env",
        distr="normal",
        level_name="plot",
        column="plot",
        level_type="spatial_gpp",
        n_neighbors=10,
        nf_min=1,
        nf_max=4,
        samples=2,
        transient=1,
        thin=1,
        chains=1,
        verbose=1,
    )

    assert "sKnot <- read.csv" in script
    assert 'HmscRandomLevel(sData = sData, sMethod = "GPP", sKnot = as.matrix(sKnot))' in script


def test_spatial_nngp_r_script_uses_nneighbours():
    script = _r_spatial_init_script(
        y_csv=Path("Y.csv"),
        x_csv=Path("X.csv"),
        study_csv=Path("study.csv"),
        sdata_csv=Path("sData.csv"),
        sknot_csv=None,
        init_file=Path("init.rds"),
        formula="~ env",
        distr="normal",
        level_name="plot",
        column="plot",
        level_type="spatial_nngp",
        n_neighbors=15,
        nf_min=1,
        nf_max=4,
        samples=2,
        transient=1,
        thin=1,
        chains=1,
        verbose=1,
    )

    assert 'HmscRandomLevel(sData = sData, sMethod = "NNGP", nNeighbours = 15)' in script


def test_compare_array_reports_shape_and_value_mismatch():
    match = _compare_array(np.array([[1.0, 2.0]]), [[1.0, 2.0]])
    mismatch = _compare_array(np.array([[1.0, 2.0]]), [[1.0], [2.0]])

    assert match["passed"]
    assert match["max_abs_diff"] == 0.0
    assert not mismatch["passed"]
    assert mismatch["max_abs_diff"] is None
