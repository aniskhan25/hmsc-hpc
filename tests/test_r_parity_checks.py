import numpy as np
import pandas as pd

from examples import run_r_parity_checks as parity


def test_compare_matrix_normalizes_r_intercept_name(tmp_path):
    expected = pd.DataFrame(
        {
            "(Intercept)": [1.0, 1.0],
            "x": [0.2, 0.8],
        },
        index=["site1", "site2"],
    )
    expected_path = tmp_path / "X_design.csv"
    expected.to_csv(expected_path)

    result = parity._compare_matrix(
        "X_design",
        np.array([[1.0, 0.2], [1.0, 0.8]]),
        expected_path,
        ["Intercept", "x"],
    )

    assert result.passed
    assert result.details["expected_names"] == ["Intercept", "x"]
    assert result.details["max_abs_diff"] == 0.0


def test_compare_vector_uses_sorted_r_factor_levels(tmp_path):
    expected = pd.DataFrame(
        {
            "code": [1, 0, 1],
            "level": ["b", "a", "b"],
        },
        index=["site1", "site2", "site3"],
    )
    expected_path = tmp_path / "Pi_0.csv"
    expected.to_csv(expected_path)

    result = parity._compare_vector(
        "random_level_plot_codes",
        np.array([1, 0, 1]),
        expected_path,
        expected_levels=["a", "b"],
    )

    assert result.passed
    assert result.details["expected_levels"] == ["a", "b"]


def test_r_script_contains_trait_phylo_and_random_level_sections(tmp_path):
    config_path = tmp_path / "model.yaml"
    config_path.write_text("response: Y.csv\n", encoding="utf-8")
    config = {
        "response": "Y.csv",
        "covariates": "X.csv",
        "formula": {"X": "~ env"},
        "traits": "traits.csv",
        "trait_formula": "~ body",
        "phylo_cov": "C.csv",
        "study_design": "study.csv",
        "random_levels": {"plot": {"column": "plot", "type": "iid"}},
    }

    script = parity._r_script(config_path, config, tmp_path / "r")

    assert "model.matrix(as.formula(\"~ env\"), data = X)" in script
    assert "X_design <- scale_hmsc(X_design)" in script
    assert "if (all(unique(mat[, j]) %in% c(0, 1))) next" in script
    assert "model.matrix(as.formula(\"~ body\"), data = Tr)" in script
    assert "!(colnames(T_design) %in% c('(Intercept)', 'Intercept'))" in script
    assert "T_design <- scale_hmsc(T_design)" in script
    assert "C[colnames(Y), colnames(Y), drop = FALSE]" in script
    assert "factor(study[[\"plot\"]])" in script
    assert "write.csv(pi, file.path(out, \"Pi_0.csv\"))" in script


def test_run_cases_can_skip_when_rscript_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(parity.shutil, "which", lambda _cmd: None)

    results = parity.run_cases(
        [parity.DEFAULT_CASES[0]],
        tmp_path,
        rscript="definitely-missing-Rscript",
        skip_if_missing=True,
    )

    assert results == {}
