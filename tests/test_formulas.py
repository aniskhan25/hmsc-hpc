import pandas as pd

from pyhmsc.formulas import build_design_matrix
from pyhmsc.formulas import covariate_names_from_formula


def test_formula_fallback_preserves_covariate_names_containing_zero_plus():
    data = pd.DataFrame(
        {
            "Hillshading270_40": [0.1, 0.2],
            "HA_All_rivers_normalised": [1.0, 2.0],
        }
    )
    formula = "~ Hillshading270_40 + HA_All_rivers_normalised"

    assert covariate_names_from_formula(formula, data) == [
        "Intercept",
        "Hillshading270_40",
        "HA_All_rivers_normalised",
    ]

    matrix = build_design_matrix(formula, data)

    assert list(matrix.columns) == ["Intercept", "Hillshading270_40", "HA_All_rivers_normalised"]
