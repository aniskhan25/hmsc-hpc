import pandas as pd

from pyhmsc.formulas import covariate_names_from_formula, normalize_formula


def test_normalize_formula_adds_tilde():
    assert normalize_formula("forest_cover + elevation") == "~ forest_cover + elevation"


def test_covariate_names_simple_formula():
    X = pd.DataFrame({"forest_cover": [1], "elevation": [2]})
    assert covariate_names_from_formula("~ forest_cover + elevation", X) == [
        "Intercept",
        "forest_cover",
        "elevation",
    ]


def test_covariate_names_dot_formula():
    X = pd.DataFrame({"forest_cover": [1], "elevation": [2]})
    assert covariate_names_from_formula("~ .", X) == [
        "Intercept",
        "forest_cover",
        "elevation",
    ]
