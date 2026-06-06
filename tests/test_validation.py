import numpy as np
import pandas as pd

from pyhmsc import HmscModel
from pyhmsc.posterior import HmscFit
from pyhmsc.validation import coefficient_sign_recovery, predictive_interval_contains_observed_mean, trait_effect_dimensions


def test_coefficient_sign_recovery_passes_for_matching_signs():
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2], "sp2": [2, 1]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="normal",
    )
    fit = HmscFit({"0": {"0": {"Beta": [[0.0, 0.0], [1.0, -1.0]]}}}, model=model)
    truth = pd.DataFrame([[0.0, 0.0], [0.5, -0.5]], index=["Intercept", "x"], columns=["sp1", "sp2"])
    result = coefficient_sign_recovery(fit, truth)
    assert result.passed


def test_predictive_interval_validation_returns_result():
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1.0, 1.2]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="normal",
    )
    posterior = {
        "0": {"0": {"Beta": [[1.0], [0.0]]}, "1": {"Beta": [[1.1], [0.0]]}},
        "1": {"0": {"Beta": [[0.9], [0.0]]}, "1": {"Beta": [[1.2], [0.0]]}},
    }
    fit = HmscFit(posterior, model=model)
    result = predictive_interval_contains_observed_mean(fit, model.X, model.Y, level=0.99)
    assert isinstance(result.passed, bool)


def test_trait_effect_dimensions_requires_named_gamma_axes():
    fit = HmscFit(
        {
            "__metadata__": {
                "names": {
                    "covariates": ["Intercept", "x"],
                    "traits": ["Intercept", "body_size"],
                }
            },
            "__arrays__": {"Gamma": np.ones((1, 2, 2, 2))},
        }
    )

    result = trait_effect_dimensions(fit)

    assert result.passed
    assert result.details["shape"] == (2, 2)
