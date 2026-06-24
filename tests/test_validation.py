import pytest
import numpy as np
import pandas as pd

from pyhmsc import HmscModel
from pyhmsc.posterior import HmscFit
from pyhmsc.validation import (
    coefficient_sign_recovery,
    predictive_interval_contains_observed_mean,
    trait_effect_dimensions,
    validate_compiled_native_model,
)


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


def test_validate_compiled_native_model_rejects_traits_with_random_levels(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 0, 1], "sp2": [0, 1, 0]}),
        X=pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        x_formula="~ x",
        distr="probit",
        traits=pd.DataFrame({"body": [1.0, 2.0]}, index=["sp1", "sp2"]),
        trait_formula="~ body",
        study_design=pd.DataFrame({"plot": ["a", "b", "a"]}),
        random_levels={"plot": {"column": "plot", "type": "iid"}},
    )
    compiled = model.compile(tmp_path / "traits-random", chains=1)

    results = validate_compiled_native_model(compiled.init_json)
    by_name = {result.name: result for result in results}

    assert not by_name["native_sampler_supported"].passed
    unsupported = by_name["native_sampler_supported"].details["unsupported"]
    assert unsupported[0]["feature"] == "traits_phylogeny_with_random_levels"
    assert unsupported[0]["traits"] is True
    assert unsupported[0]["random_levels"] == ["plot"]


def test_python_native_sample_rejects_traits_with_random_levels_before_sampling(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 0, 1], "sp2": [0, 1, 0]}),
        X=pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        x_formula="~ x",
        distr="probit",
        traits=pd.DataFrame({"body": [1.0, 2.0]}, index=["sp1", "sp2"]),
        trait_formula="~ body",
        study_design=pd.DataFrame({"plot": ["a", "b", "a"]}),
        random_levels={"plot": {"column": "plot", "type": "iid"}},
    )

    with pytest.raises(NotImplementedError, match="not sampler-ready"):
        model.sample(
            samples=1,
            transient=0,
            thin=1,
            chains=1,
            init="python-native",
            workdir=tmp_path / "sample",
        )
