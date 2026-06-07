import numpy as np
import pandas as pd

from pyhmsc import HmscModel
from pyhmsc.posterior import HmscFit


def test_beta_mean_and_prediction_for_poisson():
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2], "sp2": [3, 4]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="poisson",
    )
    posterior = {
        "0": {
            "0": {"Beta": [[0.0, 1.0], [1.0, 0.0]]},
            "1": {"Beta": [[2.0, 3.0], [1.0, 2.0]]},
        },
        "time": 0.1,
    }
    fit = HmscFit(posterior, model=model)

    beta = fit.beta_mean()

    assert list(beta.index) == ["Intercept", "x"]
    assert list(beta.columns) == ["sp1", "sp2"]
    np.testing.assert_allclose(beta.to_numpy(), [[1.0, 2.0], [1.0, 1.0]])

    pred = fit.predict(pd.DataFrame({"x": [1.0]}))
    np.testing.assert_allclose(
        pred.to_numpy(),
        np.mean(np.exp([[[1.0, 1.0]], [[3.0, 5.0]]]), axis=0),
    )

    samples = fit.predict_samples(pd.DataFrame({"x": [1.0]}))
    assert samples.shape == (1, 2, 1, 2)
    ci = fit.predict_ci(pd.DataFrame({"x": [1.0]}))
    assert ci["lower"].shape == (1, 2)
    assert ci["upper"].shape == (1, 2)

    yrep = fit.posterior_predictive(pd.DataFrame({"x": [1.0]}), rng_seed=7)
    assert yrep.shape == (1, 2, 1, 2)
    assert np.all(yrep >= 0)
    assert np.all(yrep == np.floor(yrep))

    ppc = fit.ppc_summary(model.Y, model.X, rng_seed=7)
    assert list(ppc.columns) == ["species", "observed_mean", "replicated_mean", "lower", "upper", "covered"]
    assert list(ppc["species"]) == ["sp1", "sp2"]

    richness_ppc = fit.richness_ppc_summary(model.Y, model.X, rng_seed=7)
    assert list(richness_ppc.columns) == [
        "site",
        "observed_richness",
        "replicated_richness",
        "lower",
        "upper",
        "covered",
    ]
    assert richness_ppc.shape[0] == len(model.Y)


def test_gaussian_posterior_predictive_uses_sigma():
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1.0, 2.0]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="normal",
    )
    posterior = {
        "0": {
            "0": {"Beta": [[0.0], [1.0]], "sigma": [0.1]},
            "1": {"Beta": [[0.5], [1.0]], "sigma": [0.2]},
        }
    }
    fit = HmscFit(posterior, model=model)

    yrep = fit.posterior_predictive(pd.DataFrame({"x": [1.0]}), rng_seed=3)

    assert yrep.shape == (1, 2, 1, 1)
    assert np.isfinite(yrep).all()


def test_probit_prediction_returns_probability_on_response_scale():
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [0, 1]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="probit",
    )
    posterior = {
        "0": {
            "0": {"Beta": [[0.0], [1.0]]},
            "1": {"Beta": [[0.0], [-1.0]]},
        }
    }
    fit = HmscFit(posterior, model=model)

    linear = fit.predict_samples(pd.DataFrame({"x": [1.0]}), response=False)
    probability = fit.predict_samples(pd.DataFrame({"x": [1.0]}), response=True)

    np.testing.assert_allclose(linear.reshape(-1), [1.0, -1.0])
    np.testing.assert_allclose(probability.reshape(-1), [0.84134475, 0.15865525])
    np.testing.assert_allclose(fit.predict_mean(pd.DataFrame({"x": [1.0]})).to_numpy(), [[0.5]])


def test_gradient_helpers_summarize_response_scale_predictions():
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [0, 1], "sp2": [1, 0]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="probit",
    )
    posterior = {
        "0": {
            "0": {"Beta": [[0.0, 0.0], [1.0, -1.0]]},
            "1": {"Beta": [[0.0, 0.0], [0.5, -0.5]]},
        }
    }
    fit = HmscFit(posterior, model=model)

    richness = fit.richness_gradient("x", model.X, values=[0.0, 1.0], level=0.5)
    traits = pd.DataFrame({"body_size": [10.0, 20.0]}, index=["sp1", "sp2"])
    weighted = fit.trait_weighted_gradient(
        "x",
        traits=traits,
        trait="body_size",
        X_reference=model.X,
        values=[0.0, 1.0],
        level=0.5,
    )

    assert list(richness.columns) == ["x", "mean", "lower", "upper"]
    assert list(weighted.columns) == ["x", "mean", "lower", "upper"]
    assert richness.shape == (2, 4)
    assert weighted.shape == (2, 4)
    np.testing.assert_allclose(richness["mean"].to_numpy(), [1.0, 1.0])
    assert np.isfinite(weighted[["mean", "lower", "upper"]].to_numpy()).all()
    assert weighted["mean"].between(10.0, 20.0).all()


def test_gamma_summary_uses_metadata_names():
    posterior = {
        "__metadata__": {
            "names": {
                "covariates": ["Intercept", "TMG"],
                "traits": ["Intercept", "CN"],
            }
        },
        "__arrays__": {
            "Gamma": np.array(
                [
                    [
                        [[0.0, 1.0], [2.0, 3.0]],
                        [[1.0, 2.0], [3.0, 4.0]],
                    ]
                ]
            )
        },
    }
    fit = HmscFit(posterior)

    mean = fit.gamma_mean()
    summary = fit.gamma_summary(level=0.5)

    assert list(mean.index) == ["Intercept", "TMG"]
    assert list(mean.columns) == ["Intercept", "CN"]
    assert list(summary.columns) == ["covariate", "trait", "mean", "lower", "upper"]
    assert summary.loc[(summary["covariate"] == "TMG") & (summary["trait"] == "CN"), "mean"].iloc[0] == 3.5
    assert "CN" in fit.summary("Gamma").to_string(index=False)


def test_known_random_effect_prediction_from_hdf5(tmp_path):
    import h5py

    path = tmp_path / "posterior.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Beta", data=np.zeros((1, 1, 2, 1)))
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset("Eta", data=np.array([[[[0.2], [0.5]]]]))
        level.create_dataset("Lambda", data=np.array([[[[1.0]]]]))
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="normal",
        study_design=pd.DataFrame({"plot": ["a", "b"]}),
        random_levels={"plot": {"column": "plot", "type": "iid"}},
    )
    fit = HmscFit.from_file(path, model=model)
    pred = fit.predict(pd.DataFrame({"x": [0.0, 1.0], "plot": ["a", "b"]}), random_effects="known")
    np.testing.assert_allclose(pred.to_numpy(), [[0.2], [0.5]])

    zero = fit.predict(
        pd.DataFrame({"x": [0.0], "plot": ["new"]}),
        random_effects="known",
        unseen_groups="zero",
    )
    np.testing.assert_allclose(zero.to_numpy(), [[0.0]])

    marginal = fit.predict(pd.DataFrame({"x": [0.0], "plot": ["new"]}), random_effects="marginal")
    np.testing.assert_allclose(marginal.to_numpy(), [[0.35]])


def test_species_association_summaries_from_lambda():
    posterior = {
        "__metadata__": {"names": {"species": ["sp1", "sp2", "sp3"]}},
        "__arrays__": {
            "random_levels/0/Lambda": np.array(
                [
                    [
                        [[1.0, 2.0, -1.0]],
                        [[2.0, 1.0, -2.0]],
                    ]
                ]
            )
        },
    }
    fit = HmscFit(posterior)

    matrix = fit.species_associations()
    ci = fit.species_association_ci(cred_level=0.5)
    summary = fit.species_association_summary(cred_level=0.5)
    covariance = fit.species_associations(correlation=False)

    assert list(matrix.index) == ["sp1", "sp2", "sp3"]
    assert list(matrix.columns) == ["sp1", "sp2", "sp3"]
    np.testing.assert_allclose(np.diag(matrix), [1.0, 1.0, 1.0])
    assert matrix.loc["sp1", "sp2"] == 1.0
    assert matrix.loc["sp1", "sp3"] == -1.0
    assert ci["lower"].loc["sp1", "sp2"] == 1.0
    assert ci["upper"].loc["sp1", "sp3"] == -1.0
    assert list(summary.columns) == [
        "species_1",
        "species_2",
        "mean",
        "lower",
        "upper",
        "p_positive",
        "p_negative",
    ]
    assert summary.shape[0] == 3
    sp1_sp3 = summary[(summary["species_1"] == "sp1") & (summary["species_2"] == "sp3")].iloc[0]
    assert sp1_sp3["p_negative"] == 1.0
    np.testing.assert_allclose(covariance.loc["sp1", "sp2"], 2.0)


def test_species_associations_require_x_index_for_random_slopes():
    posterior = {
        "__arrays__": {
            "random_levels/0/Lambda": np.ones((1, 1, 1, 2, 2)),
        },
    }
    fit = HmscFit(posterior)

    try:
        fit.species_associations()
    except ValueError as exc:
        assert "x_index is required" in str(exc)
    else:
        raise AssertionError("expected x_index validation error")

    matrix = fit.species_associations(x_index=1)
    assert matrix.shape == (2, 2)
