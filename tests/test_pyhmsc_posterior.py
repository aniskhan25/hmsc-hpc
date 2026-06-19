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


def test_known_random_effect_prediction_accepts_separate_study_design():
    posterior = {
        "__arrays__": {
            "Beta": np.zeros((1, 1, 2, 1)),
            "random_levels/0/Eta": np.array([[[[0.2], [0.5]]]]),
            "random_levels/0/Lambda": np.array([[[[1.0]]]]),
        }
    }
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="normal",
        study_design=pd.DataFrame({"plot": ["a", "b"]}),
        random_levels={"plot": {"column": "plot", "type": "iid"}},
    )
    fit = HmscFit(posterior, model=model)

    pred = fit.predict(
        pd.DataFrame({"x": [0.0, 1.0]}, index=["site_a", "site_b"]),
        study_design=pd.DataFrame({"plot": ["a", "b"]}),
        include_random_effects=True,
    )

    assert list(pred.index) == ["site_a", "site_b"]
    np.testing.assert_allclose(pred.to_numpy(), [[0.2], [0.5]])


def test_random_slope_prediction_uses_new_row_covariates():
    posterior = {
        "__arrays__": {
            "Beta": np.zeros((1, 1, 2, 1)),
            "random_levels/0/Eta": np.array([[[[1.0], [1.0]]]]),
            "random_levels/0/Lambda": np.array([[[[[0.0, 2.0]]]]]),
        }
    }
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="normal",
        study_design=pd.DataFrame({"plot": ["a", "b"], "slope_env": [1.0, 1.0]}),
        random_levels={"plot": {"column": "plot", "type": "iid", "x_formula": "~ slope_env"}},
    )
    fit = HmscFit(posterior, model=model)

    pred = fit.predict(
        pd.DataFrame({"x": [0.0], "slope_env": [5.0]}),
        study_design=pd.DataFrame({"plot": ["a"]}),
        random_effects="known",
    )

    np.testing.assert_allclose(pred.to_numpy(), [[10.0]])


def test_spatial_nearest_prediction_accepts_separate_coords_for_unseen_group():
    posterior = {
        "__arrays__": {
            "Beta": np.zeros((1, 1, 2, 1)),
            "random_levels/0/Eta": np.array([[[[0.2], [0.8]]]]),
            "random_levels/0/Lambda": np.array([[[[1.0]]]]),
        }
    }
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="normal",
        study_design=pd.DataFrame({"plot": ["a", "b"], "xcoord": [0.0, 10.0], "ycoord": [0.0, 0.0]}),
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_nngp",
                "coords": ["xcoord", "ycoord"],
            }
        },
    )
    fit = HmscFit(posterior, model=model)

    pred = fit.predict(
        pd.DataFrame({"x": [0.0]}),
        study_design=pd.DataFrame({"plot": ["new"]}),
        coords=pd.DataFrame({"xcoord": [9.0], "ycoord": [0.0]}),
        random_effects="known",
        unseen_groups="nearest",
    )

    np.testing.assert_allclose(pred.to_numpy(), [[0.8]])


def test_full_spatial_conditional_prediction_matches_gaussian_conditioning():
    draws = 6000
    eta = np.zeros((1, draws, 2, 1))
    eta[:, :, 1, 0] = 1.0
    posterior = {
        "__arrays__": {
            "Beta": np.zeros((1, draws, 2, 1)),
            "random_levels/0/Eta": eta,
            "random_levels/0/Lambda": np.ones((1, draws, 1, 1)),
            "random_levels/0/Alpha": np.ones((1, draws, 1), dtype=int),
        }
    }
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [0.0, 1.0]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="normal",
        study_design=pd.DataFrame(
            {"plot": ["a", "b"], "xcoord": [0.0, 1.0], "ycoord": [0.0, 0.0]}
        ),
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_full",
                "coords": ["xcoord", "ycoord"],
                "alphapw": [[1.0, 1.0]],
            }
        },
    )
    fit = HmscFit(posterior, model=model)
    new = pd.DataFrame(
        {"x": [0.0], "plot": ["new"], "xcoord": [0.5], "ycoord": [0.0]}
    )

    samples = fit.predict_samples(
        new,
        response=False,
        random_effects="known",
        spatial_prediction="conditional",
        rng_seed=41,
    ).reshape(-1)
    repeated = fit.predict_samples(
        new,
        response=False,
        random_effects="known",
        spatial_prediction="conditional",
        rng_seed=41,
    ).reshape(-1)

    train_correlation = np.exp(-1.0)
    cross_correlation = np.exp(-0.5)
    expected_mean = cross_correlation / (1.0 + train_correlation)
    expected_variance = 1.0 - 2.0 * cross_correlation**2 / (1.0 + train_correlation)
    np.testing.assert_allclose(samples, repeated)
    assert abs(samples.mean() - expected_mean) < 0.03
    assert abs(samples.var() - expected_variance) < 0.03
    np.testing.assert_array_equal(fit.alpha_samples(), np.zeros((1, draws, 1), dtype=int))


def test_full_spatial_conditional_prediction_preserves_known_eta():
    posterior = {
        "__arrays__": {
            "Beta": np.zeros((1, 2, 2, 1)),
            "random_levels/0/Eta": np.array([[[[0.2], [0.8]], [[0.3], [0.9]]]]),
            "random_levels/0/Lambda": np.ones((1, 2, 1, 1)),
            "random_levels/0/Alpha": np.ones((1, 2, 1), dtype=int),
        }
    }
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [0.0, 1.0]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="normal",
        study_design=pd.DataFrame(
            {"plot": ["a", "b"], "xcoord": [0.0, 1.0], "ycoord": [0.0, 0.0]}
        ),
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_full",
                "coords": ["xcoord", "ycoord"],
                "alphapw": [[1.0, 1.0]],
            }
        },
    )
    fit = HmscFit(posterior, model=model)

    samples = fit.predict_samples(
        pd.DataFrame({"x": [0.0], "plot": ["b"], "xcoord": [1.0], "ycoord": [0.0]}),
        response=False,
        random_effects="known",
        spatial_prediction="conditional",
        rng_seed=9,
    )

    np.testing.assert_allclose(samples[:, :, 0, 0], [[0.8, 0.9]])


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


def test_species_association_diagnostics_are_sign_invariant():
    posterior = {
        "__metadata__": {"names": {"species": ["sp1", "sp2"]}},
        "__arrays__": {
            "random_levels/0/Lambda": np.array(
                [
                    [
                        [[1.0, 2.0]],
                        [[1.0, 2.0]],
                    ],
                    [
                        [[-1.0, -2.0]],
                        [[-1.0, -2.0]],
                    ],
                ]
            )
        },
    }
    fit = HmscFit(posterior)

    lambda_diag = fit.diagnostics("Lambda")
    association_diag = fit.diagnostics("Associations")
    overview = fit.diagnostics_overview("Associations")

    assert lambda_diag["rhat"].max() > 1.0
    assert list(association_diag.columns) == [
        "random_level",
        "species_1",
        "species_2",
        "mean",
        "sd",
        "rhat",
        "ess",
    ]
    assert association_diag.shape[0] == 1
    assert association_diag.loc[0, "species_1"] == "sp1"
    assert association_diag.loc[0, "species_2"] == "sp2"
    assert association_diag.loc[0, "mean"] == 1.0
    assert association_diag.loc[0, "rhat"] == 1.0
    assert overview["param"] == "Associations"
    assert overview["random_level"] == 0
    assert overview["association"] == "correlation"


def test_aligned_eta_lambda_diagnostics_handle_sign_switching():
    posterior = {
        "__metadata__": {"names": {"species": ["sp1", "sp2"], "random_levels": [{"levels": ["plot_1"]}]}},
        "__arrays__": {
            "random_levels/0/Eta": np.array(
                [
                    [
                        [[0.5]],
                        [[0.5]],
                    ],
                    [
                        [[-0.5]],
                        [[-0.5]],
                    ],
                ]
            ),
            "random_levels/0/Lambda": np.array(
                [
                    [
                        [[1.0, 2.0]],
                        [[1.0, 2.0]],
                    ],
                    [
                        [[-1.0, -2.0]],
                        [[-1.0, -2.0]],
                    ],
                ]
            ),
        },
    }
    fit = HmscFit(posterior)

    raw_lambda = fit.diagnostics("Lambda")
    aligned_lambda = fit.diagnostics("Lambda", align=True)
    aligned_eta = fit.diagnostics("Eta", align=True)
    aligned_summary = fit.lambda_summary(align=True)
    overview = fit.diagnostics_overview("Lambda", align=True)

    assert raw_lambda["rhat"].max() > 1.0
    assert aligned_lambda["rhat"].tolist() == [1.0, 1.0]
    assert aligned_eta["rhat"].tolist() == [1.0]
    assert aligned_summary["mean"].tolist() == [1.0, 2.0]
    assert aligned_summary["aligned"].unique().tolist() == [True]
    assert overview["aligned"] is True


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


def test_eta_and_lambda_summaries_use_random_level_names():
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2], "sp2": [3, 4]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="normal",
        study_design=pd.DataFrame({"plot": ["a", "b"]}),
        random_levels={"plot": {"column": "plot", "type": "iid"}},
    )
    posterior = {
        "__arrays__": {
            "random_levels/0/Eta": np.array(
                [
                    [
                        [[0.0, 1.0], [2.0, 3.0]],
                        [[1.0, 2.0], [3.0, 4.0]],
                    ]
                ]
            ),
            "random_levels/0/Lambda": np.array(
                [
                    [
                        [[0.0, 1.0], [2.0, 3.0]],
                        [[1.0, 2.0], [3.0, 4.0]],
                    ]
                ]
            ),
        }
    }
    fit = HmscFit(posterior, model=model)

    eta_mean = fit.eta_mean()
    lambda_mean = fit.lambda_mean()
    eta_summary = fit.eta_summary(cred_level=0.5)
    lambda_summary = fit.lambda_summary(cred_level=0.5)

    assert list(eta_mean.index) == ["a", "b"]
    assert list(eta_mean.columns) == ["factor_0", "factor_1"]
    assert list(lambda_mean.index) == ["factor_0", "factor_1"]
    assert list(lambda_mean.columns) == ["sp1", "sp2"]
    assert list(eta_summary.columns) == ["random_level", "unit", "factor", "mean", "lower", "upper"]
    assert list(lambda_summary.columns) == ["random_level", "factor", "species", "mean", "lower", "upper"]
    assert eta_summary.loc[(eta_summary["unit"] == "b") & (eta_summary["factor"] == "factor_1"), "mean"].iloc[0] == 3.5
    assert lambda_summary.loc[
        (lambda_summary["factor"] == "factor_1") & (lambda_summary["species"] == "sp2"),
        "mean",
    ].iloc[0] == 3.5


def test_lambda_summary_requires_x_index_for_random_slopes():
    posterior = {
        "__arrays__": {
            "random_levels/0/Lambda": np.ones((1, 1, 1, 2, 2)),
        },
    }
    fit = HmscFit(posterior)

    try:
        fit.lambda_summary()
    except ValueError as exc:
        assert "x_index is required" in str(exc)
    else:
        raise AssertionError("expected x_index validation error")

    summary = fit.lambda_summary(x_index=1)
    assert summary.shape[0] == 2


def test_aligned_eta_diagnostics_default_to_intercept_for_random_slopes():
    posterior = {
        "__arrays__": {
            "random_levels/0/Eta": np.array(
                [
                    [
                        [[0.5]],
                        [[0.5]],
                    ],
                    [
                        [[-0.5]],
                        [[-0.5]],
                    ],
                ]
            ),
            "random_levels/0/Lambda": np.array(
                [
                    [
                        [[[1.0, 0.2], [2.0, 0.4]]],
                        [[[1.0, 0.2], [2.0, 0.4]]],
                    ],
                    [
                        [[[-1.0, -0.2], [-2.0, -0.4]]],
                        [[[-1.0, -0.2], [-2.0, -0.4]]],
                    ],
                ]
            ),
        },
    }
    fit = HmscFit(posterior)

    eta_diagnostics = fit.diagnostics("Eta", align=True)
    eta_overview = fit.diagnostics_overview("Eta", align=True)
    lambda_diagnostics = fit.diagnostics("Lambda", align=True)

    assert eta_diagnostics["rhat"].tolist() == [1.0]
    assert eta_overview["aligned"] is True
    assert lambda_diagnostics["mean"].tolist() == [1.0, 2.0]


def test_eta_and_lambda_diagnostics_use_random_level_labels():
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2], "sp2": [3, 4]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="normal",
        study_design=pd.DataFrame({"plot": ["a", "b"]}),
        random_levels={"plot": {"column": "plot", "type": "iid"}},
    )
    eta = np.array(
        [
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[0.1, 1.1], [2.1, 3.1]],
                [[0.2, 1.2], [2.2, 3.2]],
            ],
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[0.1, 1.1], [2.1, 3.1]],
                [[0.2, 1.2], [2.2, 3.2]],
            ],
        ]
    )
    lam = np.array(
        [
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[0.1, 1.1], [2.1, 3.1]],
                [[0.2, 1.2], [2.2, 3.2]],
            ],
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[0.1, 1.1], [2.1, 3.1]],
                [[0.2, 1.2], [2.2, 3.2]],
            ],
        ]
    )
    fit = HmscFit({"__arrays__": {"random_levels/0/Eta": eta, "random_levels/0/Lambda": lam}}, model=model)

    eta_diag = fit.diagnostics("Eta")
    lambda_diag = fit.diagnostics("Lambda")
    overview = fit.diagnostics_overview("Lambda")

    assert list(eta_diag.columns) == ["random_level", "unit", "factor", "mean", "sd", "rhat", "ess"]
    assert list(lambda_diag.columns) == ["random_level", "factor", "species", "mean", "sd", "rhat", "ess"]
    assert np.isfinite(
        eta_diag.loc[(eta_diag["unit"] == "b") & (eta_diag["factor"] == "factor_1"), "rhat"].iloc[0]
    )
    assert lambda_diag.loc[
        (lambda_diag["factor"] == "factor_1") & (lambda_diag["species"] == "sp2"),
        "ess",
    ].iloc[0] > 0
    assert overview["random_level"] == 0


def test_lambda_diagnostics_requires_x_index_for_random_slopes():
    posterior = {
        "__arrays__": {
            "random_levels/0/Lambda": np.ones((2, 3, 1, 2, 2)),
        },
    }
    fit = HmscFit(posterior)

    try:
        fit.diagnostics("Lambda")
    except ValueError as exc:
        assert "x_index is required" in str(exc)
    else:
        raise AssertionError("expected x_index validation error")

    diagnostics = fit.diagnostics("Lambda", x_index=1)
    assert "x_index" in diagnostics.columns
    assert diagnostics["x_index"].unique().tolist() == [1]
