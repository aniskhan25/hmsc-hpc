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
