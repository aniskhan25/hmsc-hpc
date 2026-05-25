import numpy as np
import pandas as pd

from pyhmsc import HmscModel
from pyhmsc.posterior import HmscFit


def test_hdf5_posterior_roundtrip(tmp_path):
    import h5py

    path = tmp_path / "posterior.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Beta", data=np.ones((2, 3, 2, 1)))
        handle.create_dataset("Gamma", data=np.ones((2, 3, 2, 1)) * 2)
        handle.create_dataset("sigma", data=np.ones((2, 3, 1)) * 0.5)
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset("Eta", data=np.ones((2, 3, 2, 1)))
        level.create_dataset("Lambda", data=np.ones((2, 3, 1, 1)))
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"covariates":["Intercept","x"],"species":["sp1"]},'
            '"formula":{"X":"~ x"},"distribution":"poisson"}'
        )
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
    )
    fit = HmscFit.from_file(path, model=model)
    assert fit.beta_samples().shape == (2, 3, 2, 1)
    assert fit.metadata["distribution"] == "poisson"
    np.testing.assert_allclose(fit.beta_mean().to_numpy(), np.ones((2, 1)))
    assert list(fit.beta_mean().index) == ["Intercept", "x"]
    assert list(fit.beta_mean().columns) == ["sp1"]
    np.testing.assert_allclose(fit.gamma_mean().to_numpy(), np.ones((2, 1)) * 2)
    np.testing.assert_allclose(fit.sigma_mean().to_numpy(), [0.5])
    assert fit.eta_mean(0).shape == (2, 1)
    assert fit.lambda_mean(0).shape == (1, 1)
