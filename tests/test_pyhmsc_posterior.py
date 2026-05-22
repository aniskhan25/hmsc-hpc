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
