import numpy as np
import pandas as pd

from pyhmsc import HmscModel
from pyhmsc.posterior import HmscFit


def test_hdf5_posterior_roundtrip(tmp_path):
    import h5py

    path = tmp_path / "posterior.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Beta", data=np.ones((2, 3, 2, 1)))
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
    )
    fit = HmscFit.from_file(path, model=model)
    assert fit.beta_samples().shape == (2, 3, 2, 1)
    np.testing.assert_allclose(fit.beta_mean().to_numpy(), np.ones((2, 1)))
