import numpy as np
import pandas as pd
import pytest

from pyhmsc import HmscModel
from pyhmsc.posterior import HmscFit


def test_zarr_posterior_roundtrip_when_available(tmp_path):
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "posterior.zarr"
    root = zarr.open_group(str(path), mode="w")
    if hasattr(root, "create_array"):
        root.create_array("Beta", data=np.ones((1, 2, 2, 1)), overwrite=True)
    else:
        root.create_dataset("Beta", data=np.ones((1, 2, 2, 1)), overwrite=True)
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
    )
    fit = HmscFit.from_file(path, model=model)
    assert fit.beta_mean().shape == (2, 1)
