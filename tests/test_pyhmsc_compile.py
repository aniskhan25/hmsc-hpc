import json

import numpy as np
import pandas as pd

from pyhmsc import HmscModel
from pyhmsc.serialization import read_compiled_model


def test_compile_writes_json_and_hdf5(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2], "sp2": [3, 4]}, index=["a", "b"]),
        X=pd.DataFrame({"x": [0.0, 2.0]}, index=["a", "b"]),
        x_formula="~ x",
        distr="poisson",
    )

    compiled = model.compile(tmp_path / "run", chains=2)

    metadata = json.loads(compiled.init_json.read_text(encoding="utf-8"))
    assert metadata["format"] == "pyhmsc-json-hdf5"
    assert metadata["dimensions"] == {
        "n_sites": 2,
        "n_species": 2,
        "n_covariates": 2,
        "n_traits": 1,
        "n_chains": 2,
    }
    assert metadata["arrays"]["Y"] == "init_arrays.h5:/Y"
    assert metadata["preprocessing"]["XScalePar"]["columns"] == ["Intercept", "x"]
    np.testing.assert_allclose(metadata["preprocessing"]["XScalePar"]["mean"], [0.0, 1.0])
    np.testing.assert_allclose(metadata["preprocessing"]["XScalePar"]["sd"], [1.0, np.sqrt(2.0)])

    read_metadata, arrays = read_compiled_model(compiled.init_json)
    assert read_metadata["schema_version"] == "0.1"
    np.testing.assert_allclose(arrays["Y"], [[1, 3], [2, 4]])
    np.testing.assert_allclose(arrays["X"], [[1, -np.sqrt(0.5)], [1, np.sqrt(0.5)]])
    assert arrays["Beta_init"].shape == (2, 2, 2)


def test_compile_leaves_binary_indicator_columns_unscaled(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2, 3], "sp2": [3, 4, 5]}, index=["a", "b", "c"]),
        X=pd.DataFrame({"wetland": [0.0, 1.0, 1.0]}, index=["a", "b", "c"]),
        x_formula="~ wetland",
        distr="poisson",
    )

    compiled = model.compile(tmp_path / "run", chains=1)
    metadata, arrays = read_compiled_model(compiled.init_json)

    np.testing.assert_allclose(arrays["X"], [[1, 0], [1, 1], [1, 1]])
    np.testing.assert_allclose(metadata["preprocessing"]["XScalePar"]["mean"], [0.0, 0.0])
    np.testing.assert_allclose(metadata["preprocessing"]["XScalePar"]["sd"], [1.0, 1.0])
