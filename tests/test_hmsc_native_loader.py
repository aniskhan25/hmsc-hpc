import numpy as np
import pandas as pd

from hmsc.utils.export_native_utils import load_native_params
from pyhmsc import HmscModel


def test_native_loader_builds_fixed_effect_sampler_state(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2], "sp2": [0, 3]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="poisson",
    )
    compiled = model.compile(tmp_path / "run", chains=2)

    dims, data, priors, model_hyper, random_hyper, init_list, n_chains = load_native_params(
        compiled.init_json
    )

    assert n_chains == 2
    assert model_hyper is None
    assert random_hyper == []
    assert dims["nr"] == 0
    assert dims["nc"] == 2
    assert data["distr"].tolist() == [[3, 1], [3, 1]]
    assert data["Pi"].shape == (2, 0)
    assert priors["iUGamma"].shape == (2, 2)
    assert len(init_list) == 2
    np.testing.assert_allclose(init_list[0]["Beta"].numpy(), np.zeros((2, 2)))
    np.testing.assert_allclose(init_list[0]["Xeff"].numpy(), [[1, 0], [1, 1]])


def test_native_loader_builds_iid_random_intercept_state(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2, 0], "sp2": [0, 3, 1]}),
        X=pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        x_formula="~ x",
        distr="poisson",
        study_design=pd.DataFrame({"plot": ["a", "b", "a"]}),
        random_levels={"plot": {"column": "plot", "type": "iid"}},
    )
    compiled = model.compile(tmp_path / "run", chains=2)
    dims, data, _priors, _model_hyper, random_hyper, init_list, _n_chains = load_native_params(
        compiled.init_json
    )
    assert dims["nr"] == 1
    assert dims["np"].tolist() == [2]
    assert data["Pi"].tolist() == [[0], [1], [0]]
    assert random_hyper[0]["sDim"] == 0
    assert init_list[0]["Eta"][0].shape == (2, 1)
    assert init_list[0]["Lambda"][0].shape == (1, 2)
