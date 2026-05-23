import numpy as np
import pandas as pd

from hmsc.utils.export_native_utils import load_native_params
from pyhmsc import HmscModel, simulate_fixed_effect_data


def test_simulated_fixed_effect_signs_are_encoded():
    beta = np.array([[0.0, 0.0], [0.8, -0.8]])
    Y, X, truth = simulate_fixed_effect_data(n_sites=20, beta=beta, distr="normal", seed=10)
    model = HmscModel(Y=Y, X=X, x_formula="~ x", distr="normal")
    assert truth.loc["x", "sp1"] > 0
    assert truth.loc["x", "sp2"] < 0
    assert model.Y.shape == (20, 2)


def test_simulated_traits_iid_phylo_and_spatial_compile_to_sampler_state(tmp_path):
    Y, X, _truth = simulate_fixed_effect_data(n_sites=6, distr="poisson", seed=11)
    traits = pd.DataFrame({"body": [1.0, 2.0]}, index=Y.columns)
    phylo = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=Y.columns, columns=Y.columns)
    study = pd.DataFrame(
        {
            "plot": ["a", "b", "c", "a", "b", "c"],
            "xcoord": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "ycoord": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        }
    )
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        traits=traits,
        trait_formula="~ body",
        phylo_cov=phylo,
        study_design=study,
        random_levels={"plot": {"column": "plot", "type": "spatial_full", "coords": ["xcoord", "ycoord"]}},
    )
    compiled = model.compile(tmp_path / "simulated", chains=1)
    dims, data, _priors, _model_hyper, random_hyper, init_list, _n_chains = load_native_params(
        compiled.init_json
    )
    assert dims["nt"] == 2
    assert data["C"].shape == (2, 2)
    assert random_hyper[0]["spatialMethod"] == "Full"
    assert init_list[0]["Eta"][0].shape == (3, 1)
