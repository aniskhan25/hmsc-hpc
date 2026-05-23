import pandas as pd
import pytest

from hmsc.utils.export_native_utils import load_native_params
from pyhmsc import HmscModel


def test_newick_tree_compiles_to_phylo_cov(tmp_path):
    pytest.importorskip("Bio")
    model = HmscModel(
        Y=pd.DataFrame({"sparrow": [1, 2], "owl": [0, 1], "woodpecker": [2, 1]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="poisson",
        phylo_tree="tests/fixtures/fixed_effect/tree.nwk",
    )
    compiled = model.compile(tmp_path / "tree", chains=1)
    _dims, data, _priors, _model_hyper, _random_hyper, _init_list, _n_chains = load_native_params(
        compiled.init_json
    )
    assert data["C"].shape == (3, 3)
    assert data["C"][0, 1] > 0
