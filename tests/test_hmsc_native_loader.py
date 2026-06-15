import numpy as np
import pandas as pd
import pytest

from hmsc.utils.export_native_utils import _spatial_nngp_params, load_native_params
from hmsc.run_gibbs_sampler import validate_sampler_supported_params
from pyhmsc import HmscModel
from pyhmsc.compiler import _nngp_neighbors
from pyhmsc.validation import validate_compiled_native_model


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


def test_native_loader_builds_trait_state(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2], "sp2": [0, 3]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="poisson",
        traits=pd.DataFrame({"body_size": [1.0, 2.0]}, index=["sp1", "sp2"]),
        trait_formula="~ body_size",
    )
    compiled = model.compile(tmp_path / "traits", chains=1)
    dims, data, priors, _model_hyper, _random_hyper, init_list, _n_chains = load_native_params(
        compiled.init_json
    )
    assert dims["nt"] == 2
    assert data["T"].shape == (2, 2)
    assert priors["mGamma"].shape == (4,)
    assert priors["UGamma"].shape == (4, 4)
    assert init_list[0]["Gamma"].shape == (2, 2)


def test_native_loader_builds_phylogeny_state(tmp_path):
    phylo = pd.DataFrame(
        [[1.0, 0.2], [0.2, 1.0]],
        index=["sp1", "sp2"],
        columns=["sp1", "sp2"],
    )
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2], "sp2": [0, 3]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
        distr="poisson",
        phylo_cov=phylo,
    )
    compiled = model.compile(tmp_path / "phylo", chains=1)
    _dims, data, priors, _model_hyper, _random_hyper, init_list, _n_chains = load_native_params(
        compiled.init_json
    )
    assert data["C"].shape == (2, 2)
    assert data["eC"].shape == (2,)
    assert data["VC"].shape == (2, 2)
    assert priors["rhopw"].shape[1] == 2
    assert init_list[0]["rhoInd"].shape == (2,)


def test_native_loader_builds_full_spatial_random_level_state(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2, 0], "sp2": [0, 3, 1]}),
        X=pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        x_formula="~ x",
        distr="poisson",
        study_design=pd.DataFrame(
            {"plot": ["a", "b", "c"], "xcoord": [0.0, 1.0, 0.0], "ycoord": [0.0, 0.0, 1.0]}
        ),
        random_levels={"plot": {"column": "plot", "type": "spatial_full", "coords": ["xcoord", "ycoord"]}},
    )
    compiled = model.compile(tmp_path / "spatial", chains=1)
    dims, data, _priors, _model_hyper, random_hyper, init_list, _n_chains = load_native_params(
        compiled.init_json
    )
    assert dims["nr"] == 1
    assert data["Pi"].tolist() == [[0], [1], [2]]
    assert random_hyper[0]["sDim"] == 2
    assert random_hyper[0]["spatialMethod"] == "Full"
    assert random_hyper[0]["iWg"].shape == (1, 3, 3)
    assert init_list[0]["Eta"][0].shape == (3, 1)


def test_native_loader_builds_spatial_gpp_random_level_state(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2, 0, 1], "sp2": [0, 3, 1, 2]}),
        X=pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        x_formula="~ x",
        distr="poisson",
        study_design=pd.DataFrame(
            {
                "plot": ["a", "b", "c", "d"],
                "xcoord": [0.0, 1.0, 0.0, 1.0],
                "ycoord": [0.0, 0.0, 1.0, 1.0],
            }
        ),
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_gpp",
                "coords": ["xcoord", "ycoord"],
                "n_knots": 2,
            }
        },
    )
    compiled = model.compile(tmp_path / "spatial-gpp", chains=1)
    dims, data, _priors, _model_hyper, random_hyper, init_list, _n_chains = load_native_params(
        compiled.init_json
    )
    assert dims["nr"] == 1
    assert data["Pi"].tolist() == [[0], [1], [2], [3]]
    assert random_hyper[0]["sDim"] == 2
    assert random_hyper[0]["spatialMethod"] == "GPP"
    assert random_hyper[0]["nK"] == 2
    assert random_hyper[0]["idDg"].shape == (1, 4)
    assert random_hyper[0]["Fg"].shape == (1, 2, 2)
    assert init_list[0]["Eta"][0].shape == (4, 1)


def test_native_loader_builds_spatial_nngp_random_level_state(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2, 0, 1], "sp2": [0, 3, 1, 2]}),
        X=pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        x_formula="~ x",
        distr="poisson",
        study_design=pd.DataFrame(
            {
                "plot": ["a", "b", "c", "d"],
                "xcoord": [0.0, 1.0, 0.0, 1.0],
                "ycoord": [0.0, 0.0, 1.0, 1.0],
            }
        ),
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_nngp",
                "coords": ["xcoord", "ycoord"],
                "n_neighbors": 2,
            }
        },
    )
    compiled = model.compile(tmp_path / "spatial-nngp", chains=1)
    dims, data, _priors, _model_hyper, random_hyper, init_list, _n_chains = load_native_params(
        compiled.init_json
    )
    assert dims["nr"] == 1
    assert data["Pi"].tolist() == [[0], [1], [2], [3]]
    assert random_hyper[0]["sDim"] == 2
    assert random_hyper[0]["spatialMethod"] == "NNGP"
    assert len(random_hyper[0]["iWList_csr"]) == 1
    assert random_hyper[0]["iWList_csr"][0].shape == (4, 4)
    assert len(random_hyper[0]["RiWList"]) == 1
    assert random_hyper[0]["detWg"].shape == (1,)
    assert init_list[0]["Eta"][0].shape == (4, 1)

    results = validate_compiled_native_model(compiled.init_json)
    by_name = {result.name: result for result in results}
    assert by_name["native_sampler_supported"].passed
    validate_sampler_supported_params(random_hyper)


def test_nngp_neighbor_builder_uses_previous_nearest_neighbors():
    coords = np.array([[0.0], [1.0], [3.0], [6.0]])
    dist = np.abs(coords - coords.T)

    indices, local_dists = _nngp_neighbors(dist, n_neighbors=2)

    assert indices.tolist() == [[-1, -1], [0, -1], [1, 0], [2, 1]]
    np.testing.assert_allclose(local_dists[1, :2, :2], [[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(local_dists[2, :3, :3], [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    np.testing.assert_allclose(local_dists[3, :3, :3], [[0.0, 2.0, 3.0], [2.0, 0.0, 5.0], [3.0, 5.0, 0.0]])


def test_nngp_precision_matches_full_precision_with_all_previous_neighbors():
    coords = np.array([[0.0], [0.3], [0.8], [1.7]])
    dist = np.abs(coords - coords.T)
    alpha = 0.45
    indices, local_dists = _nngp_neighbors(dist, n_neighbors=3)

    params = _spatial_nngp_params(indices, local_dists, np.asarray([[alpha, 1.0]]), np.float64)
    nngp_precision = params["iWList_csr"][0].toarray()
    full_covariance = np.exp(-dist / alpha)
    full_precision = np.linalg.inv(full_covariance)

    np.testing.assert_allclose(nngp_precision, full_precision, rtol=1e-10, atol=1e-10)
    assert np.linalg.eigvalsh(nngp_precision).min() > 0


def test_native_loader_builds_random_slope_state(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2, 0], "sp2": [0, 3, 1]}),
        X=pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        x_formula="~ x",
        distr="poisson",
        study_design=pd.DataFrame({"plot": ["a", "b", "a"], "elevation": [10.0, 20.0, 10.0]}),
        random_levels={"plot": {"column": "plot", "type": "iid", "x_formula": "~ elevation"}},
    )
    compiled = model.compile(tmp_path / "random-slope", chains=1)
    dims, _data, _priors, _model_hyper, random_hyper, init_list, _n_chains = load_native_params(
        compiled.init_json
    )
    assert dims["nr"] == 1
    assert random_hyper[0]["xDim"] == 2
    assert random_hyper[0]["xMat"].shape == (2, 2)
    assert init_list[0]["Lambda"][0].shape == (1, 2, 2)

    results = validate_compiled_native_model(compiled.init_json)
    by_name = {result.name: result for result in results}
    assert by_name["native_sampler_supported"].passed
    validate_sampler_supported_params(random_hyper)


def test_native_loader_builds_full_spatial_random_slope_state(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2, 0]}),
        X=pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        x_formula="~ x",
        distr="poisson",
        study_design=pd.DataFrame(
            {
                "plot": ["a", "b", "c"],
                "elevation": [10.0, 20.0, 30.0],
                "xcoord": [0.0, 1.0, 0.0],
                "ycoord": [0.0, 0.0, 1.0],
            }
        ),
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_full",
                "coords": ["xcoord", "ycoord"],
                "x_formula": "~ elevation",
            }
        },
    )
    compiled = model.compile(tmp_path / "spatial-random-slope", chains=1)
    _dims, _data, _priors, _model_hyper, random_hyper, _init_list, _n_chains = load_native_params(
        compiled.init_json
    )
    assert random_hyper[0]["spatialMethod"] == "Full"
    assert random_hyper[0]["xDim"] == 2
    assert random_hyper[0]["xMat"].shape == (3, 2)
    results = validate_compiled_native_model(compiled.init_json)
    by_name = {result.name: result for result in results}
    assert by_name["native_sampler_supported"].passed
    validate_sampler_supported_params(random_hyper)


def test_native_loader_builds_gpp_spatial_random_slope_state(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2, 0]}),
        X=pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        x_formula="~ x",
        distr="poisson",
        study_design=pd.DataFrame(
            {
                "plot": ["a", "b", "c"],
                "elevation": [10.0, 20.0, 30.0],
                "xcoord": [0.0, 1.0, 0.0],
                "ycoord": [0.0, 0.0, 1.0],
            }
        ),
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_gpp",
                "coords": ["xcoord", "ycoord"],
                "n_knots": 2,
                "x_formula": "~ elevation",
            }
        },
    )
    compiled = model.compile(tmp_path / "gpp-spatial-random-slope", chains=1)
    _dims, _data, _priors, _model_hyper, random_hyper, _init_list, _n_chains = load_native_params(
        compiled.init_json
    )
    assert random_hyper[0]["spatialMethod"] == "GPP"
    assert random_hyper[0]["xDim"] == 2
    assert random_hyper[0]["xMat"].shape == (3, 2)
    results = validate_compiled_native_model(compiled.init_json)
    by_name = {result.name: result for result in results}
    assert by_name["native_sampler_supported"].passed
    validate_sampler_supported_params(random_hyper)


def test_native_loader_builds_nngp_spatial_random_slope_state(tmp_path):
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2, 0]}),
        X=pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        x_formula="~ x",
        distr="poisson",
        study_design=pd.DataFrame(
            {
                "plot": ["a", "b", "c"],
                "elevation": [10.0, 20.0, 30.0],
                "xcoord": [0.0, 1.0, 0.0],
                "ycoord": [0.0, 0.0, 1.0],
            }
        ),
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_nngp",
                "coords": ["xcoord", "ycoord"],
                "n_neighbors": 2,
                "x_formula": "~ elevation",
            }
        },
    )
    compiled = model.compile(tmp_path / "nngp-spatial-random-slope", chains=1)
    _dims, _data, _priors, _model_hyper, random_hyper, _init_list, _n_chains = load_native_params(
        compiled.init_json
    )
    assert random_hyper[0]["spatialMethod"] == "NNGP"
    assert random_hyper[0]["xDim"] == 2
    assert random_hyper[0]["xMat"].shape == (3, 2)
    results = validate_compiled_native_model(compiled.init_json)
    by_name = {result.name: result for result in results}
    assert by_name["native_sampler_supported"].passed
    validate_sampler_supported_params(random_hyper)
