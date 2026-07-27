import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import hmsc.run_gibbs_sampler as runner
from hmsc.utils.export_hdf5_utils import save_chains_postList_to_hdf5
from hmsc.utils.import_utils import load_random_level_hyperparams
from hmsc.utils.export_rds_utils import (
    _extract_json_string,
    load_model_from_rds,
    save_chains_postList_to_rds,
)


def test_load_params_dispatches_json_to_native_loader(monkeypatch, tmp_path):
    calls = []
    expected = ("dims", "data", "priors", None, [], ["init"], 1)

    def fake_load_native_params(file_path, dtype):
        calls.append(("native", file_path, dtype))
        return expected

    def fake_load_model_from_rds(file_path):
        calls.append(("rds", file_path))
        raise AssertionError("RDS loader should not be used for native JSON inputs")

    monkeypatch.setattr(runner, "load_native_params", fake_load_native_params)
    monkeypatch.setattr(runner, "load_model_from_rds", fake_load_model_from_rds)

    result = runner.load_params(str(tmp_path / "init.json"), np.float32)

    assert result == expected
    assert calls == [("native", str(tmp_path / "init.json"), np.float32)]


def test_load_params_dispatches_rds_to_legacy_loader(monkeypatch, tmp_path):
    calls = []
    init_par_list = {"legacy": "init"}
    data_par_list = {"legacy": "data"}
    hmsc_model = {"legacy": "model"}
    hmsc_import = {
        "hM": hmsc_model,
        "initParList": init_par_list,
        "dataParList": data_par_list,
        "nChains": [2],
    }
    expected = ("dims", "data", "priors", None, "random", "init", 2)

    def fake_load_native_params(file_path, dtype):
        calls.append(("native", file_path, dtype))
        raise AssertionError("Native loader should not be used for RDS compatibility inputs")

    def fake_load_model_from_rds(file_path):
        calls.append(("rds", file_path))
        return hmsc_import, hmsc_model

    def fake_load_model_dims(model):
        calls.append(("dims", model))
        return expected[0]

    def fake_load_model_data(model, init, dtype):
        calls.append(("data", model, init, dtype))
        return expected[1]

    def fake_load_prior_hyperparams(model, dtype):
        calls.append(("priors", model, dtype))
        return expected[2]

    def fake_load_random_level_hyperparams(model, data, dtype):
        calls.append(("random", model, data, dtype))
        return expected[4]

    def fake_init_params(init, data, dims, random, dtype):
        calls.append(("init", init, data, dims, random, dtype))
        return expected[5]

    monkeypatch.setattr(runner, "load_native_params", fake_load_native_params)
    monkeypatch.setattr(runner, "load_model_from_rds", fake_load_model_from_rds)
    monkeypatch.setattr(runner, "load_model_dims", fake_load_model_dims)
    monkeypatch.setattr(runner, "load_model_data", fake_load_model_data)
    monkeypatch.setattr(runner, "load_prior_hyperparams", fake_load_prior_hyperparams)
    monkeypatch.setattr(runner, "load_random_level_hyperparams", fake_load_random_level_hyperparams)
    monkeypatch.setattr(runner, "init_params", fake_init_params)

    result = runner.load_params(str(tmp_path / "init_file.rds"), np.float32)

    assert result == expected
    assert calls == [
        ("rds", str(tmp_path / "init_file.rds")),
        ("dims", hmsc_model),
        ("data", hmsc_model, init_par_list, np.float32),
        ("priors", hmsc_model, np.float32),
        ("random", hmsc_model, data_par_list, np.float32),
        ("init", init_par_list, expected[1], expected[0], expected[4], np.float32),
    ]


def test_load_native_metadata_ignores_legacy_rds_inputs(tmp_path):
    missing_rds = tmp_path / "does_not_need_to_exist.rds"

    assert runner.load_native_metadata(str(missing_rds)) is None


def test_run_gibbs_sampler_writes_rds_for_legacy_output(monkeypatch, tmp_path):
    calls = []

    def fake_load_params(file_path, dtype):
        calls.append(("load_params", file_path, dtype))
        model_dims = {"ny": 1}
        model_data = {}
        prior_hyperparams = {}
        random_hyperparams = []
        init_list = [{"chain": 0}]
        return model_dims, model_data, prior_hyperparams, None, random_hyperparams, init_list, 1

    class FakeGibbsSampler:
        def __init__(self, model_dims, model_data, prior_hyperparams, random_hyperparams):
            calls.append(("gibbs_init", model_dims, model_data, prior_hyperparams, random_hyperparams))

        def sampling_routine(self, init, **kwargs):
            calls.append(("sample", init, int(kwargs["num_samples"].numpy())))
            n_samples = int(kwargs["num_samples"].numpy())
            return _fake_samples(n_samples)

    def fake_save_rds(post_list, output_path, n_chains, elapsed_time, flag_save_eta):
        calls.append(("save_rds", output_path, n_chains, flag_save_eta, len(post_list), len(post_list[0])))

    def fail_json(*_args, **_kwargs):
        raise AssertionError("JSON saver should not be used for RDS compatibility output")

    def fail_hdf5(*_args, **_kwargs):
        raise AssertionError("HDF5 saver should not be used for RDS compatibility output")

    def fail_zarr(*_args, **_kwargs):
        raise AssertionError("Zarr saver should not be used for RDS compatibility output")

    monkeypatch.setattr(runner, "load_params", fake_load_params)
    monkeypatch.setattr(runner, "GibbsSampler", FakeGibbsSampler)
    monkeypatch.setattr(runner, "save_chains_postList_to_rds", fake_save_rds)
    monkeypatch.setattr(runner, "save_chains_postList_to_json", fail_json)
    monkeypatch.setattr(runner, "save_chains_postList_to_hdf5", fail_hdf5)
    monkeypatch.setattr(runner, "save_chains_postList_to_zarr", fail_zarr)

    runner.run_gibbs_sampler(
        num_samples=2,
        sample_thining=1,
        sample_burnin=0,
        verbose=0,
        init_obj_file_path=str(tmp_path / "init_file.rds"),
        postList_file_path=str(tmp_path / "post_file.rds"),
        hmc_thin=0,
        dtype=np.float32,
    )

    assert calls[0] == ("load_params", str(tmp_path / "init_file.rds"), np.float32)
    assert calls[-1] == ("save_rds", str(tmp_path / "post_file.rds"), 1, True, 1, 2)


def test_rds_loader_reads_legacy_json_payload(tmp_path):
    pyreadr = pytest.importorskip("pyreadr")
    payload = {
        "hM": {"ny": 2, "ns": 3},
        "nChains": [1],
        "initParList": [{"Beta": [[0.0]]}],
        "dataParList": {},
    }
    rds_path = tmp_path / "init_file.rds"
    pyreadr.write_rds(str(rds_path), pd.DataFrame([[runner.json.dumps(payload)]]), compress="gzip")

    hmsc_import, hmsc_model = load_model_from_rds(str(rds_path))

    assert hmsc_import == payload
    assert hmsc_model == payload["hM"]


def test_rds_saver_writes_legacy_json_payload(tmp_path):
    pyreadr = pytest.importorskip("pyreadr")
    output_path = tmp_path / "post_file.rds"

    save_chains_postList_to_rds(
        [[_fake_snapshot()]],
        str(output_path),
        nChains=1,
        elapsedTime=12.5,
        flag_save_eta=True,
    )

    payload = runner.json.loads(_extract_json_string(pyreadr.read_r(str(output_path))))

    assert payload["time"] == 12.5
    assert payload["0"]["0"]["Beta"] == [[0.0]]
    assert payload["0"]["0"]["rhoInd"] == [1]
    assert payload["0"]["0"]["Alpha"]["0"] == [1]


def test_hdf5_saver_pads_variable_random_level_factor_shapes(tmp_path):
    h5py = pytest.importorskip("h5py")
    output_path = tmp_path / "posterior.h5"
    first = _fake_snapshot()
    first["Psi"] = [tf.ones((1, 1), dtype=tf.float64)]
    first["Delta"] = [tf.ones((1, 1), dtype=tf.float64)]
    second = _fake_snapshot()
    second["Eta"] = [tf.ones((2, 2), dtype=tf.float64)]
    second["Lambda"] = [tf.ones((2, 1), dtype=tf.float64)]
    second["Psi"] = [tf.ones((2, 1), dtype=tf.float64)]
    second["Delta"] = [tf.ones((2, 1), dtype=tf.float64)]
    second["AlphaInd"] = [tf.ones((2,), dtype=tf.int32)]

    save_chains_postList_to_hdf5([[first, second]], str(output_path), nChains=1)

    with h5py.File(output_path, "r") as handle:
        eta = handle["random_levels/0/Eta"][:]
        alpha = handle["random_levels/0/Alpha"][:]

    assert eta.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(eta[0, 0], [[0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(eta[0, 1], [[1.0, 1.0], [1.0, 1.0]])
    assert alpha.shape == (1, 2, 2)
    np.testing.assert_array_equal(alpha[0, 0], [1, 1])
    np.testing.assert_array_equal(alpha[0, 1], [2, 2])


def test_legacy_gpp_random_level_hyperparams_are_jitter_stabilized():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    knots = coords[[0, 3]]
    dist12 = np.sqrt(((coords[:, None, :] - knots[None, :, :]) ** 2).sum(axis=-1))
    dist22 = np.sqrt(((knots[:, None, :] - knots[None, :, :]) ** 2).sum(axis=-1))
    alpha_grid = np.column_stack(
        [
            np.sqrt(2.0) * np.arange(101, dtype=float) / 100.0,
            np.concatenate([[0.5], np.repeat(0.005, 100)]),
        ]
    )
    hmsc_model = {
        "nr": [1],
        "np": [4],
        "rL": {
            "plot": {
                "nu": [3.0],
                "a1": [2.0],
                "b1": [1.0],
                "a2": [3.0],
                "b2": [1.0],
                "nfMin": [1],
                "nfMax": [4],
                "sDim": [2],
                "xDim": [0],
                "spatialMethod": ["GPP"],
                "alphapw": alpha_grid.tolist(),
            }
        },
    }
    data_par = {
        "rLPar": [
            {
                "nKnots": [2],
                "distMat12": dist12.reshape(-1).tolist(),
                "distMat22": dist22.reshape(-1).tolist(),
            }
        ]
    }

    random_hyper = load_random_level_hyperparams(hmsc_model, data_par)

    assert random_hyper[0]["idDg"].shape == (101, 4)
    assert random_hyper[0]["Fg"].shape == (101, 2, 2)
    assert np.isfinite(random_hyper[0]["idDg"].numpy()).all()
    assert np.isfinite(random_hyper[0]["detDg"].numpy()).all()


def _fake_samples(n_samples):
    return {
        "Beta": tf.zeros((n_samples, 1, 1), dtype=tf.float64),
        "BetaSel": [tf.zeros((n_samples, 1), dtype=tf.float64)],
        "Gamma": tf.zeros((n_samples, 1, 1), dtype=tf.float64),
        "iV": tf.zeros((n_samples, 1, 1), dtype=tf.float64),
        "rhoInd": tf.zeros((n_samples, 1), dtype=tf.int32),
        "sigma": tf.ones((n_samples, 1), dtype=tf.float64),
        "Lambda": [tf.zeros((n_samples, 1, 1), dtype=tf.float64)],
        "Psi": [tf.ones((n_samples, 1), dtype=tf.float64)],
        "Delta": [tf.ones((n_samples, 1), dtype=tf.float64)],
        "Eta": [tf.zeros((n_samples, 1, 1), dtype=tf.float64)],
        "AlphaInd": [tf.zeros((n_samples, 1), dtype=tf.int32)],
    }


def _fake_snapshot():
    return {
        "Beta": tf.zeros((1, 1), dtype=tf.float64),
        "BetaSel": [tf.zeros((1,), dtype=tf.float64)],
        "Gamma": tf.zeros((1, 1), dtype=tf.float64),
        "iV": tf.zeros((1, 1), dtype=tf.float64),
        "rhoInd": tf.zeros((1,), dtype=tf.int32),
        "sigma": tf.ones((1,), dtype=tf.float64),
        "Lambda": [tf.zeros((1, 1), dtype=tf.float64)],
        "Psi": [tf.ones((1,), dtype=tf.float64)],
        "Delta": [tf.ones((1,), dtype=tf.float64)],
        "Eta": [tf.zeros((1, 1), dtype=tf.float64)],
        "AlphaInd": [tf.zeros((1,), dtype=tf.int32)],
        "wRRR": None,
        "PsiRRR": None,
        "DeltaRRR": None,
    }
