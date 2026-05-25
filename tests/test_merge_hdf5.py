import json
import subprocess
import sys

import numpy as np
import pytest

from pyhmsc.merge import inspect_chain_directory, merge_hdf5_posteriors
from pyhmsc.posterior import HmscFit


def test_merge_hdf5_posteriors_preserves_metadata(tmp_path):
    import h5py

    metadata = {
        "names": {
            "covariates": ["Intercept", "forest_cover"],
            "species": ["sparrow"],
        },
        "formula": {"X": "~ forest_cover"},
        "distribution": "poisson",
    }
    first = tmp_path / "chain_0.h5"
    second = tmp_path / "chain_1.h5"
    for idx, path in enumerate([first, second]):
        with h5py.File(path, "w") as handle:
            handle.attrs["nChains"] = 1
            handle.attrs["time"] = float(idx + 1)
            handle.attrs["pyhmsc_metadata"] = json.dumps(metadata)
            handle.create_dataset("Beta", data=np.ones((1, 2, 2, 1)) * idx)
            handle.create_dataset("Gamma", data=np.ones((1, 2, 1, 1)))
            handle.create_dataset("iV", data=np.ones((1, 2, 2, 2)))
            handle.create_dataset("rhoInd", data=np.ones((1, 2, 2)))
            handle.create_dataset("sigma", data=np.ones((1, 2, 1)))
            level = handle.create_group("random_levels").create_group("0")
            level.create_dataset("Eta", data=np.ones((1, 2, 2, 1)) * idx)
            level.create_dataset("Lambda", data=np.ones((1, 2, 1, 1)))

    output = merge_hdf5_posteriors([first, second], tmp_path / "posterior.h5")
    fit = HmscFit.from_file(output)

    assert fit.beta_samples().shape == (2, 2, 2, 1)
    assert list(fit.beta_mean().index) == ["Intercept", "forest_cover"]
    assert list(fit.beta_mean().columns) == ["sparrow"]
    np.testing.assert_allclose(fit.beta_mean().to_numpy(), [[0.5], [0.5]])

    with h5py.File(output, "r") as handle:
        assert handle.attrs["nChains"] == 2
        assert json.loads(handle.attrs["pyhmsc_metadata"]) == metadata


def test_cli_merge_hdf5_posteriors(tmp_path):
    import h5py

    paths = []
    for idx in range(2):
        path = tmp_path / f"posterior_chain_{idx}.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["nChains"] = 1
            handle.create_dataset("Beta", data=np.ones((1, 1, 1, 1)) * idx)
        paths.append(path)

    output = tmp_path / "merged.h5"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "merge",
            str(paths[0]),
            str(paths[1]),
            "--output",
            str(output),
            "--expected-chains",
            "0",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == str(output)
    assert HmscFit.from_file(output).beta_samples().shape == (2, 1, 1, 1)


def test_merge_hdf5_requires_matching_dataset_keys(tmp_path):
    import h5py

    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    with h5py.File(first, "w") as handle:
        handle.create_dataset("Beta", data=np.ones((1, 1, 1, 1)))
    with h5py.File(second, "w") as handle:
        handle.create_dataset("Gamma", data=np.ones((1, 1, 1, 1)))

    with pytest.raises(ValueError, match="datasets do not match"):
        merge_hdf5_posteriors([first, second], tmp_path / "merged.h5")


def test_chain_status_detects_missing_and_bad_files(tmp_path):
    import h5py

    with h5py.File(tmp_path / "posterior_chain_0.h5", "w") as handle:
        handle.create_dataset("Beta", data=np.ones((1, 5, 1, 1)))
    with h5py.File(tmp_path / "posterior_chain_2.h5", "w") as handle:
        handle.create_dataset("Gamma", data=np.ones((1, 5, 1, 1)))

    statuses = inspect_chain_directory(tmp_path, expected_chains=[0, 1, 2], expected_draws=5)
    by_chain = {status.chain: status for status in statuses}
    assert by_chain[0].status == "passed"
    assert by_chain[1].status == "missing"
    assert by_chain[2].status == "failed"
    assert "missing Beta" in by_chain[2].message


def test_cli_chain_status_prints_rerun_command(tmp_path):
    import h5py

    with h5py.File(tmp_path / "posterior_chain_0.h5", "w") as handle:
        handle.create_dataset("Beta", data=np.ones((1, 5, 1, 1)))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "chain-status",
            str(tmp_path),
            "--expected-chains",
            "0",
            "1",
            "--run-name",
            "demo",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "1 missing" in result.stdout
    assert "RUN_NAME=demo sbatch --array=1" in result.stdout


def test_merge_hdf5_rejects_missing_expected_chain(tmp_path):
    import h5py

    path = tmp_path / "posterior_chain_0.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Beta", data=np.ones((1, 1, 1, 1)))

    with pytest.raises(ValueError, match="Missing expected chain outputs"):
        merge_hdf5_posteriors([path], tmp_path / "merged.h5", expected_chains=[0, 1])
