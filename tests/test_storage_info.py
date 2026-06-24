import json
import subprocess
import sys

import numpy as np
import pytest

from pyhmsc.merge import inspect_chain_file
from pyhmsc.storage import inspect_posterior_storage


def test_hdf5_storage_info_reports_shapes_metadata_and_bytes(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    metadata = {"names": {"species": ["sp1"], "covariates": ["Intercept", "x"]}}
    with h5py.File(posterior, "w") as handle:
        handle.attrs["pyhmsc_metadata"] = json.dumps(metadata)
        handle.attrs["nChains"] = 2
        handle.create_dataset("Beta", data=np.ones((2, 3, 2, 1)))
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset("Eta", data=np.ones((2, 3, 4, 1)))

    info = inspect_posterior_storage(posterior)
    by_name = {dataset.name: dataset for dataset in info.datasets}

    assert info.format == "hdf5"
    assert info.metadata_present
    assert info.n_chains == 2
    assert info.n_draws == 3
    assert by_name["Beta"].shape == (2, 3, 2, 1)
    assert by_name["random_levels/0/Eta"].shape == (2, 3, 4, 1)
    assert info.total_nbytes == by_name["Beta"].nbytes + by_name["random_levels/0/Eta"].nbytes
    assert "metadata_present: True" in info.to_text()


def test_cli_storage_info_json(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset("Beta", data=np.ones((1, 2, 1, 1)))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "storage-info",
            str(posterior),
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "hdf5"
    assert payload["n_chains"] == 1
    assert payload["n_draws"] == 2
    assert payload["datasets"][0]["name"] == "Beta"


def test_zarr_storage_info_when_available(tmp_path):
    zarr = pytest.importorskip("zarr")

    posterior = tmp_path / "posterior.zarr"
    root = zarr.open_group(str(posterior), mode="w")
    root.attrs["pyhmsc_metadata"] = {"names": {"species": ["sp1"]}}
    root.create_array("Beta", data=np.ones((2, 3, 1, 1)), overwrite=True)
    level = root.create_group("random_levels").create_group("0")
    level.create_array("Lambda", data=np.ones((2, 3, 1, 1)), overwrite=True)

    info = inspect_posterior_storage(posterior)
    names = [dataset.name for dataset in info.datasets]

    assert info.format == "zarr"
    assert info.metadata_present
    assert info.n_chains == 2
    assert "Beta" in names
    assert "random_levels/0/Lambda" in names


def test_chain_inspection_checks_nested_expected_draws(tmp_path):
    import h5py

    posterior = tmp_path / "posterior_chain_0.h5"
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset("Beta", data=np.ones((1, 5, 1, 1)))
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset("Eta", data=np.ones((1, 4, 2, 1)))

    status, message, _metadata = inspect_chain_file(posterior, expected_draws=5)

    assert status == "failed"
    assert "random_levels/0/Eta expected 5 draws" in message
