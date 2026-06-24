import json
import subprocess
import sys


def test_storage_release_qualification_smoke(tmp_path):
    output = tmp_path / "qualification"
    result = subprocess.run(
        [
            sys.executable,
            "examples/run_storage_release_qualification.py",
            "--output",
            str(output),
            "--chains",
            "2",
            "--draws",
            "8",
            "--covariates",
            "3",
            "--species",
            "4",
            "--sites",
            "5",
            "--factors",
            "2",
            "--skip-zarr",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "hdf5_storage_info: passed" in result.stdout
    assert "nested_chain_status: passed" in result.stdout
    assert "hdf5_merge: passed" in result.stdout
    assert "nested_truncation_detection: passed" in result.stdout
    assert (output / "storage_release_qualification.txt").exists()

    payload = json.loads((output / "storage_release_qualification.json").read_text())
    by_name = {entry["name"]: entry for entry in payload}
    assert by_name["hdf5_storage_info"]["passed"]
    assert by_name["nested_truncation_detection"]["passed"]
    assert by_name["zarr_storage_info"]["details"]["skipped"]
