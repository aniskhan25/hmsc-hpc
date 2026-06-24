import subprocess
import sys

import h5py
import numpy as np

from examples.plan_long_validation import build_plan


def test_long_validation_plan_flags_only_failed_diagnostics(tmp_path):
    posterior = tmp_path / "posterior.h5"
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset(
            "Beta",
            data=np.array(
                [
                    [[[0.0], [0.0]], [[0.0], [0.1]], [[0.0], [0.0]], [[0.0], [0.1]]],
                    [[[0.0], [2.0]], [[0.0], [2.1]], [[0.0], [2.0]], [[0.0], [2.1]]],
                ]
            ),
        )
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset(
            "Lambda",
            data=np.array(
                [
                    [[[1.0, 0.2]], [[1.0, 0.2]], [[1.0, 0.2]], [[1.0, 0.2]]],
                    [[[1.0, 0.2]], [[1.0, 0.2]], [[1.0, 0.2]], [[1.0, 0.2]]],
                ]
            ),
        )
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"covariates":["Intercept","x"],"species":["sp1","sp2"]}}'
        )

    plan = build_plan(
        [posterior],
        params=["Beta", "Associations"],
        rhat_threshold=1.01,
        ess_threshold=1.0,
    )

    beta = plan.loc[plan["param"] == "Beta"].iloc[0]
    associations = plan.loc[plan["param"] == "Associations"].iloc[0]
    assert beta["needs_longer_validation"]
    assert "longer 4-chain" in beta["recommendation"]
    assert not associations["needs_longer_validation"]
    assert associations["recommendation"] == "no longer run needed for this diagnostic"


def test_long_validation_plan_cli_writes_text_and_csv(tmp_path):
    posterior = tmp_path / "posterior.h5"
    output = tmp_path / "plan.txt"
    csv_output = tmp_path / "plan.csv"
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset("Beta", data=np.ones((2, 3, 1, 1)))

    subprocess.run(
        [
            sys.executable,
            "examples/plan_long_validation.py",
            str(posterior),
            "--param",
            "Beta",
            "--output",
            str(output),
            "--csv-output",
            str(csv_output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Targeted Longer Validation Plan" in output.read_text(encoding="utf-8")
    assert "needs_longer_validation" in csv_output.read_text(encoding="utf-8")


def test_targeted_long_validation_lumi_script_syntax():
    subprocess.run(["bash", "-n", "docs/lumi_targeted_long_validation_sbatch.sh"], check=True)
