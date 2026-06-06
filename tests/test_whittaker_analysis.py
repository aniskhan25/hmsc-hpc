import subprocess
import sys

import pandas as pd


def test_whittaker_analysis_script_smoke(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    project = tmp_path / "project"
    data = project / "data"
    data.mkdir(parents=True)
    pd.DataFrame({"TMG": [-1.0, 1.0]}, index=["site_1", "site_2"]).to_csv(data / "X.csv")
    pd.DataFrame({"CN": [0.2, 1.0]}, index=["sp1", "sp2"]).to_csv(data / "traits.csv")
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset(
            "Beta",
            data=[
                [
                    [[0.0, 0.0], [-1.0, 0.5]],
                    [[0.0, 0.0], [-0.8, 0.4]],
                ]
            ],
        )
        handle.create_dataset(
            "Gamma",
            data=[
                [
                    [[-1.0, 0.5], [-0.3, 0.1]],
                    [[-1.2, 0.6], [-0.2, 0.2]],
                ]
            ],
        )
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"covariates":["Intercept","TMG"],"species":["sp1","sp2"],'
            '"traits":["Intercept","CN"]},"formula":{"X":"~ TMG"},"distribution":"probit"}'
        )

    result = subprocess.run(
        [
            sys.executable,
            "examples/analyze_whittaker_plants.py",
            "--posterior",
            str(posterior),
            "--project",
            str(project),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Whittaker Plant Validation Report" in result.stdout
    assert "negative mean effects: 1 / 2" in result.stdout
    assert "community-weighted CN" in result.stdout
    assert "Diagnostics" in result.stdout
    assert "max R-hat" in result.stdout
