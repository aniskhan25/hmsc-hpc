import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_cli_compile_yaml_config(tmp_path):
    out = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "compile",
            "tests/fixtures/fixed_effect/model.yaml",
            "--output",
            str(out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    init_json = Path(result.stdout.strip())
    assert init_json.exists()
    assert init_json == out / "init.json"

    validate = subprocess.run(
        [sys.executable, "-m", "pyhmsc", "validate-init", str(init_json), "--strict"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "native_sampler_supported: passed" in validate.stdout


def test_cli_validate_init_rejects_model_yaml():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "validate-init",
            "tests/fixtures/fixed_effect/model.yaml",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "compile MODEL.yaml" in result.stderr


def test_cli_predict_probit_uses_response_scale(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    x_path = tmp_path / "X.csv"
    output = tmp_path / "pred.csv"
    pd.DataFrame({"x": [1.0]}, index=["site_1"]).to_csv(x_path)
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset("Beta", data=[[[[0.0], [1.0]], [[0.0], [-1.0]]]])
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"covariates":["Intercept","x"],"species":["sp1"]},'
            '"formula":{"X":"~ x"},"distribution":"probit"}'
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "predict",
            str(posterior),
            "--X",
            str(x_path),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    pred = pd.read_csv(output, index_col=0)
    assert 0 <= pred.loc["site_1", "sp1"] <= 1
    assert abs(pred.loc["site_1", "sp1"] - 0.5) < 1e-8


def test_cli_summarize_gamma_uses_metadata_names(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset("Gamma", data=[[[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]]])
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"covariates":["Intercept","TMG"],"traits":["Intercept","CN"]},'
            '"formula":{"X":"~ TMG"},"distribution":"probit"}'
        )

    result = subprocess.run(
        [sys.executable, "-m", "pyhmsc", "summarize", str(posterior), "--param", "Gamma", "--level", "0.5"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "covariate" in result.stdout
    assert "trait" in result.stdout
    assert "TMG" in result.stdout
    assert "CN" in result.stdout


def test_cli_diagnostics_writes_named_report(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    output = tmp_path / "diagnostics.txt"
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset(
            "Beta",
            data=[
                [[[0.0], [1.0]], [[0.1], [1.1]], [[0.2], [1.2]]],
                [[[0.0], [1.0]], [[0.1], [1.1]], [[0.2], [1.2]]],
            ],
        )
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"covariates":["Intercept","x"],"species":["sp1"]},'
            '"formula":{"X":"~ x"},"distribution":"normal"}'
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "diagnostics",
            str(posterior),
            "--param",
            "Beta",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    text = output.read_text(encoding="utf-8")
    assert "diagnostics" in text
    assert "rhat_max" in text
    assert "ess_min" in text
    assert "Intercept" in text
    assert "sp1" in text


def test_cli_sample_and_summarize(tmp_path):
    run_dir = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "compile",
            "tests/fixtures/fixed_effect/model.yaml",
            "--output",
            str(run_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    posterior = tmp_path / "posterior.h5"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "sample",
            str(run_dir / "init.json"),
            "--output",
            str(posterior),
            "--samples",
            "1",
            "--transient",
            "0",
            "--thin",
            "1",
            "--verbose",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "pyhmsc", "summarize", str(posterior), "--param", "Beta"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "covariate" in result.stdout
    assert "forest_cover" in result.stdout
    assert "sparrow" in result.stdout

    pred_out = tmp_path / "pred.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "predict",
            str(posterior),
            "--X",
            "tests/fixtures/fixed_effect/X.csv",
            "--random-effects",
            "none",
            "--output",
            str(pred_out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert pred_out.exists()

    ppc_out = tmp_path / "ppc.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "ppc",
            str(posterior),
            "--X",
            "tests/fixtures/fixed_effect/X.csv",
            "--Y",
            "tests/fixtures/fixed_effect/Y_poisson.csv",
            "--random-effects",
            "none",
            "--seed",
            "1",
            "--output",
            str(ppc_out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert ppc_out.exists()
    assert "sparrow" in ppc_out.read_text(encoding="utf-8")

    validate = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "validate",
            str(posterior),
            "--X",
            "tests/fixtures/fixed_effect/X.csv",
            "--Y",
            "tests/fixtures/fixed_effect/Y_poisson.csv",
            "--formula",
            "~ forest_cover + elevation",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "predictive_interval_contains_observed_mean" in validate.stdout
