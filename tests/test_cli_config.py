import subprocess
import sys
from pathlib import Path


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
