import subprocess
import sys


def test_python_native_smoke_runner_compile_validate_only(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "examples/run_python_native_smoke.py",
            "--project",
            "fixed_poisson",
            "--output-root",
            str(tmp_path / "examples"),
            "--skip-sample",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "== fixed_poisson ==" in result.stdout
    assert "validate-init" in result.stdout
    assert "sample: skipped" in result.stdout
