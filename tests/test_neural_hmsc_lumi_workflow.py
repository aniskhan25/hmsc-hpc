import json
import subprocess
import sys
from pathlib import Path


def test_lumi_neural_hmsc_sbatch_scripts_are_complete_and_valid():
    train = Path("docs/lumi_neural_hmsc_train_sbatch.sh")
    benchmark = Path("docs/lumi_neural_hmsc_benchmark_sbatch.sh")

    subprocess.run(["bash", "-n", str(train)], check=True)
    subprocess.run(["bash", "-n", str(benchmark)], check=True)

    train_text = train.read_text(encoding="utf-8")
    benchmark_text = benchmark.read_text(encoding="utf-8")
    for text in (train_text, benchmark_text):
        assert "RUN_ROOT" in text
        assert "/scratch/${PROJECT_ID}/anisrahm" in text
        assert "run_neural_hmsc_benchmark.py" in text
        assert "--skip-existing" in text
        assert "rocm-smi" in text
        assert "wall_time.txt" in text
    assert "--run-mcmc-reference" not in train_text
    assert "--run-mcmc-reference" in benchmark_text


def test_neural_benchmark_runner_writes_metadata_and_reuses_outputs(tmp_path):
    output = tmp_path / "benchmark"
    cmd = [
        sys.executable,
        "examples/run_neural_hmsc_benchmark.py",
        "--output",
        str(output),
        "--suite",
        "normal",
        "--n-sites",
        "8",
        "--n-species",
        "2",
        "--train-datasets",
        "2",
        "--calibration-datasets",
        "1",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--neural-chains",
        "1",
        "--neural-draws",
        "3",
        "--skip-existing",
    ]

    subprocess.run(cmd, check=True)
    subprocess.run(cmd, check=True)

    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "benchmark_manifest.json").read_text(encoding="utf-8"))
    record = json.loads((output / "normal" / "benchmark_record.json").read_text(encoding="utf-8"))

    assert metadata["status"] == "completed"
    assert metadata["started_at"]
    assert metadata["finished_at"]
    assert metadata["args"]["skip_existing"] is True
    assert "git_commit" in metadata
    assert manifest["suite"] == ["normal"]
    assert manifest["datasets"][0]["distribution"] == "normal"
    assert Path(record["neural_posterior"]).exists()
    assert Path(record["neural_posterior_uncalibrated"]).exists()
