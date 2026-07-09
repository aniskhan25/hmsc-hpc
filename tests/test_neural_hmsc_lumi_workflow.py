import json
import subprocess
import sys
from pathlib import Path

from examples.run_neural_hmsc_benchmark import distribution_seed
from pyhmsc.posterior import HmscFit


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
        assert "--sbc-datasets" in text
        assert "--ood-regimes" in text
    assert "--coefficient-calibration" in benchmark_text
    assert "CONDITIONAL_CALIBRATION_EPOCHS" in benchmark_text
    assert "NEURAL_CHECKPOINT" in benchmark_text
    assert "PROBIT_ANCHOR" in benchmark_text
    assert "--run-mcmc-reference" not in train_text
    assert "--run-mcmc-reference" in benchmark_text


def test_distribution_seed_is_independent_of_requested_suite_order():
    assert distribution_seed(100, "normal") == 100
    assert distribution_seed(100, "probit") == 1100
    assert distribution_seed(100, "poisson") == 2100
    assert distribution_seed(100, "poisson", delta=999) == 3099


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
        "--sbc-datasets",
        "2",
        "--sbc-draws",
        "8",
        "--sbc-bins",
        "4",
        "--skip-existing",
    ]

    subprocess.run(cmd, check=True)
    subprocess.run(cmd, check=True)

    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    record = json.loads(
        (output / "normal" / "benchmark_record.json").read_text(encoding="utf-8")
    )

    assert metadata["status"] == "completed"
    assert metadata["started_at"]
    assert metadata["finished_at"]
    assert metadata["args"]["skip_existing"] is True
    assert "git_commit" in metadata
    assert manifest["suite"] == ["normal"]
    assert manifest["datasets"][0]["distribution"] == "normal"
    assert Path(record["neural_checkpoint"], "neural_checkpoint.json").exists()
    assert Path(record["neural_checkpoint"], "weights.weights.h5").exists()
    assert Path(record["neural_posterior"]).exists()
    assert Path(record["neural_posterior_uncalibrated"]).exists()
    assert Path(record["sbc_diagnostics"]).exists()
    assert (output / "neural_hmsc_sbc_diagnostics.csv").exists()
    assert (output / "neural_hmsc_sbc_diagnostics.json").exists()
    sbc_rows = json.loads(
        (output / "neural_hmsc_sbc_diagnostics.json").read_text(encoding="utf-8")
    )
    assert {row["sbc_stratum_kind"] for row in sbc_rows} >= {
        "overall",
        "coefficient",
        "design_information",
    }
    overall_rows = [row for row in sbc_rows if row["sbc_stratum_kind"] == "overall"]
    assert overall_rows
    assert all(row["sbc_stratum_label"] == "overall" for row in overall_rows)


def test_conditional_calibration_entrypoint_keeps_predictive_scalar(tmp_path):
    output = tmp_path / "conditional"
    cmd = [
        sys.executable,
        "examples/run_neural_hmsc_conditional_calibration.py",
        "--output",
        str(output),
        "--suite",
        "probit",
        "--n-sites",
        "8",
        "--n-species",
        "2",
        "--train-datasets",
        "2",
        "--calibration-datasets",
        "2",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--conditional-calibration-epochs",
        "5",
        "--neural-chains",
        "1",
        "--neural-draws",
        "3",
        "--sbc-datasets",
        "2",
        "--sbc-draws",
        "8",
        "--sbc-bins",
        "4",
        "--ood-regimes",
    ]

    subprocess.run(cmd, check=True)

    manifest = json.loads(
        (output / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    record = manifest["datasets"][0]
    coefficient_fit = HmscFit.from_file(record["neural_posterior"])
    predictive_fit = HmscFit.from_file(record["neural_predictive_distribution"])

    assert manifest["coefficient_calibration"] == "conditional"
    assert record["probit_anchor"] == "irls_laplace"
    assert record["calibration"]["method"] == "conditional_rank_aware_anchor_scale"
    assert record["predictive_calibration"]["method"] == "temperature_scale"
    assert coefficient_fit.metadata["calibration"]["semantics_version"] == 5
    assert predictive_fit.metadata["calibration"]["semantics_version"] == 2
    assert "rank_aware" in record["calibration"]
    assert "support" in record["calibration"]
    sbc_rows = json.loads(
        (output / "neural_hmsc_sbc_diagnostics.json").read_text(encoding="utf-8")
    )
    calibrated_rows = [
        row for row in sbc_rows if row["posterior_variant"] == "calibrated"
    ]
    assert calibrated_rows
    assert all("conditional_support_trust_mean" in row for row in calibrated_rows)
    assert all(
        "conditional_mean_magnitude_support_outside_fraction" in row
        for row in calibrated_rows
    )
