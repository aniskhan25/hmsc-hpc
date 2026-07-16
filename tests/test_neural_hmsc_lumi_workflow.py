import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from examples.run_neural_hmsc_benchmark import (
    _hard_pool_group_diagnostics,
    _hard_pool_selection_diagnostics,
    _matched_hard_pool_dataset_groups,
    _matched_score_dataset_groups,
    _ood_dataset_groups,
    _near_boundary_miss_scores,
    _select_constrained_hard_pool_indices,
    _select_low_overlap_candidate_pool_indices,
    _target_overlap_proxy,
    distribution_seed,
)
from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.simulator import FixedEffectDataset
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
    assert 'COEFFICIENT_CALIBRATION="${COEFFICIENT_CALIBRATION:-external_monotone}"' in benchmark_text
    assert "COEFFICIENT_CALIBRATION=conditional" in benchmark_text
    assert "RARE_CALIBRATION_DATASETS" in benchmark_text
    assert "RARE_VALIDATION_DATASETS" in benchmark_text
    assert "CONDITIONAL_CALIBRATION_EPOCHS" in benchmark_text
    assert "CONDITIONAL_CALIBRATION_OOD_UNCERTAINTY_STRENGTH" in benchmark_text
    assert "CONDITIONAL_CALIBRATION_OOD_UNCERTAINTY_MAX_MULTIPLIER" in benchmark_text
    assert "CONDITIONAL_CALIBRATION_OOD_OBJECTIVE" in benchmark_text
    assert "--conditional-calibration-ood-objective" in benchmark_text
    assert "--conditional-calibration-ood-datasets" in benchmark_text
    assert "EXTERNAL_MONOTONE_DATASETS" in benchmark_text
    assert "--external-monotone-datasets" in benchmark_text
    assert "--external-monotone-max-multiplier" in benchmark_text
    assert "NEURAL_CHECKPOINT" in benchmark_text
    assert "PROBIT_ANCHOR" in benchmark_text
    assert "--run-mcmc-reference" not in train_text
    assert "--run-mcmc-reference" in benchmark_text


def test_distribution_seed_is_independent_of_requested_suite_order():
    assert distribution_seed(100, "normal") == 100
    assert distribution_seed(100, "probit") == 1100
    assert distribution_seed(100, "poisson") == 2100
    assert distribution_seed(100, "poisson", delta=999) == 3099


def test_near_boundary_miss_scores_prioritize_actionable_ood_misses():
    class FakeEngine:
        def predict_beta_posterior(self, data):
            return BetaPosterior(
                mean=np.zeros_like(data.Beta, dtype=float),
                scale=np.ones_like(data.Beta, dtype=float),
            )

    def dataset(beta_value: float) -> FixedEffectDataset:
        sites = [f"site_{idx}" for idx in range(4)]
        species = ["sp1"]
        truth = pd.DataFrame(
            [[beta_value], [0.0], [0.0]],
            index=["Intercept", "x1", "x2"],
            columns=species,
        )
        return FixedEffectDataset(
            Y=pd.DataFrame(np.zeros((4, 1)), index=sites, columns=species),
            X=pd.DataFrame({"x1": np.zeros(4), "x2": np.zeros(4)}, index=sites),
            truth_beta=truth,
            linear_predictor=pd.DataFrame(
                np.zeros((4, 1)), index=sites, columns=species
            ),
            metadata={"distribution": "probit"},
        )

    inside, near_miss, far_miss = dataset(1.4), dataset(2.0), dataset(4.0)
    scores = _near_boundary_miss_scores(
        engine=FakeEngine(),
        datasets=[inside, near_miss, far_miss],
    )

    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_hard_ood_dataset_groups_create_independent_batches():
    datasets = [object() for _ in range(5)]

    grouped = _ood_dataset_groups(datasets, split=True)
    unsplit = _ood_dataset_groups(datasets, split=False)

    assert grouped == [datasets[::2], datasets[1::2]]
    assert unsplit == [datasets]


def test_matched_hard_ood_groups_balance_scores():
    datasets = [object() for _ in range(4)]
    groups = _matched_score_dataset_groups(
        datasets,
        scores=np.asarray([10.0, 9.0, 8.0, 1.0]),
    )

    assert groups == [[datasets[0], datasets[3]], [datasets[1], datasets[2]]]


def test_constrained_hard_pool_selection_requires_raw_misses_and_low_overlap():
    score_arrays = {
        "score": np.asarray([0.8, 0.7, 0.5, 0.4, 0.3, 0.2]),
        "raw_near_boundary_score": np.asarray([0.9, 0.85, 0.8, 0.25, 0.2, 0.1]),
        "overlap_proxy": np.asarray([0.9, 0.2, 0.15, 0.1, 0.05, 0.0]),
        "miss_rate": np.zeros(6),
        "near_boundary_miss_rate": np.zeros(6),
        "miss_excess_mean": np.zeros(6),
        "absolute_z_mean": np.zeros(6),
    }

    selection = _select_constrained_hard_pool_indices(score_arrays, keep_count=2)

    assert list(selection["selected_indices"]) == [1, 2]
    assert selection["diagnostics"]["eligible_count"] >= 2
    assert selection["diagnostics"]["used_fallback_pool"] is False


def test_low_overlap_candidate_pool_prefilter_keeps_raw_miss_candidates():
    score_arrays = {
        "score": np.asarray([0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),
        "raw_near_boundary_score": np.asarray(
            [0.9, 0.85, 0.8, 0.75, 0.7, 0.2, 0.1, 0.0]
        ),
        "overlap_proxy": np.asarray([0.95, 0.15, 0.20, 0.25, 0.30, 0.02, 0.01, 0.0]),
        "miss_rate": np.zeros(8),
        "near_boundary_miss_rate": np.zeros(8),
        "miss_excess_mean": np.zeros(8),
        "absolute_z_mean": np.zeros(8),
    }

    selection = _select_low_overlap_candidate_pool_indices(
        score_arrays,
        keep_count=3,
    )

    assert list(selection["selected_indices"]) == [1, 2, 3]
    assert selection["diagnostics"]["generated_count"] == 8
    assert selection["diagnostics"]["candidate_pool_count"] == 3
    assert selection["diagnostics"]["used_fallback_pool"] is False
    assert (
        selection["diagnostics"]["candidate_pool_summaries"]["overlap_proxy"]["max"]
        < 0.30
    )


def test_matched_hard_pool_groups_balance_raw_difficulty_and_overlap():
    datasets = [object() for _ in range(4)]
    score_arrays = {
        "score": np.asarray([0.8, 0.7, 0.2, 0.1]),
        "raw_near_boundary_score": np.asarray([1.0, 0.1, 0.9, 0.2]),
        "overlap_proxy": np.asarray([0.9, 0.1, 0.8, 0.2]),
        "miss_rate": np.zeros(4),
        "near_boundary_miss_rate": np.zeros(4),
        "miss_excess_mean": np.zeros(4),
        "absolute_z_mean": np.zeros(4),
    }

    groups = _matched_hard_pool_dataset_groups(datasets, score_arrays=score_arrays)
    raw_totals = [
        sum(
            score_arrays["raw_near_boundary_score"][datasets.index(item)]
            for item in group
        )
        for group in groups
    ]
    overlap_totals = [
        sum(score_arrays["overlap_proxy"][datasets.index(item)] for item in group)
        for group in groups
    ]

    assert [len(group) for group in groups] == [2, 2]
    assert abs(raw_totals[0] - raw_totals[1]) <= 0.2
    assert abs(overlap_totals[0] - overlap_totals[1]) <= 0.2


def test_hard_pool_diagnostics_include_selection_and_group_summaries():
    datasets = [object() for _ in range(4)]
    score_arrays = {
        "score": np.asarray([10.0, 9.0, 8.0, 7.0]),
        "raw_near_boundary_score": np.asarray([11.0, 9.5, 8.5, 7.5]),
        "overlap_proxy": np.asarray([0.1, 0.2, 0.3, 0.4]),
        "miss_rate": np.asarray([0.8, 0.7, 0.6, 0.1]),
        "near_boundary_miss_rate": np.asarray([0.4, 0.3, 0.2, 0.1]),
        "miss_excess_mean": np.asarray([0.5, 0.4, 0.3, 0.0]),
        "absolute_z_mean": np.asarray([2.4, 2.3, 2.2, 1.2]),
    }

    diagnostics = _hard_pool_selection_diagnostics(
        regime="combined_shift",
        requested_count=2,
        keep_count=4,
        candidate_count=4,
        hard_target_multiplier=2,
        hard_target_candidate_multiplier=1,
        score_arrays=score_arrays,
        selected_indices=np.asarray([0, 3]),
        selection_applied=True,
    )
    groups = _matched_score_dataset_groups(datasets, scores=score_arrays["score"])
    group_diagnostics = _hard_pool_group_diagnostics(
        datasets,
        groups,
        score_arrays=score_arrays,
    )

    assert diagnostics["regime"] == "combined_shift"
    assert diagnostics["selected_indices"] == [0, 3]
    assert diagnostics["candidate_summaries"]["score"]["count"] == 4
    assert diagnostics["selected_summaries"]["score"]["count"] == 2
    assert diagnostics["selected_summaries"]["overlap_proxy"]["mean"] == 0.25
    assert group_diagnostics["split"] is True
    assert group_diagnostics["score_total_difference"] == 0.0
    assert group_diagnostics["groups"][0]["dataset_indices"] == [0, 3]
    assert group_diagnostics["groups"][1]["summaries"]["miss_rate"]["count"] == 2


def test_gate_aware_hard_scores_penalize_in_domain_overlap_context():
    class FakeEngine:
        def predict_beta_posterior(self, data):
            return BetaPosterior(
                mean=np.zeros_like(data.Beta, dtype=float),
                scale=np.ones_like(data.Beta, dtype=float),
            )

    def dataset(x_value: float) -> FixedEffectDataset:
        sites = [f"site_{idx}" for idx in range(4)]
        species = ["sp1"]
        truth = pd.DataFrame(
            [[2.0], [0.0], [0.0]],
            index=["Intercept", "x1", "x2"],
            columns=species,
        )
        return FixedEffectDataset(
            Y=pd.DataFrame(np.zeros((4, 1)), index=sites, columns=species),
            X=pd.DataFrame(
                {"x1": np.full(4, x_value), "x2": np.full(4, x_value)}, index=sites
            ),
            truth_beta=truth,
            linear_predictor=pd.DataFrame(
                np.zeros((4, 1)), index=sites, columns=species
            ),
            metadata={"distribution": "probit"},
        )

    support_close, support_far = dataset(0.0), dataset(3.0)
    raw_scores = _near_boundary_miss_scores(
        engine=FakeEngine(),
        datasets=[support_close, support_far],
    )
    gate_scores = _near_boundary_miss_scores(
        engine=FakeEngine(),
        datasets=[support_close, support_far],
        regime="effect_size_shift",
    )
    overlap = _target_overlap_proxy(
        posterior_mean=np.zeros((2, 3, 1), dtype=float),
        X=np.stack(
            [
                np.column_stack([np.ones(4), np.zeros((4, 2))]),
                np.column_stack([np.ones(4), np.full((4, 2), 3.0)]),
            ]
        ),
        Y=np.zeros((2, 4, 1), dtype=float),
        regime="effect_size_shift",
    )

    assert raw_scores[0] == raw_scores[1]
    assert overlap[0] > overlap[1]
    assert gate_scores[0] < gate_scores[1]


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


def test_fixed_evaluation_harness_compares_identical_sbc_keys(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    output = tmp_path / "comparison"
    base.mkdir()
    candidate.mkdir()

    def rows(ood_delta: float, combined_delta: float) -> list[dict[str, object]]:
        template = []
        for domain, regime, coverage in [
            ("in_distribution", None, 0.95),
            ("ood", "covariate_shift", 0.94 + ood_delta),
            ("ood", "effect_size_shift", 0.90 + ood_delta),
            ("ood", "combined_shift", 0.88 + combined_delta),
        ]:
            template.append(
                {
                    "distribution": "probit",
                    "simulation_domain": domain,
                    "ood_regime": regime,
                    "posterior_variant": "calibrated",
                    "sbc_stratum_kind": "overall",
                    "sbc_stratum_label": "overall",
                    "sbc_beta_interval_coverage_95": coverage,
                    "sbc_rank_mean": 0.5,
                    "sbc_rank_variance": 0.08,
                    "sbc_beta_mean_rmse": 0.1,
                }
            )
        for label, coverage in [
            ("rare", 0.93),
            ("intermediate", 0.95),
            ("common", 0.96),
        ]:
            template.append(
                {
                    "distribution": "probit",
                    "simulation_domain": "in_distribution",
                    "ood_regime": None,
                    "posterior_variant": "calibrated",
                    "sbc_stratum_kind": "prevalence",
                    "sbc_stratum_label": label,
                    "sbc_beta_interval_coverage_95": coverage,
                    "sbc_rank_mean": 0.5,
                    "sbc_rank_variance": 0.08,
                    "sbc_beta_mean_rmse": 0.1,
                }
            )
        for label, coverage in [
            ("low", 0.95),
            ("intermediate", 0.94),
            ("high", 0.93),
        ]:
            template.append(
                {
                    "distribution": "probit",
                    "simulation_domain": "in_distribution",
                    "ood_regime": None,
                    "posterior_variant": "calibrated",
                    "sbc_stratum_kind": "design_information",
                    "sbc_stratum_label": label,
                    "sbc_beta_interval_coverage_95": coverage,
                    "sbc_rank_mean": 0.5,
                    "sbc_rank_variance": 0.08,
                    "sbc_beta_mean_rmse": 0.1,
                }
            )
        return template

    (base / "neural_hmsc_sbc_diagnostics.json").write_text(
        json.dumps(rows(0.0, 0.0)),
        encoding="utf-8",
    )
    (candidate / "neural_hmsc_sbc_diagnostics.json").write_text(
        json.dumps(rows(0.02, 0.02)),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "examples/compare_neural_hmsc_fixed_evaluation.py",
            "--run",
            f"base={base}",
            "--run",
            f"candidate={candidate}",
            "--baseline",
            "base",
            "--output",
            str(output),
        ],
        check=True,
    )

    report = json.loads(
        (output / "fixed_evaluation_comparison.json").read_text(encoding="utf-8")
    )
    candidate_row = next(row for row in report["summary"] if row["run"] == "candidate")

    assert report["fixed_row_key_count"] == len(rows(0.0, 0.0))
    assert candidate_row["mean_ood_gate_passed"] is True
    assert candidate_row["combined_shift_gate_passed"] is True
    assert candidate_row["fixed_evaluation_acceptance_passed"] is True
    assert (output / "fixed_evaluation_summary.csv").exists()
    assert (output / "fixed_evaluation_deltas.csv").exists()
    assert (output / "fixed_evaluation_comparison.md").exists()


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
    assert coefficient_fit.metadata["calibration"]["semantics_version"] == 6
    assert predictive_fit.metadata["calibration"]["semantics_version"] == 2
    assert "rank_aware" in record["calibration"]
    assert "support" in record["calibration"]
    assert "ood_uncertainty" in record["calibration"]["support"]
    sbc_rows = json.loads(
        (output / "neural_hmsc_sbc_diagnostics.json").read_text(encoding="utf-8")
    )
    calibrated_rows = [
        row for row in sbc_rows if row["posterior_variant"] == "calibrated"
    ]
    assert calibrated_rows
    assert all("conditional_support_trust_mean" in row for row in calibrated_rows)
    assert all(
        "conditional_ood_uncertainty_inflation_mean" in row for row in calibrated_rows
    )
    assert all(
        "conditional_mean_magnitude_support_outside_fraction" in row
        for row in calibrated_rows
    )
    assert all("conditional_effect_size_signal_mean" in row for row in calibrated_rows)
