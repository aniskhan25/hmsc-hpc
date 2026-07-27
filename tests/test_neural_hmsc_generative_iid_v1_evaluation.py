import copy

import numpy as np
import pytest
import tensorflow as tf

from examples import (
    run_generative_neural_hmsc_iid_v1_production as production_harness,
)
from pyhmsc.neural.generative_iid import (
    GenerativeIidPosteriorModel,
    JointStateLayout,
    batch_generative_iid_datasets,
    importance_weighted_no_latent_ablation_loss,
    make_stratified_response_mask,
    simulate_generative_iid_dataset,
    state_vector_from_truth,
    train_generative_iid_no_latent_ablation,
    zero_latent_state_draws,
)
from pyhmsc.neural.generative_iid_evaluation import (
    evaluate_state_draws,
    fixed_mcmc_subset_seeds,
    fixed_validation_gates,
    qualification_report,
)
from pyhmsc.neural.generative_iid_mcmc import run_exact_model_mcmc


def _dataset(seed=881001):
    mask = make_stratified_response_mask(24, 12, seed=seed)
    return simulate_generative_iid_dataset(
        n_sites=24,
        n_species=12,
        covariate_shape="normal",
        loading_stratum="medium",
        prevalence_stratum="moderate",
        seed=seed,
        response_mask=mask,
    )


def test_state_draw_evaluator_emits_all_registered_metric_families():
    dataset = _dataset()
    layout = JointStateLayout(24, 12)
    batch = batch_generative_iid_datasets(
        [dataset], max_sites=24, max_species=12
    )
    truth = np.asarray(state_vector_from_truth(batch, layout))[0]
    rng = np.random.default_rng(881002)
    draws = truth[None, :] + rng.normal(scale=0.10, size=(256, layout.size))
    row = evaluate_state_draws(
        dataset,
        draws,
        layout=layout,
        method="ordinary_fixture",
    )

    assert row["draw_count"] == 256
    assert set(row["marginal"]) == {"Beta", "R", "alpha", "log_tau"}
    assert set(row["projections"]) == {"Beta", "R", "C"}
    assert len(row["invariant_vector_draws"]["truth"]) == 48
    assert len(row["invariant_vector_draws"]["draws"]) == 256
    assert 0.0 <= row["masked_cell_brier"] <= 1.0
    assert np.isfinite(row["new_site_log_loss"])


def test_no_latent_objective_zeros_response_random_effect_and_trains():
    dataset = _dataset(881011)
    batch = batch_generative_iid_datasets(
        [dataset], max_sites=24, max_species=12
    )
    model = GenerativeIidPosteriorModel(max_sites=24, max_species=12)
    posterior = model(batch.model_inputs(), training=False)
    samples = posterior.sample(4, seed=881012)
    zeroed = np.asarray(zero_latent_state_draws(samples, posterior.layout))
    assert np.allclose(zeroed[..., posterior.layout.eta_slice], 0.0)
    assert np.allclose(zeroed[..., posterior.layout.lambda_slice], 0.0)
    loss, diagnostics = importance_weighted_no_latent_ablation_loss(
        posterior,
        batch.model_inputs(),
        draws=4,
        seed=881013,
    )
    assert np.isfinite(float(loss))
    assert np.isfinite(float(diagnostics["iwelbo"]))
    before = [np.asarray(value).copy() for value in model.weights]
    history = train_generative_iid_no_latent_ablation(
        model,
        batch,
        epochs=1,
        batch_size=1,
        model_seed=881014,
        importance_draws=2,
    )
    assert np.isfinite(history.loss).all()
    assert any(
        not np.array_equal(left, np.asarray(right))
        for left, right in zip(before, model.weights)
    )


def test_exact_mcmc_continuation_rejects_wrong_chain_state_shape():
    dataset = _dataset(881021)
    with pytest.raises(ValueError, match="continued exact-MCMC"):
        run_exact_model_mcmc(
            dataset,
            chains=2,
            warmup=2,
            draws=2,
            seed=881022,
            initial_state=np.zeros((2, 3)),
        )


def test_full_factorial_gate_engine_has_no_implicit_or_missing_gate():
    cells = production_harness._factorial_cells()
    base_seed = 881_100_001
    candidate = []
    ablation = []
    invariant_draws = np.tile(
        np.linspace(-0.2, 0.2, 48),
        (32, 1),
    )
    invariant_draws += np.arange(32)[:, None] * 0.001
    for index, cell in enumerate(cells):
        seed = base_seed + index
        loading = cell["loading_stratum"]
        score = 0.10 if loading == "weak" else 0.09
        row = {
            "seed": seed,
            **cell,
            "method": "candidate_fixture",
            "all_finite": True,
            "marginal": {
                family: {
                    "coverage_95": 0.95,
                    "interval_width_95_median": 1.0,
                    "rank_mean": 0.50,
                    "rank_variance": 0.083,
                    "element_count": 16,
                }
                for family in ("Beta", "R", "alpha", "log_tau")
            },
            "projections": {
                family: {
                    "coverage_95": 0.95,
                    "rank_mean": 0.50,
                    "rank_variance": 0.083,
                }
                for family in ("Beta", "R", "C")
            },
            "association_truth_correlation": 0.80,
            "association_rmse": 0.50,
            "association_vector_mean": [0.1, 0.2, 0.4],
            "random_effect_rmse": 0.50,
            "mean_absolute_off_diagonal_c": 0.10,
            "masked_cell_brier": score,
            "masked_cell_log_loss": score,
            "new_site_brier": 0.10,
            "new_site_log_loss": 0.10,
            "site_richness_90_coverage": 0.90,
            "species_prevalence_90_coverage": 0.90,
            "invariant_vector_draws": {
                "draws": invariant_draws.tolist(),
                "truth": np.zeros(48).tolist(),
            },
        }
        candidate.append(row)
        control = copy.deepcopy(row)
        control["method"] = "ablation_fixture"
        control["random_effect_rmse"] = 1.0
        control["masked_cell_brier"] = 0.10
        control["masked_cell_log_loss"] = 0.10
        ablation.append(control)

    subset = set(fixed_mcmc_subset_seeds(candidate))
    exact = [
        {**copy.deepcopy(row), "method": "exact_fixture"}
        for row in candidate
        if row["seed"] in subset
    ]
    python = [
        {
            "seed": row["seed"],
            "method": "python_fixture",
            "masked_cell_brier": row["masked_cell_brier"],
            "masked_cell_log_loss": row["masked_cell_log_loss"],
            "association_vector_mean": row["association_vector_mean"],
            "all_finite": True,
        }
        for row in candidate
        if row["seed"] in subset
    ]
    v0 = [
        {
            "seed": row["seed"],
            "method": "v0_fixture",
            "marginal": {
                "Beta": {"coverage_95": 0.95, "element_count": 16}
            },
            "masked_cell_brier": row["masked_cell_brier"],
            "masked_cell_log_loss": row["masked_cell_log_loss"],
            "all_finite": True,
        }
        for row in candidate
        if row["n_sites"] == 40 and row["n_species"] == 75
    ]
    gates = fixed_validation_gates(
        candidate,
        ablation_rows=ablation,
        exact_rows=exact,
        python_rows=python,
        v0_rows=v0,
        operational={
            "checkpoint_roundtrip": True,
            "permutation_invariance": True,
            "padding_invariance": True,
            "dependency_inventory_clean": True,
            "covariance_jitter_fraction": 0.0,
            "covariance_condition_max": 10.0,
        },
        mcmc_diagnostics=[
            {"split_rhat_max": 1.01, "bulk_ess_min": 500.0}
            for _ in exact
        ],
        runtime={
            "training_dev_gpu_hours": 1.0,
            "max_shape_inference_seconds": 1.0,
            "peak_device_memory_bytes": 1024.0,
            "speedup_vs_exact_mcmc": 25.0,
        },
    )
    assert len(gates) >= 45
    assert all(gates.values()), [name for name, value in gates.items() if not value]

    report = qualification_report(
        gates=gates,
        freeze_binding={"content_sha256": "a" * 64},
        seed_roles={"fixture_only": True},
        artifacts={"fixture": {"sha256": "b" * 64}},
    )
    assert report["all_gates_passed"] is True
    assert report["failed_gates"] == []
    assert report["decision"] == "eligible_to_authorize_503m_505m"
