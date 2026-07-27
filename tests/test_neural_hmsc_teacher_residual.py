from pathlib import Path

import numpy as np

from examples.evaluate_neural_hmsc_mcmc_teacher_residual import (
    _one_covariate_dataset,
    _regime_holdout_profiles,
    _split_simulated_sites,
)
from pyhmsc.neural.simulator import simulate_fixed_effect_dataset

from pyhmsc.neural.teacher_residual import (
    ContextIdentityGate,
    McmcTeacherResponseBatch,
    McmcTeacherResidualHead,
    fit_context_identity_gate,
    fit_cross_fitted_mcmc_teacher_residual_head,
    fit_mcmc_teacher_residual_head,
    response_context_features,
    response_context_summary,
)


def test_teacher_residual_selects_bounded_head_and_round_trips(tmp_path):
    training = [_batch(seed=1, offset=0.30, label="in_distribution")]
    validation = [
        _batch(seed=2, offset=0.25, label="in_distribution"),
        _batch(seed=3, offset=0.25, label="big_spatial_shape"),
    ]
    head = fit_mcmc_teacher_residual_head(
        training,
        validation,
        baseline_id="neural_predictive_affine_v1",
        epochs=150,
        learning_rate=0.02,
        identity_penalty=0.01,
        max_abs_logit_residual=0.4,
        seed=8,
    )

    assert head.selected
    prediction = head.predict_mean(validation[0].baseline_probability, validation[0].X)
    residual = head.predict_residual(
        validation[0].baseline_probability, validation[0].X
    )
    assert np.max(np.abs(residual)) <= 0.4 + 1.0e-7
    assert np.mean(prediction) > np.mean(validation[0].baseline_probability)
    assert not head.to_metadata()["training_target"].startswith("observed")

    path = head.save(tmp_path / "head")
    loaded = McmcTeacherResidualHead.load(path)
    np.testing.assert_allclose(
        loaded.predict_mean(validation[0].baseline_probability, validation[0].X),
        prediction,
        atol=1.0e-6,
    )


def test_teacher_residual_falls_back_to_identity_on_validation_degradation():
    training = [_batch(seed=4, offset=0.30, label="in_distribution")]
    validation = [
        _batch(
            seed=5,
            offset=0.30,
            label="in_distribution",
            outcome=np.zeros((5, 3)),
        ),
        _batch(
            seed=6,
            offset=0.30,
            label="big_spatial_shape",
            outcome=np.zeros((5, 3)),
        ),
    ]
    head = fit_mcmc_teacher_residual_head(
        training,
        validation,
        baseline_id="neural_predictive_affine_v1",
        epochs=100,
        learning_rate=0.02,
        seed=9,
    )

    assert not head.selected
    np.testing.assert_allclose(
        head.predict_mean(validation[0].baseline_probability, validation[0].X),
        validation[0].baseline_probability,
        atol=1.0e-7,
    )


def test_response_context_features_do_not_require_outcomes():
    baseline = np.full((4, 2), 0.4)
    X = np.column_stack([np.ones(4), np.linspace(-1.0, 1.0, 4)])

    features = response_context_features(baseline, X)

    assert features.shape == (8, 10)
    assert np.isfinite(features).all()


def test_sample_size_stable_features_isolate_bounded_site_context():
    baseline = np.asarray(
        [[0.1, 0.3], [0.2, 0.4], [0.3, 0.5], [0.4, 0.6]],
        dtype=float,
    )
    X = np.column_stack([np.ones(4), np.linspace(-1.0, 1.0, 4)])
    repeated_baseline = np.tile(baseline, (18, 1))
    repeated_X = np.tile(X, (18, 1))

    compact = response_context_features(baseline, X).reshape(4, 2, 10)
    large = response_context_features(repeated_baseline, repeated_X).reshape(
        72, 2, 10
    )
    np.testing.assert_allclose(compact[:, :, :8], large[:4, :, :8], atol=1e-6)
    assert np.all(compact[:, :, 8] < large[:4, :, 8])

    compact_summary = response_context_summary(baseline, X)
    large_summary = response_context_summary(repeated_baseline, repeated_X)
    assert compact_summary.shape == (15,)
    stable_indices = [index for index in range(15) if index not in {2, 3, 10}]
    np.testing.assert_allclose(
        compact_summary[stable_indices],
        large_summary[stable_indices],
        atol=1e-6,
    )
    assert compact_summary[10] < large_summary[10]


def test_context_identity_gate_uses_only_probability_and_design():
    batches = []
    for community in (101, 102, 103):
        batches.extend(_context_batches(community))
    gate = fit_context_identity_gate(
        batches,
        approved_labels=("effect_size_shift", "big_spatial_shape"),
        margin=0.0,
    )

    decisions = {
        batch.label: gate.decision(batch.baseline_probability, batch.X)
        for batch in _context_batches(104)
    }

    assert decisions["effect_size_shift"]["active"]
    assert decisions["big_spatial_shape"]["active"]
    assert not decisions["covariate_shift"]["active"]
    assert not decisions["rare_validation"]["active"]
    assert response_context_summary(
        batches[0].baseline_probability, batches[0].X
    ).shape == (15,)

    restored = ContextIdentityGate.from_metadata(gate.to_metadata())
    assert (
        restored.decision(batches[0].baseline_probability, batches[0].X)["active"]
        == gate.decision(batches[0].baseline_probability, batches[0].X)["active"]
    )


def test_cross_fitted_head_selects_stable_context_movement(tmp_path):
    batches = []
    for community in (201, 202, 203):
        batches.extend(_context_batches(community))

    head = fit_cross_fitted_mcmc_teacher_residual_head(
        batches,
        baseline_id="neural_predictive_affine_v1",
        epochs=80,
        learning_rate=0.02,
        identity_penalty=0.01,
        shrinkage_grid=(0.25, 0.5),
        margin_grid=(0.0, 0.25),
        seed=31,
    )

    assert head.selected
    assert head.context_gate is not None
    assert head.metadata["cross_fit"]["n_folds"] == 3
    assert any(
        candidate["accepted"] for candidate in head.metadata["cross_fit"]["candidates"]
    )
    example = {batch.label: batch for batch in _context_batches(204)}
    np.testing.assert_allclose(
        head.predict_mean(
            example["rare_validation"].baseline_probability,
            example["rare_validation"].X,
        ),
        example["rare_validation"].baseline_probability,
        atol=1.0e-7,
    )
    assert np.mean(
        head.predict_mean(
            example["effect_size_shift"].baseline_probability,
            example["effect_size_shift"].X,
        )
    ) > np.mean(example["effect_size_shift"].baseline_probability)

    loaded = McmcTeacherResidualHead.load(head.save(tmp_path / "crossfit_head"))
    assert loaded.context_gate is not None
    np.testing.assert_allclose(
        loaded.predict_mean(
            example["big_spatial_shape"].baseline_probability,
            example["big_spatial_shape"].X,
        ),
        head.predict_mean(
            example["big_spatial_shape"].baseline_probability,
            example["big_spatial_shape"].X,
        ),
        atol=1.0e-6,
    )


def test_compact_evaluator_gradient_does_not_reference_outcome_tensor():
    text = Path("pyhmsc/neural/teacher_residual.py").read_text(encoding="utf-8")
    gradient_block = text[
        text.index("with tf.GradientTape() as tape:") : text.index(
            "gradients = tape.gradient"
        )
    ]

    assert "teacher_tensor" in gradient_block
    assert "outcome" not in gradient_block


def test_teacher_corpus_uses_disjoint_training_and_holdout_sites():
    dataset = simulate_fixed_effect_dataset(
        n_sites=9,
        n_species=2,
        distribution="probit",
        seed=17,
    )

    training, holdout = _split_simulated_sites(dataset, n_train=6)

    assert len(training.X) == 6
    assert len(holdout.X) == 3
    assert set(training.X.index).isdisjoint(holdout.X.index)
    np.testing.assert_allclose(training.truth_beta, holdout.truth_beta)
    assert training.metadata["site_partition"] == "training"
    assert holdout.metadata["site_partition"] == "holdout"


def test_teacher_simulation_matches_frozen_one_covariate_shape():
    dataset = simulate_fixed_effect_dataset(
        n_sites=8,
        n_species=4,
        distribution="probit",
        seed=23,
    )

    projected = _one_covariate_dataset(dataset, seed=23)

    assert list(projected.X.columns) == ["x1"]
    assert list(projected.truth_beta.index) == ["Intercept", "x1"]
    assert projected.Y.shape == (8, 4)
    assert projected.linear_predictor.shape == (8, 4)
    assert projected.metadata["n_covariates"] == 2
    assert projected.metadata["formula"] == "~ x1"


def test_teacher_holdout_profiles_span_real_context_sizes():
    args = type(
        "Args",
        (),
        {"small_holdout_sites": 12, "holdout_sites": 20, "large_holdout_sites": 360},
    )()

    assert _regime_holdout_profiles("in_distribution", args=args) == {
        "compact": 20
    }
    assert _regime_holdout_profiles("rare_validation", args=args) == {
        "whittaker": 12,
        "compact": 20,
    }
    assert _regime_holdout_profiles("big_spatial_shape", args=args) == {
        "compact": 20,
        "big_spatial": 360,
    }


def _batch(seed, *, offset, label, outcome=None):
    rng = np.random.default_rng(seed)
    baseline = rng.uniform(0.20, 0.55, size=(5, 3))
    baseline_logit = np.log(baseline) - np.log1p(-baseline)
    teacher = 1.0 / (1.0 + np.exp(-(baseline_logit + offset)))
    X = np.column_stack(
        [
            np.ones(5),
            rng.normal(size=5),
            rng.normal(size=5),
        ]
    )
    if outcome is None:
        outcome = np.ones((5, 3))
    return McmcTeacherResponseBatch(
        baseline_probability=baseline,
        teacher_probability=teacher,
        X=X,
        Y=np.asarray(outcome, dtype=float),
        label=label,
        seed=seed,
    )


def _context_batches(community):
    settings = {
        "in_distribution": (0.80, -3.0, 0.0),
        "covariate_shift": (0.50, 4.0, 0.0),
        "effect_size_shift": (0.30, 0.0, 0.45),
        "big_spatial_shape": (0.65, 2.0, 0.45),
        "rare_validation": (0.05, 0.0, 0.0),
    }
    batches = []
    for index, (label, (probability, covariate, offset)) in enumerate(settings.items()):
        baseline = np.full((6, 4), probability, dtype=float)
        logit = np.log(baseline) - np.log1p(-baseline)
        teacher = 1.0 / (1.0 + np.exp(-(logit + offset)))
        outcome = np.ones_like(baseline) if offset > 0.0 else np.zeros_like(baseline)
        X = np.column_stack([np.ones(6), np.full(6, covariate)])
        batches.append(
            McmcTeacherResponseBatch(
                baseline_probability=baseline,
                teacher_probability=teacher,
                X=X,
                Y=outcome,
                label=label,
                seed=int(community) * 100 + index,
            )
        )
    return batches
