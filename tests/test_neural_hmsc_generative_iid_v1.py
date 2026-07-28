import itertools
import json
import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm
import tensorflow as tf

from examples import run_generative_neural_hmsc_iid_v1 as sealed_harness
from examples import (
    run_generative_neural_hmsc_iid_v1_production as production_harness,
)
from pyhmsc.neural.generative_iid import (
    GenerativeIidPosteriorModel,
    JointLowRankPosterior,
    JointStateLayout,
    association_to_correlation,
    batch_generative_iid_datasets,
    gauge_fix_factors,
    generative_log_prior,
    importance_weighted_variational_loss,
    make_stratified_response_mask,
    posterior_mean_invariants,
    probit_log_likelihood,
    simulate_generative_iid_dataset,
    state_vector_from_truth,
    train_generative_iid_model,
)
from pyhmsc.neural.generative_iid_artifact import (
    GENERATIVE_IID_MANIFEST,
    GENERATIVE_IID_WEIGHTS,
    GenerativeIidInference,
    validate_generative_iid_checkpoint,
)
from pyhmsc.neural.generative_iid_mcmc import (
    bulk_ess_values,
    exact_model_log_joint,
    fixed_rademacher_projections,
    run_exact_model_mcmc,
    split_rhat_values,
)


def _dataset(
    seed: int = 880001,
    *,
    n_sites: int = 24,
    n_species: int = 12,
    loading: str = "medium",
    prevalence: str = "moderate",
    covariate: str = "normal",
    masked: bool = False,
):
    mask = (
        make_stratified_response_mask(n_sites, n_species, seed=seed)
        if masked
        else None
    )
    return simulate_generative_iid_dataset(
        n_sites=n_sites,
        n_species=n_species,
        covariate_shape=covariate,
        loading_stratum=loading,
        prevalence_stratum=prevalence,
        seed=seed,
        response_mask=mask,
    )


def _batch(dataset, *, max_sites=24, max_species=12):
    return batch_generative_iid_datasets(
        [dataset], max_sites=max_sites, max_species=max_species
    )


def test_simulator_matches_declared_prior_and_structural_truth():
    dataset = _dataset(880001)
    batch = _batch(dataset)
    layout = JointStateLayout(24, 12)
    truth = state_vector_from_truth(batch, layout)
    observed = float(
        generative_log_prior(
            truth,
            layout=layout,
            site_mask=batch.site_mask,
            species_mask=batch.species_mask,
        )[0, 0]
    )
    manual = norm.logpdf(dataset.truth_alpha, -0.50, 0.85)
    manual += norm.logpdf(dataset.truth_log_tau, math.log(0.65), 0.45)
    manual += np.sum(
        norm.logpdf(
            dataset.truth_beta[0], dataset.truth_alpha, 0.35
        )
    )
    manual += np.sum(norm.logpdf(dataset.truth_beta[1], 0.0, 0.50))
    manual += np.sum(norm.logpdf(dataset.truth_eta, 0.0, 1.0))
    manual += np.sum(
        norm.logpdf(
            dataset.truth_lambda,
            0.0,
            math.exp(dataset.truth_log_tau),
        )
    )

    np.testing.assert_allclose(
        dataset.truth_random_effect,
        dataset.truth_eta @ dataset.truth_lambda,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        dataset.truth_association,
        dataset.truth_lambda.T @ dataset.truth_lambda,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        dataset.truth_association_correlation,
        association_to_correlation(dataset.truth_association),
        rtol=2e-6,
        atol=2e-6,
    )
    assert observed == pytest.approx(manual, rel=2e-5)


def test_disposable_shape_factorial_is_generatable_without_opening_ledger_seeds():
    shapes = ((24, 12), (40, 36), (96, 75))
    combinations = itertools.product(
        ("normal", "right_skewed"),
        ("weak", "medium", "strong"),
        ("rare", "moderate", "common"),
    )
    datasets = []
    for index, combination in enumerate(combinations):
        n_sites, n_species = shapes[index % len(shapes)]
        datasets.append(
            _dataset(
                880100 + index,
                n_sites=n_sites,
                n_species=n_species,
                covariate=combination[0],
                loading=combination[1],
                prevalence=combination[2],
            )
        )

    assert len(datasets) == 18
    assert max(
        int(dataset.metadata["parameter_attempt"]) for dataset in datasets
    ) <= 512


def test_variable_shape_batch_padding_and_masks_are_exact():
    first = _dataset(880201, n_sites=24, n_species=12, masked=True)
    second = _dataset(
        880202,
        n_sites=40,
        n_species=36,
        loading="strong",
        prevalence="common",
    )
    batch = batch_generative_iid_datasets(
        [first, second], max_sites=40, max_species=36
    )

    assert batch.X.shape == (2, 40, 2)
    assert batch.Y.shape == (2, 40, 36)
    assert batch.site_mask.sum(axis=1).tolist() == [24, 40]
    assert batch.species_mask.sum(axis=1).tolist() == [12, 36]
    assert not batch.response_mask[0, :24, :12].all()
    assert not batch.response_mask[0, 24:].any()
    assert not batch.response_mask[0, :, 12:].any()
    np.testing.assert_array_equal(batch.Beta[0, :, :12], first.truth_beta)
    np.testing.assert_array_equal(batch.Eta[1, :40], second.truth_eta)


def test_joint_low_rank_log_prob_matches_dense_normal():
    mean = tf.constant([[0.2, -0.1, 0.4, 0.7]], dtype=tf.float32)
    diagonal = tf.constant([[0.7, 0.9, 0.5, 1.1]], dtype=tf.float32)
    factor = tf.constant(
        [[[0.2, -0.1], [0.0, 0.3], [0.1, 0.2], [-0.2, 0.1]]],
        dtype=tf.float32,
    )
    posterior = JointLowRankPosterior(
        mean=mean,
        diagonal_scale=diagonal,
        low_rank_factor=factor,
        state_mask=tf.ones_like(mean, dtype=tf.bool),
        layout=JointStateLayout(24, 12),
        site_mask=tf.ones((1, 24), dtype=tf.bool),
        species_mask=tf.ones((1, 12), dtype=tf.bool),
    )
    value = tf.constant(
        [[[0.5, -0.7, 0.1, 1.2]], [[-0.2, 0.3, 0.8, 0.4]]],
        dtype=tf.float32,
    )
    covariance = np.diag(np.square(diagonal.numpy()[0]))
    covariance += factor.numpy()[0] @ factor.numpy()[0].T
    expected = multivariate_normal.logpdf(
        value.numpy()[:, 0], mean=mean.numpy()[0], cov=covariance
    )

    np.testing.assert_allclose(
        posterior.log_prob(value).numpy()[:, 0],
        expected,
        rtol=2e-5,
        atol=2e-5,
    )


def test_iwae_loss_has_finite_gradients_for_all_trainable_variables():
    batch = _batch(_dataset(880301, masked=True))
    tf.keras.utils.set_random_seed(501900001)
    model = GenerativeIidPosteriorModel(max_sites=24, max_species=12)
    with tf.GradientTape() as tape:
        posterior = model(batch.model_inputs(), training=True)
        loss, diagnostics = importance_weighted_variational_loss(
            posterior,
            batch.model_inputs(),
            draws=2,
            kl_weight=0.25,
            seed=880302,
        )
    gradients = tape.gradient(loss, model.trainable_variables)

    assert np.isfinite(float(loss))
    assert np.isfinite(float(diagnostics["iwelbo"]))
    assert all(gradient is not None for gradient in gradients)
    assert all(
        bool(
            tf.reduce_all(
                tf.math.is_finite(tf.convert_to_tensor(gradient))
            )
        )
        for gradient in gradients
    )


def test_training_loop_updates_joint_model_without_nonfinite_values():
    datasets = [_dataset(880351), _dataset(880352, masked=True)]
    batch = batch_generative_iid_datasets(
        datasets, max_sites=24, max_species=12
    )
    model = GenerativeIidPosteriorModel(max_sites=24, max_species=12)
    before = [
        np.asarray(value).copy()
        for value in model(batch.model_inputs(), training=False)
        .mean[0:1]
    ]
    history = train_generative_iid_model(
        model,
        batch,
        epochs=1,
        batch_size=1,
        model_seed=880353,
        importance_draws=2,
    )
    after = np.asarray(
        model(batch.model_inputs(), training=False).mean[0]
    )

    assert len(history.loss) == 1
    assert np.isfinite(history.loss[0])
    assert np.isfinite(history.iwelbo[0])
    assert np.isfinite(history.gradient_norm[0])
    assert not np.array_equal(before[0], after)


def test_hidden_response_values_do_not_enter_encoder_or_likelihood():
    dataset = _dataset(880401, masked=True)
    batch = _batch(dataset)
    altered_y = batch.Y.copy()
    altered_y[~batch.response_mask] = 1.0 - altered_y[~batch.response_mask]
    altered_inputs = dict(batch.model_inputs())
    altered_inputs["Y"] = altered_y
    tf.keras.utils.set_random_seed(880402)
    model = GenerativeIidPosteriorModel(max_sites=24, max_species=12)
    original = model(batch.model_inputs(), training=False)
    altered = model(altered_inputs, training=False)
    samples = original.sample(2, seed=880403)
    original_ll = probit_log_likelihood(
        samples,
        layout=original.layout,
        X=batch.X,
        Y=batch.Y,
        response_mask=batch.response_mask,
        site_mask=batch.site_mask,
        species_mask=batch.species_mask,
    )
    altered_ll = probit_log_likelihood(
        samples,
        layout=original.layout,
        X=batch.X,
        Y=altered_y,
        response_mask=batch.response_mask,
        site_mask=batch.site_mask,
        species_mask=batch.species_mask,
    )

    np.testing.assert_allclose(original.mean, altered.mean, atol=1e-6)
    np.testing.assert_allclose(
        original.low_rank_factor, altered.low_rank_factor, atol=1e-6
    )
    np.testing.assert_allclose(original_ll, altered_ll, atol=1e-6)


def test_site_species_permutation_equivariance_of_invariant_means():
    dataset = _dataset(880501, masked=True)
    batch = _batch(dataset)
    site_order = np.random.default_rng(1).permutation(24)
    species_order = np.random.default_rng(2).permutation(12)
    permuted = {
        "X": batch.X[:, site_order],
        "Y": batch.Y[:, site_order][:, :, species_order],
        "response_mask": batch.response_mask[:, site_order][
            :, :, species_order
        ],
        "site_mask": batch.site_mask[:, site_order],
        "species_mask": batch.species_mask[:, species_order],
    }
    tf.keras.utils.set_random_seed(880502)
    model = GenerativeIidPosteriorModel(max_sites=24, max_species=12)
    original = posterior_mean_invariants(
        model(batch.model_inputs(), training=False)
    )
    moved = posterior_mean_invariants(model(permuted, training=False))
    site_inverse = np.argsort(site_order)
    species_inverse = np.argsort(species_order)

    np.testing.assert_allclose(
        moved["Beta"].numpy()[..., species_inverse],
        original["Beta"].numpy(),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        moved["R"].numpy()[:, site_inverse][:, :, species_inverse],
        original["R"].numpy(),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        moved["C"].numpy()[:, species_inverse][:, :, species_inverse],
        original["C"].numpy(),
        atol=2e-5,
    )


def test_padding_is_invariant_to_other_batch_members():
    first = _dataset(880601, n_sites=24, n_species=12, masked=True)
    second = _dataset(
        880602,
        n_sites=40,
        n_species=36,
        loading="strong",
        prevalence="common",
    )
    alone = batch_generative_iid_datasets(
        [first], max_sites=40, max_species=36
    )
    together = batch_generative_iid_datasets(
        [first, second], max_sites=40, max_species=36
    )
    tf.keras.utils.set_random_seed(880603)
    model = GenerativeIidPosteriorModel(max_sites=40, max_species=36)
    first_posterior = model(alone.model_inputs(), training=False)
    pair_posterior = model(together.model_inputs(), training=False)

    np.testing.assert_allclose(
        first_posterior.mean.numpy()[0],
        pair_posterior.mean.numpy()[0],
        atol=2e-5,
    )
    np.testing.assert_allclose(
        first_posterior.low_rank_factor.numpy()[0],
        pair_posterior.low_rank_factor.numpy()[0],
        atol=2e-5,
    )


def test_gauge_fix_preserves_random_effect_and_association():
    dataset = _dataset(880701)
    fixed_eta, fixed_lambda = gauge_fix_factors(
        dataset.truth_eta, dataset.truth_lambda
    )

    np.testing.assert_allclose(
        fixed_eta @ fixed_lambda,
        dataset.truth_random_effect,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        fixed_lambda.T @ fixed_lambda,
        dataset.truth_association,
        rtol=2e-6,
        atol=2e-6,
    )
    for row in fixed_lambda:
        assert row[np.argmax(np.abs(row))] >= 0.0


def test_exact_target_wrapper_matches_shared_log_density():
    dataset = _dataset(880801, masked=True)
    batch = _batch(dataset)
    layout = JointStateLayout(24, 12)
    truth = state_vector_from_truth(batch, layout)
    shared = generative_log_prior(
        truth,
        layout=layout,
        site_mask=batch.site_mask,
        species_mask=batch.species_mask,
    ) + probit_log_likelihood(
        truth,
        layout=layout,
        X=batch.X,
        Y=batch.Y,
        response_mask=batch.response_mask,
        site_mask=batch.site_mask,
        species_mask=batch.species_mask,
    )
    wrapped = exact_model_log_joint(truth, dataset)

    assert wrapped.shape == (1,)
    np.testing.assert_allclose(wrapped, shared[0], atol=1e-5)


def test_projection_and_chain_diagnostics_are_deterministic():
    first = fixed_rademacher_projections("R", 37)
    second = fixed_rademacher_projections("R", 37)
    other = fixed_rademacher_projections("C", 37)
    rng = np.random.default_rng(880901)
    values = rng.normal(size=(4, 500, 3))

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.max(split_rhat_values(values)) < 1.05
    assert np.min(bulk_ess_values(values)) > 100.0


@pytest.mark.slow
def test_exact_mcmc_runner_executes_independently_of_neural_model():
    dataset = _dataset(880951)
    result = run_exact_model_mcmc(
        dataset,
        chains=2,
        warmup=2,
        draws=4,
        seed=880952,
    )

    assert result.samples.shape == (2, 4, 98)
    assert result.acceptance_probability.shape == (4, 2)
    assert np.all(np.isfinite(result.samples))
    assert result.diagnostics["protocol"] == (
        "generative_neural_hmsc_iid_probit_v1"
    )


def test_checkpoint_roundtrip_is_exact_and_rejects_mutation(tmp_path):
    tf.keras.utils.set_random_seed(881001)
    inference = GenerativeIidInference.create(
        max_sites=24, max_species=12
    )
    batch = _batch(_dataset(881002, masked=True))
    expected = inference.model(batch.model_inputs(), training=False)
    checkpoint = inference.save(
        tmp_path / "checkpoint",
        source_commit="unit-test",
        source_provenance={
            "commit": "unit-test",
            "branch": "feature/generative-neural-hmsc",
            "worktree_dirty": True,
            "source_files": [
                {
                    "path": "pyhmsc/neural/generative_iid.py",
                    "sha256": "0" * 64,
                }
            ],
            "environment": {
                "python": "test",
                "tensorflow": "test",
                "tensorflow_probability": "test",
                "numpy": "test",
                "platform": "test",
            },
        },
        training_manifest={
            "role": "unit_test",
            "production_seed_ranges_opened": False,
        },
    )
    payload = validate_generative_iid_checkpoint(checkpoint)
    loaded = GenerativeIidInference.load(checkpoint)
    observed = loaded.model(batch.model_inputs(), training=False)

    assert payload["dependency_inventory"] == []
    assert payload["calibration"] is None
    assert payload["source_provenance"]["worktree_dirty"] is True
    np.testing.assert_allclose(expected.mean, observed.mean, atol=1e-6)
    np.testing.assert_allclose(
        expected.low_rank_factor, observed.low_rank_factor, atol=1e-6
    )

    weights = checkpoint / GENERATIVE_IID_WEIGHTS
    with weights.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="weights hash"):
        validate_generative_iid_checkpoint(checkpoint)


def test_legacy_iid_manifest_cannot_load_as_generative_checkpoint(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / GENERATIVE_IID_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "legacy_iid_svd_checkpoint",
                "model_family": "fixed_shape_iid_residual_svd",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact kind"):
        validate_generative_iid_checkpoint(root)


def test_sealed_harness_checks_hashes_and_refuses_unconfirmed_smoke(
    monkeypatch, tmp_path
):
    sealed_harness._validate_frozen_documents()
    monkeypatch.delenv(sealed_harness.CONFIRMATION_ENV, raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_generative_neural_hmsc_iid_v1.py",
            "--mode",
            "disposable-smoke",
            "--output",
            str(tmp_path / "run"),
        ],
    )

    with pytest.raises(RuntimeError, match=sealed_harness.CONFIRMATION_ENV):
        sealed_harness.main()
    assert not (tmp_path / "run").exists()


def test_production_factorial_and_seed_roles_are_frozen():
    cells = production_harness._factorial_cells()
    assert len(cells) == 324
    assert len(
        {
            (
                cell["n_sites"],
                cell["n_species"],
                cell["covariate_shape"],
                cell["loading_stratum"],
                cell["prevalence_stratum"],
                cell["replicate"],
            )
            for cell in cells
        }
    ) == 324
    assert production_harness.TRAINING_SEEDS == tuple(
        range(501_000_001, 501_000_325)
    )
    assert production_harness.FIXED_VALIDATION_SEEDS == tuple(
        range(502_000_001, 502_000_325)
    )
    assert production_harness.TRAINING_RESPONSES_PER_CONTEXT == 2
    assert production_harness.TRAINING_EPOCHS == 200
    assert production_harness.TRAINING_BATCH_SIZE == 4
    assert production_harness.MODEL_SEED == 501_900_001


def test_production_training_refuses_before_seed_generation(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(
        production_harness.TRAIN_CONFIRMATION_ENV,
        raising=False,
    )
    monkeypatch.setattr(
        production_harness,
        "_generate_production_block",
        lambda *args, **kwargs: pytest.fail("production seeds were generated"),
    )
    with pytest.raises(RuntimeError, match="OPEN_GENERATIVE_IID_501M"):
        production_harness.train_candidate(
            tmp_path / "blocked",
            expected_source_commit="not-opened",
        )
    assert not (tmp_path / "blocked").exists()


def test_production_seal_status_keeps_all_nontraining_blocks_closed():
    status = production_harness.production_seal_status()
    assert status["candidate_training_opened"] is False
    assert status["fixed_validation_opened"] is False
    assert status["fixed_validation_executable"] is True
    assert status["reserved_seed_ranges_opened"] is False
    assert status["redesign_seed_ranges_opened"] is False


def test_source_control_state_accepts_strict_clean_host_attestation(
    monkeypatch,
):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("container has no git")

    commit = "1" * 40
    monkeypatch.setattr(sealed_harness.subprocess, "run", unavailable)
    monkeypatch.setenv(sealed_harness.HOST_SOURCE_COMMIT_ENV, commit)
    monkeypatch.setenv(
        sealed_harness.HOST_SOURCE_BRANCH_ENV,
        "detached",
    )
    monkeypatch.setenv(sealed_harness.HOST_WORKTREE_CLEAN_ENV, "1")

    assert sealed_harness._source_control_state() == (
        commit,
        "detached",
        False,
    )
    assert production_harness._require_clean_pinned_source(commit) == commit


@pytest.mark.parametrize(
    ("commit", "clean"),
    [
        ("", "1"),
        ("not-a-full-commit", "1"),
        ("1" * 40, "0"),
        ("1" * 40, ""),
    ],
)
def test_source_control_state_rejects_invalid_host_attestation(
    monkeypatch,
    commit,
    clean,
):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("container has no git")

    monkeypatch.setattr(sealed_harness.subprocess, "run", unavailable)
    monkeypatch.setenv(sealed_harness.HOST_SOURCE_COMMIT_ENV, commit)
    monkeypatch.setenv(sealed_harness.HOST_WORKTREE_CLEAN_ENV, clean)
    with pytest.raises(RuntimeError, match="host-source attestation"):
        sealed_harness._source_control_state()


@pytest.mark.parametrize(
    "fixed_validation_key",
    [
        "fixed_validation_seed_ranges_opened",
        "fixed_validation_opened",
    ],
)
def test_seed_seal_validation_accepts_canonical_and_frozen_corpus_keys(
    fixed_validation_key,
):
    production_harness._require_false_seed_flags(
        {
            fixed_validation_key: False,
            "reserved_seed_ranges_opened": False,
            "redesign_seed_ranges_opened": False,
        }
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "reserved_seed_ranges_opened": False,
            "redesign_seed_ranges_opened": False,
        },
        {
            "fixed_validation_opened": True,
            "reserved_seed_ranges_opened": False,
            "redesign_seed_ranges_opened": False,
        },
        {
            "fixed_validation_opened": False,
            "fixed_validation_seed_ranges_opened": True,
            "reserved_seed_ranges_opened": False,
            "redesign_seed_ranges_opened": False,
        },
    ],
)
def test_seed_seal_validation_rejects_missing_or_open_fixed_validation(
    payload,
):
    with pytest.raises(ValueError, match="fixed_validation"):
        production_harness._require_false_seed_flags(payload)


def test_fixed_validation_preflight_binds_distinct_training_and_evaluator_commits(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(
        production_harness.VALIDATION_CONFIRMATION_ENV,
        raising=False,
    )
    observed = {}

    def require_evaluator(commit):
        observed["evaluator"] = commit
        return commit

    def validate_training(root, *, expected_source_commit, write_validation):
        observed["training"] = expected_source_commit
        assert root == tmp_path / "training"
        assert write_validation is False
        return {
            "freeze_sha256": "f" * 64,
            "checkpoint_content_sha256": "a" * 64,
            "no_latent_ablation_content_sha256": "b" * 64,
        }

    monkeypatch.setattr(
        production_harness,
        "_require_clean_pinned_source",
        require_evaluator,
    )
    monkeypatch.setattr(
        production_harness,
        "validate_training_freeze",
        validate_training,
    )
    result = production_harness.preflight_fixed_validation(
        tmp_path / "training",
        expected_training_source_commit="training-commit",
        expected_evaluator_source_commit="evaluator-commit",
        expected_checkpoint_content_sha256="a" * 64,
        expected_ablation_content_sha256="b" * 64,
    )

    assert observed == {
        "training": "training-commit",
        "evaluator": "evaluator-commit",
    }
    assert result["training_source_commit"] == "training-commit"
    assert result["evaluator_source_commit"] == "evaluator-commit"
    assert result["fixed_validation_seed_ranges_opened"] is False


def test_training_preflight_is_read_only_and_keeps_every_block_closed(
    monkeypatch,
):
    monkeypatch.delenv(
        production_harness.TRAIN_CONFIRMATION_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        production_harness.VALIDATION_CONFIRMATION_ENV,
        raising=False,
    )
    monkeypatch.setattr(
        production_harness,
        "_require_clean_pinned_source",
        lambda commit: commit,
    )
    result = production_harness.preflight_candidate_training(
        expected_source_commit="reviewed-clean-commit",
    )
    assert result["status"] == "candidate_training_preflight_sealed"
    assert result["source_commit"] == "reviewed-clean-commit"
    assert result["candidate_training_context_count"] == 324
    assert result["training_realization_count"] == 648
    assert result["candidate_training_opened"] is False
    assert result["fixed_validation_seed_ranges_opened"] is False
    assert result["reserved_seed_ranges_opened"] is False
    assert result["redesign_seed_ranges_opened"] is False
    assert {
        item["path"] for item in result["source_files"]
    } == set(production_harness.PRODUCTION_SOURCE_PATHS)


@pytest.mark.parametrize(
    "confirmation_env",
    [
        production_harness.TRAIN_CONFIRMATION_ENV,
        production_harness.VALIDATION_CONFIRMATION_ENV,
    ],
)
def test_training_preflight_refuses_when_an_opening_token_is_present(
    monkeypatch,
    confirmation_env,
):
    monkeypatch.delenv(
        production_harness.TRAIN_CONFIRMATION_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        production_harness.VALIDATION_CONFIRMATION_ENV,
        raising=False,
    )
    monkeypatch.setenv(confirmation_env, "must-not-be-present")
    monkeypatch.setattr(
        production_harness,
        "_require_clean_pinned_source",
        lambda commit: pytest.fail("preflight inspected source after token"),
    )
    with pytest.raises(RuntimeError, match="must remain unset"):
        production_harness.preflight_candidate_training(
            expected_source_commit="not-opened",
        )


def test_fixed_validation_refuses_before_any_502m_generation(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(
        production_harness.VALIDATION_CONFIRMATION_ENV,
        raising=False,
    )
    monkeypatch.setattr(
        production_harness,
        "_generate_fixed_validation_block",
        lambda: pytest.fail("502M seeds were generated"),
    )
    with pytest.raises(RuntimeError, match="OPEN_GENERATIVE_IID_502M"):
        production_harness.run_fixed_validation(
            tmp_path / "training",
            tmp_path / "evaluation",
            expected_training_source_commit="training-not-opened",
            expected_evaluator_source_commit="evaluator-not-opened",
            expected_checkpoint_content_sha256="a" * 64,
            expected_ablation_content_sha256="b" * 64,
            release_registry=tmp_path / "release",
            python="python3",
        )
    assert not (tmp_path / "evaluation").exists()
