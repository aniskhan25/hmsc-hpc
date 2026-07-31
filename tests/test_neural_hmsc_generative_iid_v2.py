import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gammaln, logsumexp
import tensorflow as tf

from pyhmsc.neural.generative_iid import (
    JointStateLayout,
    batch_generative_iid_datasets,
    generative_log_prior,
    make_stratified_response_mask,
    probit_log_likelihood,
    simulate_generative_iid_dataset,
    state_vector_from_truth,
)
from pyhmsc.neural.generative_iid_artifact import (
    GenerativeIidInference,
)
from pyhmsc.neural.generative_iid_v2 import (
    GENERATIVE_IID_V1_ARTIFACT_SHA256,
    GENERATIVE_IID_V1_SOURCE_SHA256,
    GENERATIVE_IID_V2_REFINEMENT_STEPS,
    GenerativeIidOrbitPosteriorModel,
    MaskedLowRankStudentT,
    OrbitMatrixNormal,
    generative_iid_v2_log_joint,
    importance_weighted_orbit_loss,
)
from pyhmsc.neural.generative_iid_v2_artifact import (
    GENERATIVE_IID_V2_MANIFEST,
    GENERATIVE_IID_V2_WEIGHTS,
    GenerativeIidOrbitInference,
    validate_generative_iid_v2_checkpoint,
)


def _dataset(
    seed=981001,
    *,
    n_sites=24,
    n_species=12,
    masked=True,
    loading="medium",
    prevalence="moderate",
):
    response_mask = (
        make_stratified_response_mask(n_sites, n_species, seed=seed) if masked else None
    )
    return simulate_generative_iid_dataset(
        n_sites=n_sites,
        n_species=n_species,
        covariate_shape="normal",
        loading_stratum=loading,
        prevalence_stratum=prevalence,
        seed=seed,
        response_mask=response_mask,
    )


def _batch(dataset, *, max_sites=24, max_species=12):
    return batch_generative_iid_datasets(
        [dataset], max_sites=max_sites, max_species=max_species
    )


def _source_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_masked_low_rank_student_t_matches_dense_reference():
    dtype = tf.float64
    mean = tf.constant([[0.2, -0.1, 0.4, 7.0]], dtype=dtype)
    log_scale = tf.math.log(tf.constant([[0.7, 0.9, 0.5, 1.0]], dtype=dtype))
    factor = tf.constant(
        [[[0.2, -0.1], [0.0, 0.3], [0.1, 0.2], [0.0, 0.0]]],
        dtype=dtype,
    )
    mask = tf.constant([[True, True, True, False]])
    df = tf.constant([7.5], dtype=dtype)
    posterior = MaskedLowRankStudentT(
        mean=mean,
        log_diagonal_scale=log_scale,
        low_rank_factor=factor,
        degrees_of_freedom=df,
        mask=mask,
    )
    values = tf.constant(
        [[[0.5, -0.7, 0.1, -100.0]], [[-0.2, 0.3, 0.8, 100.0]]],
        dtype=dtype,
    )
    active_mean = mean.numpy()[0, :3]
    covariance = np.diag(np.exp(2.0 * log_scale.numpy()[0, :3]))
    covariance += factor.numpy()[0, :3] @ factor.numpy()[0, :3].T
    inverse = np.linalg.inv(covariance)
    logdet = np.linalg.slogdet(covariance)[1]
    dimension = 3
    expected = []
    for value in values.numpy()[:, 0, :3]:
        residual = value - active_mean
        quadratic = residual @ inverse @ residual
        expected.append(
            gammaln((7.5 + dimension) / 2.0)
            - gammaln(7.5 / 2.0)
            - 0.5 * (dimension * math.log(7.5 * math.pi) + logdet)
            - 0.5 * (7.5 + dimension) * math.log1p(quadratic / 7.5)
        )
    np.testing.assert_allclose(
        posterior.log_prob(values).numpy()[:, 0],
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    for seed in (981002, None):
        samples = posterior.sample(5, seed=seed).numpy()
        assert samples.shape == (5, 1, 4)
        np.testing.assert_array_equal(samples[:, 0, 3], mean.numpy()[0, 3])


def test_matrix_normal_base_density_matches_dense_kronecker_reference():
    dtype = tf.float64
    mean = tf.constant(
        [[[0.2, -0.1], [0.5, 0.3], [-0.2, 0.4], [8.0, -9.0]]],
        dtype=dtype,
    )
    diagonal = np.array([[0.7, 0.9, 0.6, 1.0]])
    factor_np = np.array([[[0.2, -0.1], [0.0, 0.3], [0.1, 0.2], [0.0, 0.0]]])
    posterior = OrbitMatrixNormal(
        mean=mean,
        log_diagonal_scale=tf.math.log(tf.constant(diagonal, dtype=dtype)),
        low_rank_factor=tf.constant(factor_np, dtype=dtype),
        row_mask=tf.constant([[True, True, True, False]]),
    )
    value = tf.constant(
        [[[[0.4, -0.3], [0.1, 0.8], [-0.5, 0.2], [100.0, 100.0]]]],
        dtype=dtype,
    )
    covariance = np.diag(np.square(diagonal[0, :3]))
    covariance += factor_np[0, :3] @ factor_np[0, :3].T
    residual = value.numpy()[0, 0, :3] - mean.numpy()[0, :3]
    expected = (
        -3.0 * math.log(2.0 * math.pi)
        - np.linalg.slogdet(covariance)[1]
        - 0.5
        * sum(
            residual[:, column] @ np.linalg.solve(covariance, residual[:, column])
            for column in range(2)
        )
    )
    assert posterior.base_log_prob(value).numpy()[0, 0] == pytest.approx(
        expected, abs=1e-6
    )


def test_analytic_o2_orbit_density_matches_4096_angle_quadrature():
    dtype = tf.float64
    mean_np = np.array([[[0.25, -0.15], [0.45, 0.30], [-0.20, 0.35]]])
    diagonal = np.array([[0.75, 0.65, 0.90]])
    factor_np = np.array([[[0.18, -0.08], [0.05, 0.22], [-0.12, 0.15]]])
    value_np = np.array([[[[0.40, -0.25], [0.15, 0.70], [-0.45, 0.10]]]])
    posterior = OrbitMatrixNormal(
        mean=tf.constant(mean_np, dtype=dtype),
        log_diagonal_scale=tf.math.log(tf.constant(diagonal, dtype=dtype)),
        low_rank_factor=tf.constant(factor_np, dtype=dtype),
        row_mask=tf.ones((1, 3), dtype=tf.bool),
    )
    observed = float(posterior.log_prob(value_np).numpy()[0, 0])
    covariance = np.diag(np.square(diagonal[0]))
    covariance += factor_np[0] @ factor_np[0].T
    inverse = np.linalg.inv(covariance)
    logdet = np.linalg.slogdet(covariance)[1]
    base_constant = -3.0 * math.log(2.0 * math.pi) - logdet
    angles = 2.0 * math.pi * np.arange(4096) / 4096.0
    component_log_density = []
    z = value_np[0, 0]
    for reflection in (1.0, -1.0):
        cosine = np.cos(angles)
        sine = np.sin(angles)
        matrices = np.stack(
            [
                np.stack([cosine, -sine * reflection], axis=-1),
                np.stack([sine, cosine * reflection], axis=-1),
            ],
            axis=-2,
        )
        rotated = np.einsum("ri,aij->arj", z, matrices)
        residual = rotated - mean_np[0]
        quadratic = np.einsum("ari,rs,asi->a", residual, inverse, residual)
        component_log_density.append(base_constant - 0.5 * quadratic)
    expected = logsumexp(np.concatenate(component_log_density)) - math.log(2 * 4096)
    assert observed == pytest.approx(expected, abs=1e-6)


def test_orbit_density_and_target_are_invariant_to_o2_action():
    batch = _batch(_dataset(981101, masked=True))
    layout = JointStateLayout(24, 12)
    state = tf.cast(state_vector_from_truth(batch, layout), tf.float64)[None, ...]
    parameters = layout.unpack(state)
    angle = tf.constant(0.73, tf.float64)
    orthogonal = tf.stack(
        [
            tf.stack([tf.cos(angle), -tf.sin(angle)]),
            tf.stack([tf.sin(angle), tf.cos(angle)]),
        ]
    )
    moved_eta = tf.einsum("kbni,ij->kbnj", parameters["Eta"], orthogonal)
    moved_lambda = tf.einsum(
        "ij,kbjs->kbis", tf.transpose(orthogonal), parameters["Lambda"]
    )
    moved = tf.concat(
        [
            parameters["alpha"][..., None],
            tf.reshape(parameters["Beta"], [1, 1, 24]),
            tf.reshape(moved_eta, [1, 1, 48]),
            tf.reshape(moved_lambda, [1, 1, 24]),
            parameters["log_tau"][..., None],
        ],
        axis=-1,
    )
    original_prior = generative_log_prior(
        state,
        layout=layout,
        site_mask=batch.site_mask,
        species_mask=batch.species_mask,
    )
    moved_prior = generative_log_prior(
        moved,
        layout=layout,
        site_mask=batch.site_mask,
        species_mask=batch.species_mask,
    )
    original_likelihood = probit_log_likelihood(
        state,
        layout=layout,
        **batch.model_inputs(),
    )
    moved_likelihood = probit_log_likelihood(
        moved,
        layout=layout,
        **batch.model_inputs(),
    )
    wrapped = generative_iid_v2_log_joint(
        state,
        batch.model_inputs(),
        layout=layout,
        site_mask=batch.site_mask,
        species_mask=batch.species_mask,
    )
    mean = tf.constant(
        [[[0.25, -0.15], [0.45, 0.30], [-0.20, 0.35]]],
        tf.float64,
    )
    orbit = OrbitMatrixNormal(
        mean=mean,
        log_diagonal_scale=tf.math.log(tf.constant([[0.75, 0.65, 0.90]], tf.float64)),
        low_rank_factor=tf.constant(
            [[[0.18, -0.08], [0.05, 0.22], [-0.12, 0.15]]],
            tf.float64,
        ),
        row_mask=tf.ones((1, 3), tf.bool),
    )
    latent_value = tf.constant(
        [[[[0.40, -0.25], [0.15, 0.70], [-0.45, 0.10]]]],
        tf.float64,
    )
    moved_latent = tf.einsum("dbri,ij->dbrj", latent_value, orthogonal)
    np.testing.assert_allclose(
        orbit.log_prob(latent_value),
        orbit.log_prob(moved_latent),
        atol=1e-6,
    )
    np.testing.assert_allclose(original_prior, moved_prior, atol=1e-9)
    np.testing.assert_allclose(original_likelihood, moved_likelihood, atol=1e-9)
    np.testing.assert_array_equal(wrapped, original_prior + original_likelihood)
    np.testing.assert_allclose(
        tf.einsum("kbni,kbis->kbns", parameters["Eta"], parameters["Lambda"]),
        tf.einsum("kbni,kbis->kbns", moved_eta, moved_lambda),
        atol=2e-5,
    )


def test_four_step_refinement_has_finite_end_to_end_gradients_and_monotone_accepts():
    batch = _batch(_dataset(981201, masked=True))
    tf.keras.utils.set_random_seed(981202)
    model = GenerativeIidOrbitPosteriorModel(max_sites=24, max_species=12)
    with tf.GradientTape() as tape:
        posterior = model(
            batch.model_inputs(),
            training=True,
            refine=True,
            refinement_seed=981203,
        )
        loss, diagnostics = importance_weighted_orbit_loss(
            posterior,
            batch.model_inputs(),
            draws=2,
            seed=981204,
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    assert len(posterior.refinement_trace) == len(GENERATIVE_IID_V2_REFINEMENT_STEPS)
    assert np.isfinite(float(loss))
    assert np.isfinite(float(diagnostics["iwelbo"]))
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(
        bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients
    )
    for record in posterior.refinement_trace:
        baseline = record["baseline_iwelbo"].numpy()
        accepted = record["accepted_iwelbo"].numpy()
        assert np.all(accepted >= baseline - 1e-6)


def test_hidden_responses_permutation_and_padding_preserve_v2_distribution():
    first = _dataset(981301, masked=True)
    second = _dataset(
        981302,
        n_sites=40,
        n_species=36,
        masked=False,
        loading="strong",
        prevalence="common",
    )
    alone = batch_generative_iid_datasets([first], max_sites=40, max_species=36)
    together = batch_generative_iid_datasets(
        [first, second], max_sites=40, max_species=36
    )
    altered = dict(alone.model_inputs())
    altered_y = altered["Y"].copy()
    altered_y[~altered["response_mask"]] = 1.0 - altered_y[~altered["response_mask"]]
    altered["Y"] = altered_y
    tf.keras.utils.set_random_seed(981303)
    model = GenerativeIidOrbitPosteriorModel(max_sites=40, max_species=36)
    base = model(alone.model_inputs(), refine=False)
    paired = model(together.model_inputs(), refine=False)
    hidden_changed = model(altered, refine=False)
    np.testing.assert_allclose(
        base.global_posterior.mean[0],
        paired.global_posterior.mean[0],
        atol=2e-5,
    )
    np.testing.assert_allclose(
        base.latent_row_features[0],
        paired.latent_row_features[0],
        atol=2e-5,
    )
    np.testing.assert_allclose(
        base.global_posterior.mean,
        hidden_changed.global_posterior.mean,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        base.latent_row_features,
        hidden_changed.latent_row_features,
        atol=2e-5,
    )

    compact = _batch(first)
    site_order = np.random.default_rng(4).permutation(24)
    species_order = np.random.default_rng(5).permutation(12)
    permuted_inputs = {
        "X": compact.X[:, site_order],
        "Y": compact.Y[:, site_order][:, :, species_order],
        "response_mask": compact.response_mask[:, site_order][:, :, species_order],
        "site_mask": compact.site_mask[:, site_order],
        "species_mask": compact.species_mask[:, species_order],
    }
    compact_model = GenerativeIidOrbitPosteriorModel(max_sites=24, max_species=12)
    original = compact_model(compact.model_inputs(), refine=False)
    moved = compact_model(permuted_inputs, refine=False)
    original_moments = original.invariant_moments()
    moved_moments = moved.invariant_moments()
    site_inverse = np.argsort(site_order)
    species_inverse = np.argsort(species_order)
    np.testing.assert_allclose(
        moved_moments["Beta"].numpy()[..., species_inverse],
        original_moments["Beta"].numpy(),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        moved_moments["R"].numpy()[:, site_inverse][:, :, species_inverse],
        original_moments["R"].numpy(),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        moved_moments["C"].numpy()[:, species_inverse][:, :, species_inverse],
        original_moments["C"].numpy(),
        atol=2e-5,
    )


def test_v2_checkpoint_roundtrip_rejects_v1_and_tampering(tmp_path):
    tf.keras.utils.set_random_seed(981401)
    inference = GenerativeIidOrbitInference.create(max_sites=24, max_species=12)
    batch = _batch(_dataset(981402))
    expected = inference.model(batch.model_inputs(), refine=False)
    checkpoint = inference.save(
        tmp_path / "v2",
        source_commit="ordinary-fixture",
        source_provenance={
            "branch": "feature/generative-neural-hmsc",
            "worktree_dirty": True,
            "source_files": [
                {
                    "path": "pyhmsc/neural/generative_iid_v2.py",
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
            "role": "ordinary_fixture",
            "ledger_seeds_opened": False,
        },
    )
    manifest = validate_generative_iid_v2_checkpoint(checkpoint)
    loaded = GenerativeIidOrbitInference.load(checkpoint)
    observed = loaded.model(batch.model_inputs(), refine=False)
    assert manifest["calibration"] is None
    assert manifest["dependency_inventory"] == []
    np.testing.assert_allclose(
        expected.global_posterior.mean,
        observed.global_posterior.mean,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        expected.global_posterior.low_rank_factor,
        observed.global_posterior.low_rank_factor,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        expected.latent_row_features,
        observed.latent_row_features,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        expected.conditional_latent_mean(expected.global_posterior.mean),
        observed.conditional_latent_mean(observed.global_posterior.mean),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        expected.latent_low_rank_factor,
        observed.latent_low_rank_factor,
        atol=1e-6,
    )

    v1 = GenerativeIidInference.create(max_sites=24, max_species=12)
    v1_checkpoint = v1.save(
        tmp_path / "v1",
        source_commit="ordinary-fixture",
        source_provenance={
            "branch": "feature/generative-neural-hmsc",
            "worktree_dirty": True,
            "source_files": [{"path": "v1", "sha256": "0" * 64}],
            "environment": {
                "python": "test",
                "tensorflow": "test",
                "tensorflow_probability": "test",
                "numpy": "test",
                "platform": "test",
            },
        },
        training_manifest={"role": "ordinary_fixture"},
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_generative_iid_v2_checkpoint(v1_checkpoint)

    with (checkpoint / GENERATIVE_IID_V2_WEIGHTS).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="weights hash"):
        validate_generative_iid_v2_checkpoint(checkpoint)


def test_v1_sources_remain_byte_identical_and_v2_schema_is_frozen():
    assert _source_sha256("pyhmsc/neural/generative_iid.py") == (
        GENERATIVE_IID_V1_SOURCE_SHA256
    )
    assert _source_sha256("pyhmsc/neural/generative_iid_artifact.py") == (
        GENERATIVE_IID_V1_ARTIFACT_SHA256
    )
    assert GENERATIVE_IID_V2_MANIFEST != "generative_iid_checkpoint.json"
    assert (
        json.loads(
            Path(
                "docs/generative_neural_hmsc_iid_v2_seed_reaudit_2026-07-31.json.md"
            ).read_text()
        )["authorization"]["disposable_593m_594m"]
        is False
    )


@pytest.mark.slow
def test_maximum_shape_sampling_and_density_do_not_materialize_state_covariance():
    dataset = _dataset(
        981501,
        n_sites=96,
        n_species=75,
        masked=False,
        loading="weak",
        prevalence="rare",
    )
    batch = _batch(dataset, max_sites=96, max_species=75)
    model = GenerativeIidOrbitPosteriorModel(max_sites=96, max_species=75)
    posterior = model(
        batch.model_inputs(),
        refine=True,
        refinement_seed=981502,
    )
    samples = posterior.sample(1, seed=981502)
    log_q = posterior.log_prob(samples)
    assert samples.shape == (1, 1, JointStateLayout(96, 75).size)
    assert np.all(np.isfinite(log_q.numpy()))
    assert posterior.global_posterior.low_rank_factor.shape == (
        1,
        152,
        16,
    )
    assert posterior.latent_low_rank_factor.shape == (1, 171, 16)
