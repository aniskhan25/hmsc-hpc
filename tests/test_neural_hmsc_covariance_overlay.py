import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import tensorflow as tf

from pyhmsc.neural.covariance_inference import (
    CORRELATION_OVERLAY_MANIFEST,
    CORRELATION_OVERLAY_WEIGHTS,
    CorrelationFeatureNormalizer,
    FixedProbitCorrelationHead,
    FixedProbitCovarianceInference,
    bivariate_beta_negative_log_probability,
    correlation_features,
    correlation_from_raw_delta,
    covariance_scale_tril,
)
from pyhmsc.neural.inference import NeuralHmscCompatibilityError, NeuralHmscInference
from pyhmsc.neural.models import (
    probit_irls_laplace_anchor,
    probit_irls_laplace_full_anchor,
)
from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.storage import write_beta_posterior_hdf5
from pyhmsc.neural.train import fixed_shape_training_data


ROOT = Path(__file__).resolve().parents[1]


def test_full_laplace_anchor_preserves_legacy_mean_and_marginal_scale_exactly():
    rng = np.random.default_rng(910)
    X = np.stack([np.ones((2, 40)), rng.normal(size=(2, 40))], axis=-1).astype(
        np.float32
    )
    beta = rng.normal(size=(2, 2, 4)).astype(np.float32)
    probability = 0.5 * (
        1.0 + tf.math.erf(tf.einsum("bnk,bks->bns", X, beta) / np.sqrt(2.0))
    )
    Y = rng.binomial(1, probability.numpy()).astype(np.float32)

    legacy_mean, legacy_scale = probit_irls_laplace_anchor(X, Y)
    full = probit_irls_laplace_full_anchor(X, Y)

    np.testing.assert_array_equal(full.mean.numpy(), legacy_mean.numpy())
    np.testing.assert_array_equal(full.scale.numpy(), legacy_scale.numpy())
    covariance = full.scale_tril @ tf.transpose(full.scale_tril, [0, 1, 3, 2])
    np.testing.assert_allclose(
        tf.transpose(tf.sqrt(tf.linalg.diag_part(covariance)), [0, 2, 1]),
        full.scale,
        rtol=1e-6,
        atol=1e-7,
    )


def test_zero_initialized_head_reproduces_clipped_laplace_correlation():
    tf.keras.utils.set_random_seed(211900001)
    head = FixedProbitCorrelationHead()
    raw = head(tf.ones((3, 5, 9)))
    anchor = tf.constant([[-0.99, -0.4, 0.0, 0.6, 0.99]] * 3, dtype=tf.float32)

    delta, correlation = correlation_from_raw_delta(anchor, raw)

    np.testing.assert_array_equal(raw.numpy(), np.zeros((3, 5), dtype=np.float32))
    np.testing.assert_array_equal(delta.numpy(), np.zeros((3, 5), dtype=np.float32))
    np.testing.assert_allclose(
        correlation.numpy(),
        np.clip(anchor.numpy(), -0.979, 0.979),
        atol=2e-7,
    )


def test_covariance_reconstruction_preserves_marginals_and_is_positive_definite():
    scale = tf.constant([[[0.4], [1.7]]], dtype=tf.float32)
    correlation = tf.constant([[0.93]], dtype=tf.float32)

    scale_tril = covariance_scale_tril(scale, correlation)
    covariance = scale_tril @ tf.transpose(scale_tril, [0, 1, 3, 2])

    np.testing.assert_allclose(
        tf.linalg.diag_part(covariance).numpy()[0, 0],
        np.square([0.4, 1.7]),
        rtol=1e-6,
    )
    assert np.min(np.linalg.eigvalsh(covariance.numpy())) > 0.0


def test_bivariate_loss_matches_direct_numpy_formula():
    truth = np.asarray([[[0.4, -1.1], [0.2, 0.8]]], dtype=np.float32)
    mean = np.asarray([[[0.1, -0.7], [0.0, 0.5]]], dtype=np.float32)
    scale = np.asarray([[[0.7, 1.2], [0.4, 0.9]]], dtype=np.float32)
    rho = np.asarray([[0.35, -0.55]], dtype=np.float32)

    observed = bivariate_beta_negative_log_probability(truth, mean, scale, rho)
    expected = []
    for species in range(2):
        covariance = np.asarray(
            [
                [
                    scale[0, species, 0] ** 2,
                    rho[0, species] * np.prod(scale[0, species]),
                ],
                [
                    rho[0, species] * np.prod(scale[0, species]),
                    scale[0, species, 1] ** 2,
                ],
            ]
        )
        residual = truth[0, species] - mean[0, species]
        expected.append(
            0.5
            * (
                2.0 * np.log(2.0 * np.pi)
                + np.linalg.slogdet(covariance)[1]
                + residual @ np.linalg.inv(covariance) @ residual
            )
        )
    np.testing.assert_allclose(observed.numpy()[0], expected, rtol=1e-6)


def test_hdf5_sampling_roundtrip_recovers_requested_correlation(tmp_path):
    mean = tf.zeros((1, 2, 1), dtype=tf.float32)
    scale = tf.constant([[[0.8], [1.3]]], dtype=tf.float32)
    scale_tril = covariance_scale_tril(scale, tf.constant([[0.62]]))
    posterior = BetaPosterior(mean=mean, scale=scale, scale_tril=scale_tril)
    path = write_beta_posterior_hdf5(
        posterior,
        tmp_path / "posterior.h5",
        covariate_names=("Intercept", "TMG"),
        species_names=("sp1",),
        distribution="probit",
        formula="~ TMG",
        chains=1,
        draws=10_000,
        seed=5601,
    )

    with h5py.File(path, "r") as handle:
        draws = np.asarray(handle["Beta"])[0, :, :, 0]

    assert np.corrcoef(draws.T)[0, 1] == pytest.approx(0.62, abs=0.03)


def test_feature_order_and_training_only_normalizer_are_finite():
    base, dataset = _base_and_dataset(5610)
    data = fixed_shape_training_data([dataset])
    posterior = base.predict_beta_posterior(data)
    laplace = probit_irls_laplace_full_anchor(data.X, data.Y)

    features = correlation_features(data.X, data.Y, posterior, laplace)
    normalizer = CorrelationFeatureNormalizer.fit(features)
    normalized = normalizer.transform(features).numpy()

    assert features.shape == (1, 75, 9)
    assert np.all(np.isfinite(normalized))
    np.testing.assert_allclose(np.mean(normalized, axis=(0, 1)), 0.0, atol=2e-5)


def test_overlay_artifact_roundtrip_and_weight_hash_rejection(tmp_path, monkeypatch):
    base, dataset = _base_and_dataset(5620)
    data = fixed_shape_training_data([dataset])
    posterior = base.predict_beta_posterior(data)
    laplace = probit_irls_laplace_full_anchor(data.X, data.Y)
    normalizer = CorrelationFeatureNormalizer.fit(
        correlation_features(data.X, data.Y, posterior, laplace)
    )
    binding = {"fixture": "immutable"}
    engine = FixedProbitCovarianceInference.initialize(
        base,
        normalizer=normalizer,
        base_binding=binding,
        model_seed=211900001,
    )
    artifact = engine.save(tmp_path / "overlay")
    fake_release = SimpleNamespace(load_checkpoint=lambda seed: base)
    monkeypatch.setattr(
        "pyhmsc.neural.covariance_inference.load_neural_hmsc_release",
        lambda registry_root: fake_release,
    )
    monkeypatch.setattr(
        "pyhmsc.neural.covariance_inference.validate_bound_v0_1_release",
        lambda release: binding,
    )

    loaded = FixedProbitCovarianceInference.load(
        artifact, registry_root=tmp_path / "registry"
    )
    first = engine.predict_details(dataset)
    second = loaded.predict_details(dataset)
    np.testing.assert_array_equal(first.posterior.mean, second.posterior.mean)
    np.testing.assert_array_equal(first.posterior.scale, second.posterior.scale)
    np.testing.assert_array_equal(first.correlation, second.correlation)
    manifest = json.loads((artifact / CORRELATION_OVERLAY_MANIFEST).read_text())
    assert _sha256(artifact / CORRELATION_OVERLAY_WEIGHTS) == manifest["weights_sha256"]

    with (artifact / CORRELATION_OVERLAY_WEIGHTS).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="weight hash mismatch"):
        FixedProbitCovarianceInference.load(
            artifact, registry_root=tmp_path / "registry"
        )


def test_overlay_rejects_formula_order_and_structural_inputs():
    base, dataset = _base_and_dataset(5630)
    data = fixed_shape_training_data([dataset])
    posterior = base.predict_beta_posterior(data)
    laplace = probit_irls_laplace_full_anchor(data.X, data.Y)
    engine = FixedProbitCovarianceInference.initialize(
        base,
        normalizer=CorrelationFeatureNormalizer.fit(
            correlation_features(data.X, data.Y, posterior, laplace)
        ),
        base_binding={"fixture": "immutable"},
        model_seed=211900001,
    )
    X = np.column_stack([np.ones(40), dataset.X["TMG"]])
    Y = dataset.Y.to_numpy()

    with pytest.raises(NeuralHmscCompatibilityError, match="formula"):
        engine.predict_beta_posterior(
            {
                "X": X,
                "Y": Y,
                "formula": "~ other",
                "distribution": "probit",
                "covariate_names": ["Intercept", "TMG"],
            }
        )
    with pytest.raises(NeuralHmscCompatibilityError, match="structural inputs"):
        engine.predict_beta_posterior(
            {
                "X": X,
                "Y": Y,
                "formula": "~ TMG",
                "distribution": "probit",
                "covariate_names": ["Intercept", "TMG"],
                "traits": np.ones((75, 1)),
            }
        )


def test_frozen_protocol_hashes_and_seed_barriers(tmp_path):
    harness = _load_harness()
    assert _sha256(harness.PREREGISTRATION_PATH) == harness.M56_PREREGISTRATION_SHA256
    assert _sha256(harness.AUDIT_PATH) == harness.M56_AUDIT_SHA256
    with pytest.raises(PermissionError, match="exact confirmation"):
        harness._require_confirmation("wrong", harness.TRAIN_CONFIRMATION)
    harness._assert_seed_roles(
        range(291_000_001, 291_000_028),
        range(292_000_001, 292_000_028),
        production=False,
    )
    with pytest.raises(ValueError, match="production seed"):
        harness._assert_seed_roles(
            [211_000_001],
            [292_000_001],
            production=False,
        )
    corpus = harness.build_m56_corpus(range(291_000_001, 291_000_028), smoke=True)
    assert len(corpus) == 27
    assert {row.metadata["predictor_scale"] for row in corpus} == {1.0}
    assert [row.metadata["seed"] for row in corpus] == list(
        range(291_000_001, 291_000_028)
    )


def _base_and_dataset(seed):
    harness = _load_harness()
    dataset = harness.simulate_m56_community(
        seed=seed,
        predictor_location=0.0,
        predictor_scale=1.0,
        prevalence_name="balanced",
        target_prevalence=0.30,
        effect_name="moderate",
        effect_magnitude=0.75,
        replicate=0,
    )
    base = NeuralHmscInference.for_fixed_effects(
        n_sites=40,
        n_species=75,
        n_covariates=2,
        distribution="probit",
        formula="~ TMG",
        covariate_names=("Intercept", "TMG"),
        species_names=tuple(dataset.Y.columns),
        hidden_units=(8,),
        probit_anchor="irls_laplace",
    )
    return base, dataset


def _load_harness():
    path = ROOT / "examples/qualify_neural_hmsc_covariance.py"
    spec = importlib.util.spec_from_file_location("m56_covariance_harness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()
