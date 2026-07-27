import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.stats import multivariate_t
import tensorflow as tf

from pyhmsc.neural.inference import NeuralHmscCompatibilityError, NeuralHmscInference
from pyhmsc.neural.models import probit_irls_laplace_full_anchor
from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.student_t_inference import (
    M57_AUDIT_SHA256,
    M57_DECISION_SHA256,
    M57_PREREGISTRATION_SHA256,
    STUDENT_T_FEATURE_NAMES,
    STUDENT_T_OVERLAY_MANIFEST,
    STUDENT_T_OVERLAY_WEIGHTS,
    FixedProbitStudentTHead,
    FixedProbitStudentTInference,
    StudentTBetaPosterior,
    StudentTFeatureNormalizer,
    bivariate_student_t_negative_log_probability,
    fit_fixed_probit_student_t_overlay,
    sample_student_t_beta_posterior,
    student_t_features,
    student_t_posterior_from_raw,
    validate_bound_m56_negative,
    validate_bound_variable_v1,
    write_student_t_beta_posterior_hdf5,
)
from pyhmsc.neural.train import fixed_shape_training_data


ROOT = Path(__file__).resolve().parents[1]


def test_student_t_density_matches_scipy_multivariate_t():
    mean = tf.constant([[[0.2], [-0.4]]], dtype=tf.float32)
    marginal = tf.constant([[[0.7], [1.2]]], dtype=tf.float32)
    rho = 0.45
    covariance = np.asarray(
        [[0.7**2, rho * 0.7 * 1.2], [rho * 0.7 * 1.2, 1.2**2]],
        dtype=np.float32,
    )
    covariance_tril = tf.constant(
        np.linalg.cholesky(covariance)[None, None, ...], dtype=tf.float32
    )
    nu = tf.constant([[7.5]], dtype=tf.float32)
    student_scale = covariance_tril * tf.sqrt((nu - 2.0) / nu)[..., None, None]
    posterior = StudentTBetaPosterior(
        mean=mean,
        marginal_scale=marginal,
        covariance_tril=covariance_tril,
        student_t_scale_tril=student_scale,
        degrees_of_freedom=nu,
    )
    truth = tf.constant([[[0.8], [-1.1]]], dtype=tf.float32)

    observed = bivariate_student_t_negative_log_probability(posterior, truth)
    scipy_scale = covariance * ((7.5 - 2.0) / 7.5)
    expected = -multivariate_t.logpdf(
        [0.8, -1.1], loc=[0.2, -0.4], shape=scipy_scale, df=7.5
    )

    assert float(observed[0, 0]) == pytest.approx(expected, rel=2e-6)


def test_zero_initialized_head_is_exact_base_independent_student_t_nu10():
    base = BetaPosterior(
        mean=tf.constant([[[0.3, -0.2], [0.8, 0.1]]], dtype=tf.float32),
        scale=tf.constant([[[0.4, 0.7], [1.1, 0.6]]], dtype=tf.float32),
    )
    tf.keras.utils.set_random_seed(321900001)
    head = FixedProbitStudentTHead()
    raw = head(tf.ones((1, 2, 15), dtype=tf.float32))
    posterior = student_t_posterior_from_raw(base, raw)

    np.testing.assert_allclose(raw.numpy()[..., :5], 0.0, atol=0.0)
    np.testing.assert_allclose(posterior.mean, base.mean, atol=1e-7)
    np.testing.assert_allclose(posterior.marginal_scale, base.scale, atol=1e-7)
    np.testing.assert_allclose(posterior.degrees_of_freedom, 10.0, atol=1e-6)
    covariance = posterior.covariance_tril @ tf.transpose(
        posterior.covariance_tril, [0, 1, 3, 2]
    )
    assert np.max(np.abs(covariance.numpy()[..., 0, 1])) == pytest.approx(0.0)


def test_frozen_raw_transforms_are_bounded_and_covariance_consistent():
    base = BetaPosterior(
        mean=tf.zeros((1, 2, 2), dtype=tf.float32),
        scale=tf.constant([[[0.5, 0.8], [1.0, 1.4]]], dtype=tf.float32),
    )
    raw = tf.constant(
        [
            [
                [100.0, -100.0, 100.0, -100.0, 100.0, -100.0],
                [-100.0, 100.0, -100.0, 100.0, -100.0, 100.0],
            ]
        ],
        dtype=tf.float32,
    )
    posterior = student_t_posterior_from_raw(base, raw)
    covariance = posterior.covariance_tril @ tf.transpose(
        posterior.covariance_tril, [0, 1, 3, 2]
    )
    diagonal = tf.transpose(tf.sqrt(tf.linalg.diag_part(covariance)), [0, 2, 1])

    np.testing.assert_allclose(diagonal, posterior.marginal_scale, rtol=2e-6)
    assert np.min(np.linalg.eigvalsh(covariance.numpy())) > 0.0
    assert float(tf.reduce_min(posterior.degrees_of_freedom)) >= 2.1 - 1e-6
    assert float(tf.reduce_max(posterior.degrees_of_freedom)) <= 30.0 + 1e-6
    movement = np.abs(np.asarray(posterior.mean) - np.asarray(base.mean))
    assert np.all(movement <= 2.0 * np.asarray(base.scale) + 1e-6)
    multipliers = np.asarray(posterior.marginal_scale) / np.asarray(base.scale)
    assert np.all(multipliers >= np.exp(-1.5) - 1e-6)
    assert np.all(multipliers <= np.exp(1.5) + 1e-6)


def test_sampling_recovers_mean_marginal_sd_and_correlation():
    mean = tf.constant([[[0.2], [-0.5]]], dtype=tf.float32)
    marginal = tf.constant([[[0.8], [1.3]]], dtype=tf.float32)
    rho = 0.62
    covariance = np.asarray([[0.8**2, rho * 0.8 * 1.3], [rho * 0.8 * 1.3, 1.3**2]])
    covariance_tril = tf.constant(
        np.linalg.cholesky(covariance)[None, None], dtype=tf.float32
    )
    nu = tf.constant([[10.0]], dtype=tf.float32)
    posterior = StudentTBetaPosterior(
        mean=mean,
        marginal_scale=marginal,
        covariance_tril=covariance_tril,
        student_t_scale_tril=(
            covariance_tril * tf.sqrt((nu - 2.0) / nu)[..., None, None]
        ),
        degrees_of_freedom=nu,
    )

    samples = sample_student_t_beta_posterior(posterior, draws=10_000, seed=5702)
    values = samples[:, 0, :, 0]

    np.testing.assert_allclose(np.mean(values, axis=0), [0.2, -0.5], atol=0.03)
    np.testing.assert_allclose(np.std(values, axis=0), [0.8, 1.3], rtol=0.03)
    assert np.corrcoef(values.T)[0, 1] == pytest.approx(rho, abs=0.03)


def test_feature_order_values_and_population_normalizer():
    base, dataset = _base_and_dataset(5703)
    data = fixed_shape_training_data([dataset])
    base_posterior = base.predict_beta_posterior(data, calibrated=True)
    laplace = probit_irls_laplace_full_anchor(data.X, data.Y)
    features = student_t_features(data.X, data.Y, base_posterior, laplace)
    normalizer = StudentTFeatureNormalizer.fit(features)
    normalized = normalizer.transform(features).numpy()

    assert len(STUDENT_T_FEATURE_NAMES) == 15
    assert features.shape == (1, 75, 15)
    np.testing.assert_array_equal(features[..., 0], base_posterior.mean[:, 0, :])
    np.testing.assert_array_equal(features[..., 1], base_posterior.mean[:, 1, :])
    np.testing.assert_allclose(
        features[..., 2], np.log(np.asarray(base_posterior.scale)[:, 0, :])
    )
    assert np.all(np.isfinite(normalized))
    np.testing.assert_allclose(np.mean(normalized, axis=(0, 1)), 0.0, atol=3e-5)


def test_student_t_loss_has_finite_gradients_only_for_head():
    base, dataset = _base_and_dataset(5704)
    data = fixed_shape_training_data([dataset])
    base_posterior = base.predict_beta_posterior(data, calibrated=True)
    laplace = probit_irls_laplace_full_anchor(data.X, data.Y)
    features = student_t_features(data.X, data.Y, base_posterior, laplace)
    head = FixedProbitStudentTHead()
    with tf.GradientTape() as tape:
        raw = head(features)
        posterior = student_t_posterior_from_raw(base_posterior, raw)
        loss = tf.reduce_mean(
            bivariate_student_t_negative_log_probability(posterior, data.Beta)
        )
    head_count = len(head.trainable_variables)
    gradients = tape.gradient(
        loss, head.trainable_variables + base.model.trainable_variables
    )

    assert all(value is not None for value in gradients[:head_count])
    assert all(np.all(np.isfinite(value)) for value in gradients[:head_count])
    assert all(value is None for value in gradients[head_count:])


def test_hdf5_student_t_semantics_and_empirical_fixture(tmp_path):
    mean = tf.zeros((1, 2, 1), dtype=tf.float32)
    marginal = tf.constant([[[0.8], [1.3]]], dtype=tf.float32)
    covariance = np.asarray([[0.8**2, 0.62 * 0.8 * 1.3], [0.62 * 0.8 * 1.3, 1.3**2]])
    covariance_tril = tf.constant(
        np.linalg.cholesky(covariance)[None, None], dtype=tf.float32
    )
    nu = tf.constant([[10.0]], dtype=tf.float32)
    posterior = StudentTBetaPosterior(
        mean=mean,
        marginal_scale=marginal,
        covariance_tril=covariance_tril,
        student_t_scale_tril=(
            covariance_tril * tf.sqrt((nu - 2.0) / nu)[..., None, None]
        ),
        degrees_of_freedom=nu,
    )
    path = write_student_t_beta_posterior_hdf5(
        posterior,
        tmp_path / "posterior.h5",
        covariate_names=("Intercept", "TMG"),
        species_names=("sp1",),
        chains=1,
        draws=10_000,
        seed=5705,
    )

    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["Beta"])[0, :, :, 0]
        metadata = json.loads(handle.attrs["pyhmsc_metadata"])
        assert handle["StudentTDegreesOfFreedom"][0] == pytest.approx(10.0)
        assert handle["StudentTCovarianceCholesky"].shape == (1, 2, 2)
        assert handle["StudentTScaleCholesky"].shape == (1, 2, 2)

    assert metadata["posterior_family"] == "bivariate_student_t"
    np.testing.assert_allclose(np.std(values, axis=0), [0.8, 1.3], rtol=0.03)
    assert np.corrcoef(values.T)[0, 1] == pytest.approx(0.62, abs=0.03)


def test_overlay_roundtrip_and_all_binding_rejections(tmp_path, monkeypatch):
    base, dataset = _base_and_dataset(5706)
    paired = _paired_datasets(dataset)
    data = fixed_shape_training_data(paired)
    base_posterior = base.predict_beta_posterior(data, calibrated=True)
    laplace = probit_irls_laplace_full_anchor(data.X, data.Y)
    binding = {"release": "fixed"}
    variable = {"variable": "fixed"}
    m56 = {"negative": "fixed"}
    engine = FixedProbitStudentTInference.initialize(
        base,
        normalizer=StudentTFeatureNormalizer.fit(
            student_t_features(data.X, data.Y, base_posterior, laplace)
        ),
        base_binding=binding,
        variable_v1_binding=variable,
        m56_negative_binding=m56,
        model_seed=321900001,
    )
    artifact = engine.save(tmp_path / "overlay")
    fake_release = SimpleNamespace(load_checkpoint=lambda seed: base)
    monkeypatch.setattr(
        "pyhmsc.neural.student_t_inference.load_neural_hmsc_release",
        lambda root: fake_release,
    )
    monkeypatch.setattr(
        "pyhmsc.neural.student_t_inference.validate_bound_v0_1_release",
        lambda release: binding,
    )
    monkeypatch.setattr(
        "pyhmsc.neural.student_t_inference.validate_bound_variable_v1",
        lambda root: variable,
    )
    monkeypatch.setattr(
        "pyhmsc.neural.student_t_inference.validate_bound_m56_negative",
        lambda root: m56,
    )
    loaded = FixedProbitStudentTInference.load(
        artifact,
        registry_root=tmp_path / "release",
        variable_registry_root=tmp_path / "variable",
        m56_root=tmp_path / "m56",
    )
    first = engine.predict_beta_posterior(dataset)
    second = loaded.predict_beta_posterior(dataset)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.covariance_tril, second.covariance_tril)
    manifest = json.loads((artifact / STUDENT_T_OVERLAY_MANIFEST).read_text())
    assert _sha256(artifact / STUDENT_T_OVERLAY_WEIGHTS) == manifest["weights_sha256"]

    with (artifact / STUDENT_T_OVERLAY_WEIGHTS).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="weight hash mismatch"):
        FixedProbitStudentTInference.load(
            artifact,
            registry_root=tmp_path / "release",
            variable_registry_root=tmp_path / "variable",
            m56_root=tmp_path / "m56",
        )


def test_overlay_rejects_shape_formula_order_distribution_and_structure():
    base, dataset = _base_and_dataset(5707)
    paired = _paired_datasets(dataset)
    data = fixed_shape_training_data(paired)
    base_posterior = base.predict_beta_posterior(data, calibrated=True)
    laplace = probit_irls_laplace_full_anchor(data.X, data.Y)
    engine = FixedProbitStudentTInference.initialize(
        base,
        normalizer=StudentTFeatureNormalizer.fit(
            student_t_features(data.X, data.Y, base_posterior, laplace)
        ),
        base_binding={},
        variable_v1_binding={},
        m56_negative_binding={},
        model_seed=321900001,
    )
    X = np.column_stack([np.ones(40), dataset.X["TMG"]])
    Y = dataset.Y.to_numpy()
    common = {
        "X": X,
        "Y": Y,
        "formula": "~ TMG",
        "distribution": "probit",
        "covariate_names": ["Intercept", "TMG"],
    }
    with pytest.raises(NeuralHmscCompatibilityError, match="formula"):
        engine.predict_beta_posterior({**common, "formula": "~ other"})
    with pytest.raises(NeuralHmscCompatibilityError, match="probit"):
        engine.predict_beta_posterior({**common, "distribution": "poisson"})
    with pytest.raises(NeuralHmscCompatibilityError, match="order"):
        engine.predict_beta_posterior(
            {**common, "covariate_names": ["TMG", "Intercept"]}
        )
    with pytest.raises(NeuralHmscCompatibilityError, match="structural"):
        engine.predict_beta_posterior({**common, "traits": np.ones((75, 1))})
    with pytest.raises((ValueError, NeuralHmscCompatibilityError)):
        engine.predict_beta_posterior({**common, "X": X[:-1], "Y": Y[:-1]})


def test_paired_training_keeps_responses_together_and_trains_one_epoch():
    base, dataset = _base_and_dataset(5708)
    paired = _paired_datasets(dataset)
    engine, history = fit_fixed_probit_student_t_overlay(
        base,
        paired,
        base_binding={"release": "fixed"},
        variable_v1_binding={"variable": "fixed"},
        m56_negative_binding={"negative": "fixed"},
        model_seed=321900001,
        epochs=1,
        batch_contexts=1,
        learning_rate=0.0005,
    )

    assert np.isfinite(history["loss"][-1])
    assert engine.training_record["owning_context_count"] == 1
    assert engine.training_record["realization_count"] == 2
    with pytest.raises(ValueError, match="replicates 0 and 1"):
        fit_fixed_probit_student_t_overlay(
            base,
            paired[:1],
            base_binding={},
            variable_v1_binding={},
            m56_negative_binding={},
            model_seed=321900001,
            epochs=1,
        )


def test_frozen_protocol_hashes_seed_barriers_and_disposable_pairing():
    harness = _load_harness()
    assert _sha256(harness.DECISION_PATH) == M57_DECISION_SHA256
    assert _sha256(harness.AUDIT_PATH) == M57_AUDIT_SHA256
    assert _sha256(harness.PREREGISTRATION_PATH) == M57_PREREGISTRATION_SHA256
    with pytest.raises(PermissionError, match="exact confirmation"):
        harness._require_confirmation("wrong", harness.TRAIN_CONFIRMATION)
    harness._assert_seed_roles(
        range(391_000_001, 391_000_028),
        range(392_000_001, 392_000_028),
        production=False,
    )
    with pytest.raises(ValueError, match="seed roles"):
        harness._assert_seed_roles(
            range(321_000_001, 321_000_325),
            range(322_000_001, 322_000_325),
            production=False,
        )
    corpus = harness.build_m57_corpus(
        range(391_000_001, 391_000_028), paired=True, smoke=True
    )
    assert len(corpus) == 54
    assert {row.metadata["predictor_scale"] for row in corpus} == {1.0}
    for first, second in zip(corpus[::2], corpus[1::2]):
        assert (
            first.metadata["owning_context_seed"]
            == second.metadata["owning_context_seed"]
        )
        np.testing.assert_array_equal(first.X, second.X)
        np.testing.assert_array_equal(first.truth_beta, second.truth_beta)
        assert not np.array_equal(first.Y, second.Y)


def test_lumi_train_validation_wrapper_opens_only_321m_and_322m():
    wrapper = (
        ROOT / "docs/lumi_neural_hmsc_student_t_m57_train_validation_sbatch.sh"
    ).read_text(encoding="utf-8")

    assert "GENERATE_M57_STUDENT_T_TRAIN_VALIDATION" in wrapper
    assert "321000001-321000324" in wrapper
    assert "322000001-322000324" in wrapper
    assert "323000001-325000324" in wrapper
    assert "OPEN_M57_RESERVED_STUDENT_T_EVALUATION" not in wrapper
    assert "\n  train-validate \\\n" in wrapper
    assert "\n  evaluate \\\n" not in wrapper


@pytest.mark.skipif(
    not Path(
        "/private/tmp/neural_hmsc_variable_deployments/"
        "neural_hmsc_variable_probit_v1/baseline.json"
    ).exists(),
    reason="immutable local regression bundles are unavailable",
)
def test_immutable_variable_v1_and_m56_negative_hash_regressions():
    variable = validate_bound_variable_v1(
        "/private/tmp/neural_hmsc_variable_deployments"
    )
    m56 = validate_bound_m56_negative(
        "/private/tmp/neural_hmsc_m56_train_validation_20192218"
    )
    assert variable["baseline_id"] == "neural_hmsc_variable_probit_v1"
    assert m56["validation_passed"] is False


def _base_and_dataset(seed):
    harness = _load_harness()
    dataset = harness.simulate_m57_community(
        seed=seed,
        predictor_location=0.0,
        predictor_scale=1.0,
        prevalence_name="balanced",
        target_prevalence=0.30,
        effect_name="moderate",
        effect_magnitude=0.75,
        context_replicate=0,
        response_replicate=0,
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


def _paired_datasets(dataset):
    harness = _load_harness()
    metadata = dataset.metadata
    second = harness.simulate_m57_community(
        seed=metadata["seed"],
        predictor_location=metadata["predictor_location"],
        predictor_scale=metadata["predictor_scale"],
        prevalence_name=metadata["prevalence"],
        target_prevalence=metadata["target_prevalence"],
        effect_name=metadata["effect"],
        effect_magnitude=metadata["effect_magnitude"],
        context_replicate=metadata["context_replicate"],
        response_replicate=1,
    )
    return [dataset, second]


def _load_harness():
    path = ROOT / "examples/qualify_neural_hmsc_student_t.py"
    spec = importlib.util.spec_from_file_location("m57_student_t_harness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()
