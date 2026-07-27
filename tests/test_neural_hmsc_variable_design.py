import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

import pyhmsc
from pyhmsc.compiler import compile_hmsc_model
from pyhmsc.neural import (
    VARIABLE_DESIGN_CHECKPOINT_VERSION,
    VARIABLE_DESIGN_MODEL_FAMILY,
    NeuralHmscCompatibilityError,
    VariableDesignBetaPosteriorModel,
    VariableDesignNeuralHmscInference,
    variable_design_training_data,
)
from pyhmsc.neural.simulator import FixedEffectDataset
from pyhmsc.neural.variable_inference import validate_variable_shape_baseline
from pyhmsc.neural.release import load_neural_hmsc_release


FIXED_BASELINE_HASH = "affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8"
VARIABLE_BASELINE_HASH = (
    "badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9"
)


def test_variable_design_training_data_pads_all_dimensions():
    small = _dataset(n_sites=8, n_species=2, n_covariates=2, seed=5401)
    large = _dataset(n_sites=12, n_species=5, n_covariates=4, seed=5402)

    data = variable_design_training_data([small, large])

    assert data.X.shape == (2, 12, 4)
    assert data.Y.shape == (2, 12, 5)
    assert data.Beta.shape == (2, 4, 5)
    np.testing.assert_array_equal(data.site_mask.sum(axis=1), [8, 12])
    np.testing.assert_array_equal(data.species_mask.sum(axis=1), [2, 5])
    np.testing.assert_array_equal(data.covariate_mask.sum(axis=1), [2, 4])
    np.testing.assert_array_equal(data.X[0, 8:, :], 0.0)
    np.testing.assert_array_equal(data.X[0, :, 2:], 0.0)
    np.testing.assert_array_equal(data.Y[0, :, 2:], 0.0)
    assert data.covariate_names == (
        ("Intercept", "x1"),
        ("Intercept", "x1", "x2", "x3"),
    )


def test_variable_design_model_is_padding_invariant():
    model = VariableDesignBetaPosteriorModel(hidden_units=(12,))
    small = _dataset(n_sites=8, n_species=2, n_covariates=2, seed=5410)
    large = _dataset(n_sites=12, n_species=5, n_covariates=4, seed=5411)
    alone = variable_design_training_data([small])
    padded = variable_design_training_data([small, large])
    _activate_head(model, alone, seed=5412)

    alone_posterior = _predict(model, alone)
    padded_posterior = _predict(model, padded)

    np.testing.assert_allclose(
        alone_posterior.mean.numpy()[0],
        padded_posterior.mean.numpy()[0, :2, :2],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        alone_posterior.scale.numpy()[0],
        padded_posterior.scale.numpy()[0, :2, :2],
        atol=1e-6,
    )
    np.testing.assert_array_equal(padded_posterior.mean.numpy()[0, 2:, :], 0.0)
    np.testing.assert_array_equal(padded_posterior.scale.numpy()[0, 2:, :], 0.0)
    np.testing.assert_array_equal(padded_posterior.mean.numpy()[0, :, 2:], 0.0)
    np.testing.assert_array_equal(padded_posterior.scale.numpy()[0, :, 2:], 0.0)


def test_variable_design_model_respects_site_species_and_covariate_permutations():
    model = VariableDesignBetaPosteriorModel(hidden_units=(12,))
    dataset = _dataset(n_sites=14, n_species=4, n_covariates=5, seed=5420)
    data = variable_design_training_data([dataset])
    _activate_head(model, data, seed=5421)
    reference = _predict(model, data)

    site_order = np.array([8, 1, 11, 3, 5, 0, 13, 2, 12, 6, 9, 4, 10, 7])
    site_data = _replace_data(
        data,
        X=data.X[:, site_order, :],
        Y=data.Y[:, site_order, :],
        site_mask=data.site_mask[:, site_order],
    )
    site_posterior = _predict(model, site_data)
    np.testing.assert_allclose(
        reference.mean.numpy(), site_posterior.mean.numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        reference.scale.numpy(), site_posterior.scale.numpy(), atol=1e-6
    )

    species_order = np.array([2, 0, 3, 1])
    species_data = _replace_data(
        data,
        Y=data.Y[:, :, species_order],
        Beta=data.Beta[:, :, species_order],
        species_mask=data.species_mask[:, species_order],
    )
    species_posterior = _predict(model, species_data)
    np.testing.assert_allclose(
        reference.mean.numpy()[:, :, species_order],
        species_posterior.mean.numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reference.scale.numpy()[:, :, species_order],
        species_posterior.scale.numpy(),
        atol=1e-6,
    )

    covariate_order = np.array([0, 3, 1, 4, 2])
    names = tuple(data.covariate_names[0][index] for index in covariate_order)
    covariate_data = _replace_data(
        data,
        X=data.X[:, :, covariate_order],
        Beta=data.Beta[:, covariate_order, :],
        covariate_mask=data.covariate_mask[:, covariate_order],
        covariate_names=(names,),
    )
    covariate_posterior = _predict(model, covariate_data)
    np.testing.assert_allclose(
        reference.mean.numpy()[:, covariate_order, :],
        covariate_posterior.mean.numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reference.scale.numpy()[:, covariate_order, :],
        covariate_posterior.scale.numpy(),
        atol=1e-6,
    )


def test_variable_design_checkpoint_roundtrip_and_compiled_formula_provenance(
    tmp_path,
):
    engine = _engine()
    checkpoint = engine.save(tmp_path / "variable_design_checkpoint")
    loaded = VariableDesignNeuralHmscInference.load(checkpoint)
    dataset = _dataset(n_sites=16, n_species=4, n_covariates=4, seed=5430)
    compiled = compile_hmsc_model(
        Y=dataset.Y,
        X=dataset.X,
        formula="~ x1 + x2 + x3",
        distr="probit",
        chains=1,
        output=tmp_path / "compiled",
    )

    before = engine.predict_beta_posterior(compiled.init_json)
    after = loaded.predict_beta_posterior(compiled.init_json)
    report = loaded.check_compatibility(compiled.init_json)
    fit = loaded.infer(
        compiled.init_json,
        draws=5,
        chains=1,
        seed=5431,
        output=tmp_path / "posterior.h5",
    )
    manifest = json.loads(
        (checkpoint / "neural_checkpoint.json").read_text(encoding="utf-8")
    )
    weights = checkpoint / "weights.weights.h5"

    np.testing.assert_allclose(before.mean.numpy(), after.mean.numpy(), atol=1e-7)
    np.testing.assert_allclose(before.scale.numpy(), after.scale.numpy(), atol=1e-7)
    assert report["formula"] == "~ x1 + x2 + x3"
    assert report["covariate_names"] == ["Intercept", "x1", "x2", "x3"]
    assert report["dimensions"] == {
        "n_sites": 16,
        "n_covariates": 4,
        "n_species": 4,
    }
    assert fit.beta_samples().shape == (1, 5, 4, 4)
    assert manifest["checkpoint_version"] == VARIABLE_DESIGN_CHECKPOINT_VERSION
    assert manifest["model_family"] == VARIABLE_DESIGN_MODEL_FAMILY
    assert manifest["artifacts"]["weights"]["sha256"] == _sha256(weights)
    assert not hasattr(pyhmsc, "VariableDesignNeuralHmscInference")


def test_variable_design_checkpoint_rejects_weight_tampering(tmp_path):
    checkpoint = _engine().save(tmp_path / "candidate")
    weights = checkpoint / "weights.weights.h5"
    weights.write_bytes(weights.read_bytes() + b"tampered")

    with pytest.raises(NeuralHmscCompatibilityError, match="weight hash mismatch"):
        VariableDesignNeuralHmscInference.load(checkpoint)


def test_variable_design_training_and_calibration_roundtrip(tmp_path):
    engine = _engine()
    train = [
        _dataset(n_sites=8, n_species=2, n_covariates=2, seed=5435),
        _dataset(n_sites=12, n_species=5, n_covariates=4, seed=5436),
    ]
    calibration = [
        _dataset(n_sites=9, n_species=3, n_covariates=3, seed=5437),
        _dataset(n_sites=14, n_species=4, n_covariates=5, seed=5438),
    ]
    history = engine.fit(train, epochs=1, batch_size=2, seed=5439)
    fitted = engine.fit_calibration(
        calibration,
        provenance={
            "independent_from_training": True,
            "target_ecological_response_used": False,
            "seeds": [5437, 5438],
            "corpus_id": "test_variable_design_calibration",
        },
    )
    checkpoint = engine.save(tmp_path / "calibrated")
    loaded = VariableDesignNeuralHmscInference.load(checkpoint)
    raw = loaded.predict_beta_posterior(calibration[0], calibrated=False)
    calibrated = loaded.predict_beta_posterior(calibration[0])

    assert all(np.isfinite(value) for values in history.values() for value in values)
    assert loaded.calibration == fitted
    np.testing.assert_allclose(calibrated.mean.numpy(), raw.mean.numpy())
    np.testing.assert_allclose(
        calibrated.scale.numpy(),
        raw.scale.numpy() * fitted.scale_multiplier,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_intercept", "leading intercept"),
        ("rank_deficient", "rank deficient"),
        ("too_many_covariates", "outside checkpoint range"),
        ("wrong_distribution", "distribution must be 'probit'"),
    ],
)
def test_variable_design_compatibility_rejects_unsupported_inputs(mutation, message):
    engine = _engine()
    dataset = _dataset(n_sites=16, n_species=3, n_covariates=4, seed=5440)
    design = np.column_stack(
        [np.ones(len(dataset.X)), dataset.X[["x1", "x2", "x3"]].to_numpy()]
    )
    mapping = {
        "X": design,
        "Y": dataset.Y,
        "distribution": "probit",
        "formula": "~ x1 + x2 + x3",
        "covariate_names": ["Intercept", "x1", "x2", "x3"],
        "species_names": list(dataset.Y.columns),
    }
    if mutation == "missing_intercept":
        mapping["X"] = design.copy()
        mapping["X"][:, 0] = np.arange(len(design), dtype=float)
    elif mutation == "rank_deficient":
        mapping["X"] = design.copy()
        mapping["X"][:, 3] = mapping["X"][:, 2]
    elif mutation == "too_many_covariates":
        mapping["X"] = np.column_stack([design, design[:, 1], design[:, 2]])
        mapping["covariate_names"] = [
            "Intercept",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
        ]
    else:
        mapping["distribution"] = "poisson"

    with pytest.raises(NeuralHmscCompatibilityError, match=message):
        engine.check_compatibility(mapping)


def test_variable_design_changes_preserve_frozen_baseline_hashes():
    fixed_root = Path("/private/tmp/neural_hmsc_releases")
    variable_root = Path(
        "/private/tmp/neural_hmsc_variable_deployments/"
        "neural_hmsc_variable_probit_v1"
    )
    if not fixed_root.exists() or not variable_root.exists():
        pytest.skip("local immutable baseline registries are unavailable")

    fixed = load_neural_hmsc_release(fixed_root)
    variable = validate_variable_shape_baseline(variable_root)

    assert fixed.manifest["content_sha256"] == FIXED_BASELINE_HASH
    assert variable["content_sha256"] == VARIABLE_BASELINE_HASH


def _engine():
    return VariableDesignNeuralHmscInference.for_fixed_effects(
        min_sites=6,
        max_sites=24,
        min_species=2,
        max_species=8,
        min_covariates=2,
        max_covariates=5,
        hidden_units=(12,),
    )


def _dataset(*, n_sites, n_species, n_covariates, seed):
    rng = np.random.default_rng(seed)
    predictor_names = [f"x{index}" for index in range(1, n_covariates)]
    X = pd.DataFrame(
        rng.normal(size=(n_sites, n_covariates - 1)), columns=predictor_names
    )
    covariate_names = ["Intercept", *predictor_names]
    beta = rng.normal(scale=0.45, size=(n_covariates, n_species))
    beta[0] -= 0.35
    design = np.column_stack([np.ones(n_sites), X.to_numpy()])
    linear = design @ beta
    probability = ndtr(linear)
    Y = pd.DataFrame(
        rng.binomial(1, probability),
        columns=[f"sp{index + 1}" for index in range(n_species)],
    )
    truth = pd.DataFrame(beta, index=covariate_names, columns=Y.columns)
    return FixedEffectDataset(
        Y=Y,
        X=X,
        truth_beta=truth,
        linear_predictor=pd.DataFrame(linear, columns=Y.columns),
        metadata={
            "distribution": "probit",
            "formula": "~ " + " + ".join(predictor_names),
        },
    )


def _predict(model, data):
    return model(
        {
            "X": data.X,
            "Y": data.Y,
            "site_mask": data.site_mask,
            "species_mask": data.species_mask,
            "covariate_mask": data.covariate_mask,
        },
        training=False,
    )


def _activate_head(model, data, *, seed):
    _predict(model, data)
    rng = np.random.default_rng(seed)
    model.shared_projection.kernel.assign(
        rng.normal(scale=0.03, size=model.shared_projection.kernel.shape)
    )
    model.shared_projection.bias.assign(
        rng.normal(scale=0.01, size=model.shared_projection.bias.shape)
    )


def _replace_data(data, **changes):
    values = {
        "X": data.X,
        "Y": data.Y,
        "Beta": data.Beta,
        "site_mask": data.site_mask,
        "species_mask": data.species_mask,
        "covariate_mask": data.covariate_mask,
        "covariate_names": data.covariate_names,
    }
    values.update(changes)
    return type(data)(**values)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
