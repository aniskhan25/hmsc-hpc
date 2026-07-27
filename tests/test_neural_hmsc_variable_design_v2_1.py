import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from pyhmsc.neural import (
    GATED_VARIABLE_DESIGN_CHECKPOINT_VERSION,
    GATED_VARIABLE_DESIGN_MODEL_FAMILY,
    GatedVariableDesignBetaPosteriorModel,
    GatedVariableDesignNeuralHmscInference,
    NeuralHmscCompatibilityError,
    VariableDesignNeuralHmscInference,
    variable_design_predictive_auxiliary_data,
    variable_design_probit_score_loss,
    variable_design_training_data,
)
from pyhmsc.neural.models import probit_irls_laplace_anchor
from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.simulator import FixedEffectDataset


SCRIPT = (
    Path(__file__).parents[1]
    / "examples/qualify_neural_hmsc_variable_design_v2_1.py"
)
SPEC = importlib.util.spec_from_file_location("m54_v2_1_qualification", SCRIPT)
M54 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M54)


def test_gated_model_starts_at_exact_anchor_with_bounded_gate():
    dataset = _dataset(n_sites=12, n_species=3, n_covariates=4, seed=54101)
    data = variable_design_training_data([dataset])
    model = GatedVariableDesignBetaPosteriorModel(hidden_units=(12,))

    posterior = _predict(model, data)
    anchor_mean, anchor_scale = probit_irls_laplace_anchor(
        data.X,
        data.Y,
        site_mask=data.site_mask,
    )
    gate = _gate(model, data).numpy()

    np.testing.assert_allclose(posterior.mean.numpy(), anchor_mean.numpy(), atol=1e-7)
    np.testing.assert_allclose(
        posterior.scale.numpy(), anchor_scale.numpy(), atol=1e-7
    )
    np.testing.assert_allclose(gate, 0.5, atol=1e-7)
    assert model.shared_projection.units == 3


def test_gated_model_preserves_padding_and_permutation_properties():
    model = GatedVariableDesignBetaPosteriorModel(hidden_units=(12,))
    small = _dataset(n_sites=8, n_species=2, n_covariates=2, seed=54110)
    large = _dataset(n_sites=14, n_species=5, n_covariates=5, seed=54111)
    alone = variable_design_training_data([small])
    padded = variable_design_training_data([small, large])
    _activate_head(model, alone, seed=54112)

    reference = _predict(model, alone)
    reference_gate = _gate(model, alone).numpy()
    padded_posterior = _predict(model, padded)
    padded_gate = _gate(model, padded).numpy()
    np.testing.assert_allclose(
        reference.mean.numpy()[0], padded_posterior.mean.numpy()[0, :2, :2], atol=1e-6
    )
    np.testing.assert_allclose(
        reference.scale.numpy()[0],
        padded_posterior.scale.numpy()[0, :2, :2],
        atol=1e-6,
    )
    np.testing.assert_allclose(reference_gate[0], padded_gate[0, :2, :2], atol=1e-6)
    np.testing.assert_array_equal(padded_gate[0, 2:, :], 0.0)
    np.testing.assert_array_equal(padded_gate[0, :, 2:], 0.0)

    full = variable_design_training_data([large])
    _activate_head(model, full, seed=54113)
    reference = _predict(model, full)
    reference_gate = _gate(model, full).numpy()
    site_order = np.array([8, 1, 11, 3, 5, 0, 13, 2, 12, 6, 9, 4, 10, 7])
    site_data = _replace_data(
        full,
        X=full.X[:, site_order, :],
        Y=full.Y[:, site_order, :],
        site_mask=full.site_mask[:, site_order],
    )
    np.testing.assert_allclose(
        reference.mean.numpy(), _predict(model, site_data).mean.numpy(), atol=1e-6
    )
    np.testing.assert_allclose(reference_gate, _gate(model, site_data).numpy(), atol=1e-6)

    species_order = np.array([2, 0, 4, 3, 1])
    species_data = _replace_data(
        full,
        Y=full.Y[:, :, species_order],
        Beta=full.Beta[:, :, species_order],
        species_mask=full.species_mask[:, species_order],
    )
    np.testing.assert_allclose(
        reference.mean.numpy()[:, :, species_order],
        _predict(model, species_data).mean.numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reference_gate[:, :, species_order],
        _gate(model, species_data).numpy(),
        atol=1e-6,
    )

    covariate_order = np.array([0, 3, 1, 4, 2])
    names = tuple(full.covariate_names[0][index] for index in covariate_order)
    covariate_data = _replace_data(
        full,
        X=full.X[:, :, covariate_order],
        Beta=full.Beta[:, covariate_order, :],
        covariate_mask=full.covariate_mask[:, covariate_order],
        covariate_names=(names,),
    )
    np.testing.assert_allclose(
        reference.mean.numpy()[:, covariate_order, :],
        _predict(model, covariate_data).mean.numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reference_gate[:, covariate_order, :],
        _gate(model, covariate_data).numpy(),
        atol=1e-6,
    )


def test_predictive_auxiliary_pairs_are_independent_with_shared_truth():
    contexts = [
        _dataset(n_sites=12, n_species=2, n_covariates=2, seed=54201),
        _dataset(n_sites=18, n_species=4, n_covariates=5, seed=54202),
    ]
    heldouts = [
        _heldout(contexts[0], seed=54301),
        _heldout(contexts[1], seed=54302),
    ]

    paired = variable_design_predictive_auxiliary_data(contexts, heldouts)

    assert paired.context_seeds == (54201, 54202)
    assert paired.heldout_seeds == (54301, 54302)
    np.testing.assert_array_equal(paired.contexts.Beta, paired.heldouts.Beta)
    assert not np.array_equal(paired.contexts.X, paired.heldouts.X)
    assert not np.shares_memory(paired.contexts.Y, paired.heldouts.Y)

    changed = replace_dataset_truth(heldouts[0], delta=0.1)
    with pytest.raises(ValueError, match="coefficient truth differs"):
        variable_design_predictive_auxiliary_data(
            contexts, [changed, heldouts[1]]
        )


def test_frozen_probit_score_loss_matches_manual_calculation():
    heldout = _dataset(n_sites=10, n_species=2, n_covariates=3, seed=54401)
    data = variable_design_training_data([heldout])
    mean = np.array(
        [[[-0.2, 0.1], [0.3, -0.4], [0.15, 0.2]]], dtype=np.float32
    )
    scale = np.full_like(mean, 0.25)
    posterior = BetaPosterior(mean=mean, scale=scale)

    loss, brier, log_loss = variable_design_probit_score_loss(
        posterior, data, indices=[0]
    )

    design = data.X[0]
    linear_mean = design @ mean[0]
    linear_variance = np.square(design) @ np.square(scale[0])
    probability = ndtr(linear_mean / np.sqrt(1.0 + linear_variance))
    response = data.Y[0]
    expected_brier = np.mean(np.square(response - probability))
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    expected_log_loss = -np.mean(
        response * np.log(clipped) + (1.0 - response) * np.log(1.0 - clipped)
    )
    assert float(brier.numpy()) == pytest.approx(expected_brier, rel=1e-6)
    assert float(log_loss.numpy()) == pytest.approx(expected_log_loss, rel=1e-6)
    assert float(loss.numpy()) == pytest.approx(
        0.5 * expected_brier + 0.5 * expected_log_loss, rel=1e-6
    )


def test_gated_training_calibration_and_checkpoint_roundtrip(tmp_path):
    coefficient_train = [
        _dataset(n_sites=8, n_species=2, n_covariates=2, seed=54501),
        _dataset(n_sites=12, n_species=4, n_covariates=4, seed=54502),
    ]
    contexts = [
        _dataset(n_sites=8, n_species=2, n_covariates=2, seed=54601),
        _dataset(n_sites=12, n_species=4, n_covariates=4, seed=54602),
    ]
    heldouts = [_heldout(contexts[0], seed=54701), _heldout(contexts[1], seed=54702)]
    calibration = [
        _dataset(n_sites=9, n_species=3, n_covariates=3, seed=54801),
        _dataset(n_sites=14, n_species=4, n_covariates=5, seed=54802),
    ]
    auxiliary = variable_design_predictive_auxiliary_data(contexts, heldouts)
    engine = _engine()

    history = engine.fit(
        coefficient_train,
        predictive_auxiliary=auxiliary,
        epochs=1,
        batch_size=2,
        seed=54901,
    )
    fitted = engine.fit_calibration(
        calibration,
        provenance={
            "independent_from_training": True,
            "target_ecological_response_used": False,
            "seeds": [54801, 54802],
            "corpus_id": "test_m54_v2_1_calibration",
        },
    )
    checkpoint = engine.save(tmp_path / "gated")
    loaded = GatedVariableDesignNeuralHmscInference.load(checkpoint)
    before = engine.predict_beta_posterior(calibration[0])
    after = loaded.predict_beta_posterior(calibration[0])
    before_gate = engine.predict_support_gate(calibration[0])
    after_gate = loaded.predict_support_gate(calibration[0])
    manifest = json.loads(
        (checkpoint / "neural_checkpoint.json").read_text(encoding="utf-8")
    )

    assert all(np.isfinite(value) for values in history.values() for value in values)
    assert loaded.calibration == fitted
    np.testing.assert_allclose(before.mean.numpy(), after.mean.numpy(), atol=1e-7)
    np.testing.assert_allclose(before.scale.numpy(), after.scale.numpy(), atol=1e-7)
    np.testing.assert_allclose(before_gate.numpy(), after_gate.numpy(), atol=1e-7)
    assert manifest["checkpoint_version"] == GATED_VARIABLE_DESIGN_CHECKPOINT_VERSION
    assert manifest["model_family"] == GATED_VARIABLE_DESIGN_MODEL_FAMILY
    assert manifest["model"]["projection_outputs"] == 3
    assert manifest["training_objective"]["predictive_weight"] == 1.0
    assert manifest["training_objective"]["coefficient_mse_weight"] == 0.25
    with pytest.raises(
        NeuralHmscCompatibilityError, match="unsupported variable-design checkpoint"
    ):
        VariableDesignNeuralHmscInference.load(checkpoint)


def test_gated_training_rejects_objective_or_seed_role_drift():
    coefficient = [_dataset(n_sites=8, n_species=2, n_covariates=2, seed=55001)]
    context = [_dataset(n_sites=8, n_species=2, n_covariates=2, seed=55101)]
    heldout = [_heldout(context[0], seed=55201)]
    auxiliary = variable_design_predictive_auxiliary_data(context, heldout)
    engine = _engine()

    with pytest.raises(ValueError, match="MSE weight is frozen"):
        engine.fit(
            coefficient,
            predictive_auxiliary=auxiliary,
            epochs=1,
            batch_size=1,
            mse_weight=0.2,
            seed=55301,
        )
    with pytest.raises(ValueError, match="predictive auxiliary weight is frozen"):
        engine.fit(
            coefficient,
            predictive_auxiliary=auxiliary,
            epochs=1,
            batch_size=1,
            predictive_weight=0.5,
            seed=55301,
        )


def test_v2_1_seed_roles_confirmation_barriers_and_preregistration(tmp_path):
    blocks = M54._seed_blocks(M54.PRODUCTION_STARTS, M54.PRODUCTION_COUNT)
    M54._assert_protocol_seed_roles(blocks, production=True)
    assert all(len(block) == 243 for block in blocks.values())
    assert M54._validate_preregistration() == M54.PREREGISTRATION_SHA256

    smoke = M54._seed_blocks(M54.SMOKE_STARTS, M54.SMOKE_COUNT)
    M54._assert_protocol_seed_roles(smoke, production=False)
    assert all(len(block) == 27 for block in smoke.values())

    train_output = tmp_path / "train_must_not_exist"
    with pytest.raises(ValueError, match="train/aux/calibration confirmation"):
        M54.train_and_freeze(
            argparse.Namespace(
                confirmation="wrong",
                output=train_output,
                fixed_registry=tmp_path / "fixed",
                variable_baseline=tmp_path / "variable",
            )
        )
    assert not train_output.exists()

    evaluation_output = tmp_path / "evaluation_must_not_exist"
    with pytest.raises(ValueError, match="reserved-evaluation confirmation"):
        M54.evaluate_frozen_candidate(
            argparse.Namespace(
                confirmation="wrong",
                freeze_root=tmp_path / "missing",
                output=evaluation_output,
                fixed_registry=tmp_path / "fixed",
                variable_baseline=tmp_path / "variable",
            )
        )
    assert not evaluation_output.exists()


def _engine():
    return GatedVariableDesignNeuralHmscInference.for_fixed_effects(
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
    return FixedEffectDataset(
        Y=Y,
        X=X,
        truth_beta=pd.DataFrame(beta, index=covariate_names, columns=Y.columns),
        linear_predictor=pd.DataFrame(linear, columns=Y.columns),
        metadata={
            "distribution": "probit",
            "formula": "~ " + " + ".join(predictor_names),
            "seed": int(seed),
            "target_condition": 2.0,
            "prevalence_stratum": "balanced",
            "effect_stratum": "moderate",
            "strata": {
                "site": f"site_{n_sites}",
                "species": f"species_{n_species}",
                "covariate": f"covariate_{n_covariates}",
                "design_condition": "condition_0",
            },
        },
    )


def _heldout(context, *, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=context.X.shape), columns=context.X.columns
    )
    design = np.column_stack([np.ones(len(X)), X.to_numpy()])
    linear = design @ context.truth_beta.to_numpy(dtype=float)
    probability = ndtr(linear)
    Y = pd.DataFrame(
        rng.binomial(1, probability), columns=context.Y.columns
    )
    return FixedEffectDataset(
        Y=Y,
        X=X,
        truth_beta=context.truth_beta.copy(),
        linear_predictor=pd.DataFrame(linear, columns=Y.columns),
        metadata={
            **context.metadata,
            "seed": int(seed),
            "paired_context_seed": int(context.metadata["seed"]),
            "predictive_heldout": True,
        },
    )


def replace_dataset_truth(dataset, *, delta):
    return FixedEffectDataset(
        Y=dataset.Y.copy(),
        X=dataset.X.copy(),
        truth_beta=dataset.truth_beta + float(delta),
        linear_predictor=dataset.linear_predictor.copy(),
        metadata=dict(dataset.metadata),
    )


def _predict(model, data):
    return model(_inputs(data), training=False)


def _gate(model, data):
    return model.support_gate(_inputs(data))


def _inputs(data):
    return {
        "X": data.X,
        "Y": data.Y,
        "site_mask": data.site_mask,
        "species_mask": data.species_mask,
        "covariate_mask": data.covariate_mask,
    }


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
