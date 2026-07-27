import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from pyhmsc.compiler import compile_hmsc_model
from pyhmsc.neural import (
    NEURAL_CHECKPOINT_VERSION,
    VARIABLE_CHECKPOINT_VERSION,
    NeuralHmscCompatibilityError,
    VariableShapeBetaCalibration,
    VariableShapeNeuralHmscInference,
    freeze_variable_shape_baseline,
    load_variable_shape_baseline,
    simulate_fixed_effect_dataset,
    validate_variable_shape_baseline,
)
from pyhmsc.neural.train import variable_shape_training_data


def test_variable_public_api_round_trips_calibration_and_compiled_inference(tmp_path):
    engine = _engine()
    engine.calibration = _calibration()
    checkpoint = engine.save(tmp_path / "variable_checkpoint")
    loaded = VariableShapeNeuralHmscInference.load(checkpoint)
    dataset = simulate_fixed_effect_dataset(
        n_sites=9, n_species=3, distribution="probit", seed=8100
    )
    compiled = compile_hmsc_model(
        Y=dataset.Y,
        X=dataset.X,
        formula="~ x1 + x2",
        distr="probit",
        chains=1,
        output=tmp_path / "compiled",
    )

    raw = loaded.predict_beta_posterior(compiled.init_json, calibrated=False)
    calibrated = loaded.predict_beta_posterior(compiled.init_json)
    report = loaded.check_compatibility(compiled.init_json)
    fit = loaded.infer(
        compiled.init_json,
        draws=6,
        chains=1,
        seed=8101,
        output=tmp_path / "posterior.h5",
    )
    manifest = json.loads(
        (checkpoint / "neural_checkpoint.json").read_text(encoding="utf-8")
    )

    np.testing.assert_allclose(calibrated.mean.numpy(), raw.mean.numpy())
    np.testing.assert_allclose(
        calibrated.scale.numpy(), raw.scale.numpy() * 1.25, rtol=1e-6
    )
    assert report["dimensions"] == {
        "n_sites": 9,
        "n_covariates": 3,
        "n_species": 3,
    }
    assert fit.beta_samples().shape == (1, 6, 3, 3)
    assert manifest["checkpoint_version"] == VARIABLE_CHECKPOINT_VERSION
    assert manifest["model_family"] == "variable_shape_fixed_effect_beta"
    assert manifest["shape_range"] == {
        "n_sites": [6, 12],
        "n_species": [2, 5],
        "n_covariates": [3, 3],
    }
    assert NEURAL_CHECKPOINT_VERSION == "0.4"


def test_variable_public_api_is_invariant_to_batch_padding():
    engine = _engine()
    small = simulate_fixed_effect_dataset(
        n_sites=7, n_species=2, distribution="probit", seed=8200
    )
    large = simulate_fixed_effect_dataset(
        n_sites=12, n_species=5, distribution="probit", seed=8201
    )
    single = engine.predict_beta_posterior(small, calibrated=False)
    padded = engine.predict_beta_posterior(
        variable_shape_training_data([small, large]), calibrated=False
    )

    np.testing.assert_allclose(
        single.mean.numpy()[0], padded.mean.numpy()[0, :, :2], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        single.scale.numpy()[0],
        padded.scale.numpy()[0, :, :2],
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("n_sites", "n_species", "message"),
    [
        (5, 2, "site count 5 is outside"),
        (13, 2, "site count 13 is outside"),
        (8, 1, "species count 1 is outside"),
        (8, 6, "species count 6 is outside"),
    ],
)
def test_variable_public_api_rejects_shapes_outside_declared_range(
    n_sites, n_species, message
):
    dataset = simulate_fixed_effect_dataset(
        n_sites=n_sites,
        n_species=n_species,
        distribution="probit",
        seed=8300 + n_sites + n_species,
    )

    with pytest.raises(NeuralHmscCompatibilityError, match=message):
        _engine().predict_beta_posterior(dataset)


def test_variable_public_api_rejects_traits_and_formula_changes(tmp_path):
    engine = _engine()
    Y = pd.DataFrame(np.ones((8, 2)), columns=["sp1", "sp2"])
    X = pd.DataFrame({"x1": np.arange(8), "x2": np.arange(8) / 2})
    traits = pd.DataFrame({"body": [1.0, 2.0]}, index=Y.columns)
    compiled_traits = compile_hmsc_model(
        Y=Y,
        X=X,
        formula="~ x1 + x2",
        distr="probit",
        traits=traits,
        trait_formula="~ body",
        chains=1,
        output=tmp_path / "compiled_traits",
    )

    with pytest.raises(
        NeuralHmscCompatibilityError, match="unsupported compiled features"
    ):
        engine.check_compatibility(compiled_traits.init_json)
    with pytest.raises(NeuralHmscCompatibilityError, match="formula"):
        engine.predict_beta_posterior(
            {
                "X": np.column_stack([np.ones(8), X.to_numpy()]),
                "Y": Y,
                "distribution": "probit",
                "formula": "~ x2 + x1",
                "covariate_names": ["Intercept", "x1", "x2"],
                "species_names": list(Y.columns),
            }
        )


def test_variable_public_api_rejects_tampered_calibration(tmp_path):
    engine = _engine()
    engine.calibration = _calibration()
    checkpoint = engine.save(tmp_path / "variable_checkpoint")
    artifact = checkpoint / "variable_coefficient_calibration.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["calibration"]["scale_multiplier"] = 9.0
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        NeuralHmscCompatibilityError, match="calibration artifact hash mismatch"
    ):
        VariableShapeNeuralHmscInference.load(checkpoint)


def test_variable_shape_baseline_freeze_is_immutable_and_loadable(tmp_path):
    engine = _engine()
    engine.calibration = _calibration()
    checkpoint = engine.save(tmp_path / "candidate")
    qualification = _write_multiseed_qualification(tmp_path, checkpoint)

    path = freeze_variable_shape_baseline(
        registry_root=tmp_path / "registry",
        candidate_checkpoint=checkpoint,
        qualification_root=qualification,
    )
    payload = validate_variable_shape_baseline(path)
    loaded = load_variable_shape_baseline(tmp_path / "registry")

    assert payload["baseline_id"] == "neural_hmsc_variable_probit_v1"
    assert payload["fixed_release_id"] == "neural_hmsc_v0_1"
    assert payload["fixed_release_modified"] is False
    assert loaded.shape_range == _engine().shape_range
    assert loaded.calibration.scale_multiplier == pytest.approx(1.25)
    with pytest.raises(FileExistsError, match="already exists"):
        freeze_variable_shape_baseline(
            registry_root=tmp_path / "registry",
            candidate_checkpoint=checkpoint,
            qualification_root=qualification,
        )


def test_variable_shape_baseline_rejects_checkpoint_mutation(tmp_path):
    engine = _engine()
    engine.calibration = _calibration()
    checkpoint = engine.save(tmp_path / "candidate")
    qualification = _write_multiseed_qualification(tmp_path, checkpoint)
    path = freeze_variable_shape_baseline(
        registry_root=tmp_path / "registry",
        candidate_checkpoint=checkpoint,
        qualification_root=qualification,
    )
    (path.parent / "checkpoint/weights.weights.h5").write_bytes(b"changed")

    with pytest.raises(ValueError, match="baseline hash mismatch"):
        validate_variable_shape_baseline(path)


def _engine():
    return VariableShapeNeuralHmscInference.for_fixed_effects(
        min_sites=6,
        max_sites=12,
        min_species=2,
        max_species=5,
    )


def _calibration():
    return VariableShapeBetaCalibration(
        scale_multiplier=1.25,
        n_coefficients=120,
        provenance={
            "kind": "independent_variable_shape_simulation_calibration",
            "target_ecological_response_used": False,
            "shape_selection_role": "predeclared_range",
            "seeds": [8001, 8002],
            "corpus_id": "test_variable_shape_calibration_v1",
        },
    )


def _write_multiseed_qualification(tmp_path, checkpoint):
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    runs = []
    for seed in (20260730, 20260731, 20260732):
        run_root = tmp_path / f"run_{seed}"
        run_root.mkdir()
        report = run_root / "variable_shape_qualification.json"
        report.write_text(json.dumps({"seed": seed, "passed": True}), encoding="utf-8")
        report.with_suffix(".md").write_text("# Passed\n", encoding="utf-8")
        runs.append(
            {
                "base_seed": seed,
                "decision": "variable_shape_probit_qualified",
                "all_gates_passed": True,
                "report_path": str(report),
                "report_sha256": _sha256(report),
            }
        )
    aggregate = {
        "kind": "neural_hmsc_variable_shape_multiseed_qualification",
        "decision": "variable_shape_probit_promoted",
        "all_runs_passed": True,
        "candidate_selected_using_sensitivity_outcomes": False,
        "fixed_release_modified": False,
        "candidate_checkpoint": {
            "manifest_sha256": _sha256(checkpoint / "neural_checkpoint.json"),
            "weights_sha256": _sha256(checkpoint / "weights.weights.h5"),
        },
        "runs": runs,
    }
    (qualification / "variable_shape_multiseed_qualification.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )
    (qualification / "variable_shape_multiseed_qualification.md").write_text(
        "# Promoted\n", encoding="utf-8"
    )
    return qualification


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
