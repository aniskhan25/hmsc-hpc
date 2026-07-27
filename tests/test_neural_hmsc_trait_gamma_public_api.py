import json

import numpy as np
import pytest

from pyhmsc.compiler import compile_hmsc_model
from pyhmsc.neural import (
    TRAIT_GAMMA_BASELINE_ID,
    TraitGammaNeuralHmscInference,
    freeze_trait_gamma_baseline,
    finite_sample_conformal_quantile,
    load_trait_gamma_baseline,
    package_trait_gamma_calibration,
    simulate_trait_gamma_boundary_dataset,
    validate_trait_gamma_baseline,
)
from pyhmsc.neural.inference import NeuralHmscCompatibilityError
from pyhmsc.neural.trait_inference import _sha256


def _datasets(count, seed=5300):
    return [
        simulate_trait_gamma_boundary_dataset(
            n_sites=12,
            n_species=5,
            seed=seed + index,
            beta_residual_scale=0.15,
        )
        for index in range(count)
    ]


def _engine():
    return TraitGammaNeuralHmscInference.for_trait_gamma(
        n_sites=12,
        n_species=5,
        n_covariates=2,
        n_traits=1,
        hidden_units=(8,),
    )


def _calibrate(engine):
    engine.fit_calibration(
        _datasets(3, 5400),
        provenance={
            "corpus_id": "trait_gamma_test_calibration",
            "seeds": [5400, 5401, 5402],
            "independent_from_training": True,
        },
    )


def test_boundary_simulator_matches_compiler_scaled_trait_semantics(tmp_path):
    dataset = _datasets(1)[0]
    compiled = compile_hmsc_model(
        Y=dataset.Y,
        X=dataset.X,
        formula="~ TMG",
        distr="probit",
        output=tmp_path / "compiled",
        traits=dataset.traits,
        trait_formula="~ CN",
    )
    from pyhmsc.serialization import read_compiled_model

    metadata, arrays = read_compiled_model(compiled.init_json)
    assert metadata["formula"] == {"X": "~ TMG", "T": "~ CN"}
    assert metadata["names"]["traits"] == ["CN"]
    np.testing.assert_allclose(
        arrays["T"], dataset.trait_design[["CN"]].to_numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        arrays["X"][:, 1], dataset.X["TMG"].to_numpy(), atol=1e-6
    )


def test_trait_gamma_anchor_is_species_permutation_invariant():
    engine = _engine()
    dataset = _datasets(1)[0]
    original = engine.predict_gamma_posterior(dataset, calibrated=False)
    order = [3, 1, 4, 0, 2]
    mapping = {
        "X": np.column_stack([np.ones(len(dataset.X)), dataset.X["TMG"]]),
        "Y": dataset.Y.to_numpy()[:, order],
        "T": dataset.trait_design.to_numpy()[order],
        "distribution": "probit",
        "formula": "~ TMG",
        "trait_formula": "~ CN",
        "covariate_names": ["Intercept", "TMG"],
        "trait_names": ["CN"],
    }
    permuted = engine.predict_gamma_posterior(mapping, calibrated=False)
    np.testing.assert_allclose(original.mean, permuted.mean, atol=2e-5)
    np.testing.assert_allclose(original.scale, permuted.scale, atol=2e-5)


def test_checkpoint_compiled_inference_emits_beta_and_gamma(tmp_path):
    engine = _engine()
    engine.fit(_datasets(4), epochs=1, batch_size=2, seed=5500)
    _calibrate(engine)
    checkpoint = engine.save(tmp_path / "checkpoint")
    loaded = TraitGammaNeuralHmscInference.load(checkpoint)
    dataset = _datasets(1, 5600)[0]
    compiled = compile_hmsc_model(
        Y=dataset.Y,
        X=dataset.X,
        formula="~ TMG",
        distr="probit",
        output=tmp_path / "compiled",
        traits=dataset.traits,
        trait_formula="~ CN",
    )
    compatibility = loaded.check_compatibility(compiled.init_json)
    fit = loaded.infer(
        compiled.init_json,
        chains=2,
        draws=7,
        seed=5601,
        output=tmp_path / "posterior.h5",
    )
    assert compatibility["posterior_parameters"] == ["Beta", "Gamma"]
    assert compatibility["joint_posterior_coupling"] is False
    assert fit.beta_samples().shape == (2, 7, 2, 5)
    assert fit.gamma_samples().shape == (2, 7, 2, 1)
    assert list(fit.gamma_mean().columns) == ["CN"]


def test_trait_gamma_rejects_changed_formula_and_missing_traits(tmp_path):
    engine = _engine()
    dataset = _datasets(1)[0]
    with pytest.raises(NeuralHmscCompatibilityError, match="T formula"):
        engine.check_compatibility(
            {
                "X": np.column_stack([np.ones(12), dataset.X["TMG"]]),
                "Y": dataset.Y,
                "T": dataset.trait_design,
                "trait_formula": "~ wrong",
            }
        )
    compiled = compile_hmsc_model(
        Y=dataset.Y,
        X=dataset.X,
        formula="~ TMG",
        distr="probit",
        output=tmp_path / "no_traits",
    )
    with pytest.raises(NeuralHmscCompatibilityError, match="no trait structure"):
        engine.check_compatibility(compiled.init_json)


def test_checkpoint_rejects_calibration_tampering(tmp_path):
    engine = _engine()
    _calibrate(engine)
    checkpoint = engine.save(tmp_path / "checkpoint")
    calibration = checkpoint / "gamma_calibration.json"
    payload = json.loads(calibration.read_text())
    payload["scale_multiplier"] += 0.1
    calibration.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="artifact hash differs"):
        TraitGammaNeuralHmscInference.load(checkpoint)


def test_freeze_and_load_immutable_trait_gamma_baseline(tmp_path):
    engine = _engine()
    _calibrate(engine)
    checkpoint = engine.save(tmp_path / "candidate")
    report = tmp_path / "qualification.json"
    report.write_text(
        json.dumps(
            {
                "decision": "trait_gamma_probit_promoted",
                "all_gates_passed": True,
                "fixed_release_content_sha256": "fixed-digest",
                "variable_release_content_sha256": "variable-digest",
            }
        )
    )
    baseline = freeze_trait_gamma_baseline(
        registry_root=tmp_path / "registry",
        candidate_checkpoint=checkpoint,
        qualification_report=report,
        fixed_release_digest="fixed-digest",
        variable_release_digest="variable-digest",
    )
    manifest = validate_trait_gamma_baseline(baseline)
    loaded = load_trait_gamma_baseline(tmp_path / "registry")
    assert manifest["baseline_id"] == TRAIT_GAMMA_BASELINE_ID
    assert manifest["existing_releases_modified"] is False
    assert loaded.dimensions == engine.dimensions
    with pytest.raises(FileExistsError):
        freeze_trait_gamma_baseline(
            registry_root=tmp_path / "registry",
            candidate_checkpoint=checkpoint,
            qualification_report=report,
            fixed_release_digest="fixed-digest",
            variable_release_digest="variable-digest",
        )


def test_frozen_baseline_detects_inventory_mutation(tmp_path):
    engine = _engine()
    _calibrate(engine)
    checkpoint = engine.save(tmp_path / "candidate")
    report = tmp_path / "qualification.json"
    report.write_text(
        json.dumps(
            {
                "decision": "trait_gamma_probit_promoted",
                "all_gates_passed": True,
                "fixed_release_content_sha256": "fixed-digest",
                "variable_release_content_sha256": "variable-digest",
            }
        )
    )
    baseline = freeze_trait_gamma_baseline(
        registry_root=tmp_path / "registry",
        candidate_checkpoint=checkpoint,
        qualification_report=report,
        fixed_release_digest="fixed-digest",
        variable_release_digest="variable-digest",
    )
    weights = baseline / "checkpoint" / "weights.weights.h5"
    before = _sha256(weights)
    weights.write_bytes(weights.read_bytes() + b"tamper")
    assert _sha256(weights) != before
    with pytest.raises(ValueError, match="inventory differs"):
        validate_trait_gamma_baseline(baseline)


def test_prior_baseline_identifiers_remain_fixed():
    from pyhmsc.neural.release import NEURAL_HMSC_RELEASE_ID
    from pyhmsc.neural.variable_inference import VARIABLE_SHAPE_BASELINE_ID

    assert NEURAL_HMSC_RELEASE_ID == "neural_hmsc_v0_1"
    assert VARIABLE_SHAPE_BASELINE_ID == "neural_hmsc_variable_probit_v1"


def test_finite_sample_conformal_quantile_uses_upper_order_statistic():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert finite_sample_conformal_quantile(scores, target_coverage=0.6) == 0.3
    assert finite_sample_conformal_quantile(scores, target_coverage=0.8) == 0.4
    with pytest.raises(ValueError, match="non-empty and finite"):
        finite_sample_conformal_quantile([], target_coverage=0.95)


def test_package_conformal_calibration_preserves_weight_bytes(tmp_path):
    engine = _engine()
    source = engine.save(tmp_path / "source")
    source_hash = _sha256(source / "weights.weights.h5")
    engine.fit_calibration(
        _datasets(9, 5800),
        method="split_conformal_scalar_gamma_scale",
        provenance={
            "corpus_id": "trait_gamma_conformal_test",
            "seeds": list(range(5800, 5809)),
            "independent_from_training": True,
        },
    )
    packaged = package_trait_gamma_calibration(
        source,
        tmp_path / "packaged",
        calibration=engine.calibration,
        expected_weights_sha256=source_hash,
    )
    loaded = TraitGammaNeuralHmscInference.load(packaged)
    assert _sha256(packaged / "weights.weights.h5") == source_hash
    assert loaded.calibration.method == "split_conformal_scalar_gamma_scale"
