import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyhmsc.compiler import compile_hmsc_model
from pyhmsc.neural import (
    NEURAL_CHECKPOINT_VERSION,
    NeuralHmscCompatibilityError,
    NeuralHmscInference,
    simulate_fixed_effect_dataset,
)


def test_public_neural_hmsc_api_saves_loads_and_infers(tmp_path):
    train = [
        simulate_fixed_effect_dataset(
            n_sites=12,
            n_species=2,
            distribution="normal",
            seed=1100 + idx,
        )
        for idx in range(2)
    ]
    test = simulate_fixed_effect_dataset(
        n_sites=12,
        n_species=2,
        distribution="normal",
        seed=1200,
    )
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=12,
        n_species=2,
        distribution="normal",
        formula="~ x1 + x2",
    )

    history = engine.fit(train, epochs=1, batch_size=1)
    checkpoint = engine.save(tmp_path / "checkpoint")
    loaded = NeuralHmscInference.load(checkpoint)
    fit = loaded.infer(
        test,
        draws=5,
        chains=1,
        seed=123,
        output=tmp_path / "posterior.h5",
    )

    assert history.loss
    assert (checkpoint / "neural_checkpoint.json").exists()
    assert fit.output_file == tmp_path / "posterior.h5"
    assert fit.beta_samples().shape == (1, 5, 3, 2)
    assert list(fit.beta_mean().index) == ["Intercept", "x1", "x2"]
    assert list(fit.beta_mean().columns) == ["sp1", "sp2"]
    assert fit.beta_ci()["lower"].shape == (3, 2)
    assert fit.predict_mean(test.X).shape == test.Y.shape
    assert loaded.predict_beta_posterior(test).mean.shape == (1, 3, 2)


def test_public_neural_hmsc_api_infers_from_compiled_artifact(tmp_path):
    dataset = simulate_fixed_effect_dataset(
        n_sites=10,
        n_species=2,
        distribution="normal",
        seed=1300,
    )
    compiled = compile_hmsc_model(
        Y=dataset.Y,
        X=dataset.X,
        formula="~ x1 + x2",
        distr="normal",
        chains=1,
        output=tmp_path / "compiled",
    )
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=10,
        n_species=2,
        distribution="normal",
        formula="~ x1 + x2",
    )

    report = engine.check_compatibility(compiled.init_json)
    fit = engine.infer(compiled.init_json, draws=4, chains=1, seed=1301)

    assert report["compatible"] is True
    assert report["dimensions"] == {"n_sites": 10, "n_covariates": 3, "n_species": 2}
    assert fit.beta_samples().shape == (1, 4, 3, 2)
    assert fit.predict_mean(dataset.X).shape == dataset.Y.shape


def test_public_neural_hmsc_api_rejects_unsupported_compiled_structures(tmp_path):
    Y = pd.DataFrame(np.ones((6, 2)), columns=["sp1", "sp2"])
    X = pd.DataFrame({"x1": np.linspace(0, 1, 6), "x2": np.linspace(1, 0, 6)})
    traits = pd.DataFrame({"body": [1.0, 2.0]}, index=["sp1", "sp2"])
    compiled = compile_hmsc_model(
        Y=Y,
        X=X,
        formula="~ x1 + x2",
        distr="normal",
        traits=traits,
        trait_formula="~ body",
        chains=1,
        output=tmp_path / "compiled_traits",
    )
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=6,
        n_species=2,
        distribution="normal",
        formula="~ x1 + x2",
    )

    with pytest.raises(NeuralHmscCompatibilityError, match="unsupported compiled features"):
        engine.infer(compiled.init_json, draws=3)


def test_public_neural_hmsc_checkpoint_manifest_is_versioned(tmp_path):
    engine = NeuralHmscInference.for_fixed_effects(n_sites=4, n_species=1)

    checkpoint = engine.save(tmp_path / "checkpoint")
    manifest = json.loads((checkpoint / "neural_checkpoint.json").read_text(encoding="utf-8"))

    assert manifest["checkpoint_version"] == NEURAL_CHECKPOINT_VERSION
    assert manifest["training_corpus_version"] == "0.1"
    assert manifest["model_family"] == "fixed_effect_beta"
    assert "limitations" in manifest


def test_public_neural_hmsc_load_rejects_unknown_checkpoint_version(tmp_path):
    engine = NeuralHmscInference.for_fixed_effects(n_sites=4, n_species=1)
    checkpoint = engine.save(tmp_path / "checkpoint")
    manifest_path = checkpoint / "neural_checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_version"] = "99.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NeuralHmscCompatibilityError, match="unsupported Neural-HMSC checkpoint version"):
        NeuralHmscInference.load(checkpoint)
