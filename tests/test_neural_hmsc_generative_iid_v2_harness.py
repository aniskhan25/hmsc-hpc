import json
from pathlib import Path

import pytest

from examples import run_generative_neural_hmsc_iid_v2 as harness
from pyhmsc.neural.generative_iid import simulate_generative_iid_dataset


def _clear_opening_tokens(monkeypatch):
    for name in tuple(harness.os.environ):
        if name.startswith("OPEN_GENERATIVE_IID"):
            monkeypatch.delenv(name, raising=False)


def test_v2_disposable_factorial_and_seed_roles_are_frozen():
    cells = harness._factorial_cells()
    assert len(cells) == 18
    assert (
        len(
            {
                (
                    cell["n_sites"],
                    cell["n_species"],
                    cell["covariate_shape"],
                    cell["loading_stratum"],
                    cell["prevalence_stratum"],
                )
                for cell in cells
            }
        )
        == 18
    )
    assert harness.TRAINING_SEEDS == tuple(range(593_000_001, 593_000_019))
    assert harness.VALIDATION_SEEDS == tuple(range(594_000_001, 594_000_019))
    assert harness.MODEL_SEED == 511_900_001
    assert harness.SMOKE_EPOCHS == 2
    assert not hasattr(harness, "PRODUCTION_SEEDS")
    assert not hasattr(harness, "RESERVED_SEEDS")


def test_v2_disposable_harness_validates_all_frozen_documents():
    harness._validate_frozen_documents()
    assert set(harness.SOURCE_PATHS) == {
        "pyhmsc/neural/generative_iid.py",
        "pyhmsc/neural/generative_iid_mcmc.py",
        "pyhmsc/neural/generative_iid_v2.py",
        "pyhmsc/neural/generative_iid_v2_artifact.py",
        "pyhmsc/neural/__init__.py",
        "examples/run_generative_neural_hmsc_iid_v2.py",
        ("docs/generative_neural_hmsc_iid_v2_orbit_" "preregistration_2026-07-31.md"),
        ("docs/generative_neural_hmsc_iid_v2_seed_" "reaudit_2026-07-31.json.md"),
        ("docs/generative_neural_hmsc_iid_v2_representation_" "decision_2026-07-31.md"),
        ("docs/generative_neural_hmsc_iid_v2_" "implementation_2026-07-31.md"),
        ("docs/generative_neural_hmsc_iid_v2_" "numerical_review_2026-08-01.md"),
    }
    inventory = {
        item["path"]: item["sha256"] for item in harness._source_file_inventory()
    }
    assert inventory["pyhmsc/neural/generative_iid_v2.py"] == (
        "87828857ee1718a8825a1a15e7af99abe49a86ee4d179f6cbce6591162aa71bc"
    )
    assert inventory[
        "docs/generative_neural_hmsc_iid_v2_numerical_review_2026-08-01.md"
    ] == harness.NUMERICAL_REVIEW_SHA256


def test_token_free_preflight_is_read_only_and_seed_sealed(monkeypatch):
    _clear_opening_tokens(monkeypatch)
    commit = "1" * 40
    monkeypatch.setattr(
        harness,
        "_require_clean_pinned_source",
        lambda expected: expected,
    )
    monkeypatch.setattr(
        harness,
        "_generate_block",
        lambda *args, **kwargs: pytest.fail("preflight generated a disposable dataset"),
    )
    result = harness.preflight_disposable_smoke(expected_source_commit=commit)

    assert result["status"] == ("generative_iid_v2_disposable_preflight_sealed")
    assert result["source_commit"] == commit
    assert result["factorial_cell_count"] == 18
    assert result["factorial_unique_cell_count"] == 18
    assert result["simulation_generation_called"] is False
    assert result["output_created"] is False
    assert result["disposable_seed_ranges_opened"] is False
    assert result["production_511m_opened"] is False
    assert result["fixed_validation_512m_opened"] is False
    assert result["reserved_513m_515m_opened"] is False
    assert result["authorization_required"] is True
    assert {item["path"] for item in result["source_files"]} == set(
        harness.SOURCE_PATHS
    )


def test_preflight_refuses_any_opening_token_before_source_or_seed_access(
    monkeypatch,
):
    _clear_opening_tokens(monkeypatch)
    monkeypatch.setenv(
        harness.CONFIRMATION_ENV,
        harness.CONFIRMATION_VALUE,
    )
    monkeypatch.setattr(
        harness,
        "_validate_frozen_documents",
        lambda: pytest.fail("preflight crossed token boundary"),
    )
    monkeypatch.setattr(
        harness,
        "_require_clean_pinned_source",
        lambda *args: pytest.fail("preflight inspected source"),
    )
    with pytest.raises(RuntimeError, match="must remain unset"):
        harness.preflight_disposable_smoke(expected_source_commit="1" * 40)


def test_disposable_execution_refuses_before_output_or_seed_generation(
    monkeypatch, tmp_path
):
    _clear_opening_tokens(monkeypatch)
    monkeypatch.setattr(
        harness,
        "_generate_block",
        lambda *args, **kwargs: pytest.fail(
            "blocked execution generated a disposable dataset"
        ),
    )
    output = tmp_path / "blocked"
    with pytest.raises(RuntimeError, match=harness.CONFIRMATION_ENV):
        harness.run_disposable_smoke(
            output,
            expected_source_commit="1" * 40,
        )
    assert not output.exists()


def test_disposable_execution_rejects_unrelated_opening_token(monkeypatch):
    _clear_opening_tokens(monkeypatch)
    monkeypatch.setenv(
        harness.CONFIRMATION_ENV,
        harness.CONFIRMATION_VALUE,
    )
    monkeypatch.setenv("OPEN_GENERATIVE_IID_V2_511M_TRAINING", "wrong")
    with pytest.raises(RuntimeError, match="unrelated"):
        harness._require_disposable_token_only()


def test_seal_status_never_claims_seed_access(monkeypatch):
    _clear_opening_tokens(monkeypatch)
    harness._require_no_opening_tokens()
    status = harness.seal_status()
    assert status["disposable_seed_ranges_opened"] is False
    assert status["production_511m_opened"] is False
    assert status["fixed_validation_512m_opened"] is False
    assert status["reserved_513m_515m_opened"] is False
    assert status["confirmation_present"] is False


def test_clean_host_attestation_is_strict(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("container has no git")

    commit = "2" * 40
    monkeypatch.setattr(harness.subprocess, "run", unavailable)
    monkeypatch.setenv(harness.HOST_SOURCE_COMMIT_ENV, commit)
    monkeypatch.setenv(harness.HOST_SOURCE_BRANCH_ENV, "detached")
    monkeypatch.setenv(harness.HOST_WORKTREE_CLEAN_ENV, "1")
    assert harness._source_control_state() == (
        commit,
        "detached",
        False,
    )

    monkeypatch.setenv(harness.HOST_WORKTREE_CLEAN_ENV, "0")
    with pytest.raises(RuntimeError, match="attestation"):
        harness._source_control_state()


def test_dataset_fingerprint_is_deterministic_on_ordinary_fixture():
    first = simulate_generative_iid_dataset(
        n_sites=24,
        n_species=12,
        covariate_shape="normal",
        loading_stratum="medium",
        prevalence_stratum="moderate",
        seed=982001,
    )
    second = simulate_generative_iid_dataset(
        n_sites=24,
        n_species=12,
        covariate_shape="normal",
        loading_stratum="medium",
        prevalence_stratum="moderate",
        seed=982001,
    )
    other = simulate_generative_iid_dataset(
        n_sites=24,
        n_species=12,
        covariate_shape="normal",
        loading_stratum="medium",
        prevalence_stratum="moderate",
        seed=982002,
    )
    assert harness._dataset_sha256(first) == harness._dataset_sha256(second)
    assert harness._dataset_sha256(first) != harness._dataset_sha256(other)


def test_freeze_inventory_rejects_unknown_or_mutated_artifacts(tmp_path):
    output = tmp_path / "run"
    checkpoint = output / "checkpoint"
    checkpoint.mkdir(parents=True)
    files = {
        "corpus_manifest": output / "corpus_manifest.json",
        "report": output / "disposable_smoke_report.json",
        "checkpoint_manifest": (checkpoint / "generative_iid_orbit_checkpoint.json"),
        "checkpoint_weights": checkpoint / "weights.weights.h5",
    }
    for index, path in enumerate(files.values()):
        path.write_bytes(f"fixture-{index}".encode())
    freeze = {
        "kind": "generative_iid_v2_disposable_freeze",
        "protocol": harness.GENERATIVE_IID_V2_PROTOCOL,
        "artifacts": {
            name: harness._artifact_record(path, output=output)
            for name, path in files.items()
        },
    }
    harness._validate_freeze_inventory(output, freeze)

    (output / "unknown.txt").write_text("unknown")
    with pytest.raises(ValueError, match="unknown artifacts"):
        harness._validate_freeze_inventory(output, freeze)
    (output / "unknown.txt").unlink()

    files["report"].write_text(json.dumps({"mutated": True}))
    with pytest.raises(ValueError, match="report hash"):
        harness._validate_freeze_inventory(output, freeze)
