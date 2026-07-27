import argparse
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "examples/qualify_neural_hmsc_variable_design.py"
SPEC = importlib.util.spec_from_file_location("m54_qualification", SCRIPT)
M54 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M54)


def test_m54_production_roles_are_exact_and_disjoint():
    blocks = []
    for role, spec in M54.ROLE_SPECS.items():
        assert role in {"candidate", "sensitivity_a", "sensitivity_b"}
        for phase in ("train", "calibration", "evaluation"):
            block = M54._seed_block(spec[phase], M54.PRODUCTION_COUNT)
            assert len(block) == 243
            assert all(M54._is_production_seed(seed) for seed in block)
            blocks.append(block)
    M54._assert_disjoint(*blocks)


def test_m54_production_factorial_and_smoke_marginals_are_balanced():
    production = M54._corpus_schedule("production")
    smoke = M54.build_corpus(
        M54._seed_block(M54.SMOKE_STARTS["train"], 27), profile="smoke"
    )

    base_cells = [tuple(value[0] for value in row[:4]) for row in production]
    assert len(production) == 243
    assert len(set(base_cells)) == 81
    assert {base_cells.count(cell) for cell in set(base_cells)} == {3}
    for position in range(6):
        values = [
            row[position][0] if position < 4 else row[position] for row in production
        ]
        assert max(values.count(value) for value in set(values)) == min(
            values.count(value) for value in set(values)
        )
    assert M54._marginals_balanced(smoke)
    assert all(
        not M54._is_production_seed(int(dataset.metadata["seed"])) for dataset in smoke
    )


def test_m54_stratified_simulation_is_deterministic_and_hits_condition_target():
    kwargs = {
        "seed": 91_123_456,
        "n_sites": 40,
        "n_species": 20,
        "n_covariates": 5,
        "target_condition": 10.0,
        "prevalence": "balanced",
        "effect": "moderate",
        "strata": {
            "site": "site_1",
            "species": "species_1",
            "covariate": "covariate_1",
            "design_condition": "condition_1",
        },
    }
    first = M54.simulate_stratified_dataset(**kwargs)
    second = M54.simulate_stratified_dataset(**kwargs)

    assert first.X.equals(second.X)
    assert first.Y.equals(second.Y)
    assert first.truth_beta.equals(second.truth_beta)
    assert first.metadata["actual_condition"] == pytest.approx(10.0, rel=1e-6)


def test_m54_train_confirmation_fails_before_output_or_corpus(tmp_path):
    output = tmp_path / "must_not_exist"
    args = argparse.Namespace(
        role="candidate",
        confirmation="wrong",
        output=output,
        fixed_registry=tmp_path / "missing_fixed",
        variable_baseline=tmp_path / "missing_variable",
    )

    with pytest.raises(ValueError, match="train/calibration confirmation"):
        M54.train_and_freeze(args)
    assert not output.exists()


def test_m54_evaluation_confirmation_fails_before_freeze_or_output(tmp_path):
    output = tmp_path / "must_not_exist"
    args = argparse.Namespace(
        role="candidate",
        confirmation="wrong",
        freeze_root=tmp_path / "missing_freeze",
        output=output,
        fixed_registry=tmp_path / "missing_fixed",
        variable_baseline=tmp_path / "missing_variable",
    )

    with pytest.raises(ValueError, match="reserved-evaluation confirmation"):
        M54.evaluate_frozen_role(args)
    assert not output.exists()
