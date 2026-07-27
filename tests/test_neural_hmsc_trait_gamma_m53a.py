from argparse import Namespace

import pytest

from examples.run_neural_hmsc_trait_gamma_m53a import (
    EVALUATION_BLOCK_STARTS,
    OPEN_RESERVED_CONFIRMATION,
    RESERVED_SIMULATION_SEEDS,
    SMOKE_CALIBRATION_START,
    SMOKE_EVALUATION_START,
    _assert_disposable,
    factorial_corpus,
    run_reserved_evaluation,
)


def test_disposable_factorial_is_balanced_and_avoids_reserved_seeds():
    _, calibration_rows = factorial_corpus((SMOKE_CALIBRATION_START,), 36)
    _, evaluation_rows = factorial_corpus((SMOKE_EVALUATION_START,), 45)
    calibration_counts = {}
    for row in calibration_rows:
        calibration_counts[row["cell"]] = calibration_counts.get(row["cell"], 0) + 1
    evaluation_counts = {}
    for row in evaluation_rows:
        evaluation_counts[row["cell"]] = evaluation_counts.get(row["cell"], 0) + 1
    assert set(calibration_counts.values()) == {4}
    assert set(evaluation_counts.values()) == {5}
    assert not ({row["seed"] for row in calibration_rows} & RESERVED_SIMULATION_SEEDS)
    assert not ({row["seed"] for row in evaluation_rows} & RESERVED_SIMULATION_SEEDS)
    _assert_disposable(calibration_rows + evaluation_rows)


def test_disposable_guard_rejects_reserved_evaluation_seed():
    with pytest.raises(ValueError, match="reserved seeds"):
        _assert_disposable(
            [
                {
                    "seed": EVALUATION_BLOCK_STARTS[0],
                    "cell": "test",
                }
            ]
        )


def test_reserved_evaluation_requires_exact_confirmation_before_file_access(tmp_path):
    args = Namespace(
        confirmation="wrong",
        calibration_root=tmp_path / "missing",
        output=tmp_path / "output",
    )
    with pytest.raises(ValueError, match="remains sealed"):
        run_reserved_evaluation(args)
    assert OPEN_RESERVED_CONFIRMATION == "OPEN_M53A_RESERVED_EVALUATION"
