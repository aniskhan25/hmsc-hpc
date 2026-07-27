from pathlib import Path

import numpy as np
import pandas as pd

from examples.evaluate_neural_hmsc_probability_ensemble import (
    evaluate_probability_ensembles,
    summarize_probability_ensembles,
)


def test_probability_ensemble_requires_full_and_leave_one_out_stability():
    observed = np.ones((2, 2), dtype=float)
    scale = {seed: np.full((2, 2), 0.5) for seed in (1, 2, 3)}
    affine = {
        1: np.full((2, 2), 0.7),
        2: np.full((2, 2), 0.7),
        3: np.full((2, 2), 0.3),
    }
    rows = evaluate_probability_ensembles(
        scale,
        affine,
        observed,
        dataset="big_spatial",
    )
    rows.extend(
        evaluate_probability_ensembles(
            scale,
            scale,
            observed,
            dataset="whittaker",
        )
    )

    summary = summarize_probability_ensembles(pd.DataFrame(rows))

    assert summary["decision"] == "probability_ensemble_promotion_candidate"
    assert summary["all_full_and_leave_one_out_no_degradation"]
    assert summary["full_big_spatial_genuine_proper_score_improvement"]
    assert not summary["target_response_used_for_selection"]


def test_probability_ensemble_reports_qualified_mcmc_without_using_it_as_gate():
    observed = np.ones((2, 2), dtype=float)
    scale = {seed: np.full((2, 2), 0.5) for seed in (1, 2, 3)}
    affine = {seed: np.full((2, 2), 0.6) for seed in (1, 2, 3)}
    mcmc = {seed: np.full((2, 2), 0.8) for seed in (1, 2, 3)}
    rows = evaluate_probability_ensembles(
        scale,
        affine,
        observed,
        dataset="big_spatial",
        mcmc_predictions=mcmc,
    )
    rows.extend(
        evaluate_probability_ensembles(
            scale,
            scale,
            observed,
            dataset="whittaker",
            mcmc_predictions=mcmc,
        )
    )

    summary = summarize_probability_ensembles(pd.DataFrame(rows))

    assert summary["decision"] == "probability_ensemble_promotion_candidate"
    assert summary["mcmc_comparison"]["reference_role"] == (
        "qualified_python_mcmc_diagnostic_comparator"
    )
    assert not summary["mcmc_comparison"]["neural_equivalence_claimed"]
    assert rows[0]["affine_vs_mcmc_brier_score_ratio"] > 1.0


def test_probability_ensemble_loads_outcomes_after_member_predictions():
    text = Path(
        "examples/evaluate_neural_hmsc_probability_ensemble.py"
    ).read_text(encoding="utf-8")

    affine_prediction = text.index("affine_subset.predict_mean")
    mcmc_prediction = text.index("mcmc_fits[positions[seed]].predict_mean")
    outcomes = text.index("observed_frame = pd.read_csv")

    assert affine_prediction < outcomes
    assert mcmc_prediction < outcomes
