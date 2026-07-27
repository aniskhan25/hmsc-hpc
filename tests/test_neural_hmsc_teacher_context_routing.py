from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.check_neural_hmsc_teacher_context_routing import (
    _context_design,
    _route_requirement_passed,
    _validate_covariate_only_path,
    _validate_manifest_contract,
)


def test_route_requirements_are_fail_closed():
    approved = ("effect_size_shift", "big_spatial_shape")
    identity = {
        "active": False,
        "selected_label": "rare_validation",
        "within_approved_distance_cap": False,
    }
    target = {
        "active": True,
        "selected_label": "big_spatial_shape",
        "within_approved_distance_cap": True,
    }
    outside_support = {
        "active": False,
        "selected_label": "in_distribution",
        "within_approved_distance_cap": False,
    }

    assert _route_requirement_passed(
        "identity", identity, approved_labels=approved, max_delta=0.0
    )
    assert _route_requirement_passed(
        "approved_context", target, approved_labels=approved, max_delta=0.01
    )
    assert not _route_requirement_passed(
        "approved_context",
        outside_support,
        approved_labels=approved,
        max_delta=0.0,
    )


def test_context_design_uses_frozen_covariate_order():
    X = pd.DataFrame({"unused": [9.0, 8.0], "TMG": [-1.0, 2.0]})
    design = _context_design(
        X,
        {"covariates": ["Intercept", "TMG"]},
    )

    np.testing.assert_allclose(design, [[1.0, -1.0], [1.0, 2.0]])


def test_routing_accepts_only_heldout_covariate_file(tmp_path):
    test_dir = tmp_path / "data" / "test"
    test_dir.mkdir(parents=True)
    X_path = test_dir / "X.csv"
    X_path.write_text("site,TMG\na,0\n", encoding="utf-8")

    _validate_covariate_only_path(X_path)
    with pytest.raises(ValueError, match="covariate X.csv"):
        _validate_covariate_only_path(test_dir / "response.csv")


def test_manifest_contract_rejects_outcome_selected_artifact():
    manifest = {
        "kind": "pyhmsc_predictive_probability_ensemble",
        "aggregation": "arithmetic_mean_response_probability",
        "ordered_members": True,
        "calibration_role": "affine_branch",
        "provenance": {
            "dataset": "whittaker",
            "response_semantics": "predictive_only",
            "selection_outcomes_used": True,
        },
    }

    with pytest.raises(ValueError, match="selection used outcomes"):
        _validate_manifest_contract(manifest, dataset="whittaker")


def test_routing_harness_has_no_response_or_scoring_input():
    source = Path(
        "examples/check_neural_hmsc_teacher_context_routing.py"
    ).read_text(encoding="utf-8")

    assert 'pd.read_csv(X_path' in source
    assert 'Y.csv' not in source
    assert 'brier_score' not in source
    assert 'log_loss' not in source
