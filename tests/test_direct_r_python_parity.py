import numpy as np
import pandas as pd
import pytest

from examples.run_direct_r_python_parity import (
    _acceptance_gates,
    _boundary_checks,
    _load_existing_python_native,
    _metric_deltas,
    _random_level_compare,
)


def test_direct_boundary_checks_include_random_design_codes():
    native = {
        "Y": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "X": np.array([[1.0, -0.5], [1.0, 0.5]]),
        "T": np.ones((2, 1)),
        "Pi": np.array([[0], [1]]),
    }
    r_model = {
        "YScaled": native["Y"].tolist(),
        "XScaled": native["X"].tolist(),
        "TrScaled": native["T"].tolist(),
        "Pi": (native["Pi"] + 1).tolist(),
    }

    checks = _boundary_checks(native, r_model)

    assert all(check["passed"] for check in checks.values())
    assert checks["Pi"]["max_abs_diff"] == 0.0


def test_direct_metric_and_random_gates_pass_for_close_runs():
    beta_compare = {
        "shape_match": True,
        "mean_correlation": 0.99,
    }
    gamma_compare = {
        "shape_match": True,
        "mean_correlation": 0.98,
    }
    random_compare = {
        "n_levels": 1,
        "levels": [
            {
                "eta_shape_match": True,
                "lambda_shape_match": True,
                "association_compare": {
                    "shape_match": True,
                    "mean_correlation": 0.91,
                },
            }
        ],
    }
    metrics = pd.DataFrame(
        [
            {"model": "python_native", "prediction_mae": 1.1, "prediction_rmse": 1.4},
            {"model": "r_bridge", "prediction_mae": 1.0, "prediction_rmse": 1.3},
        ]
    )
    deltas = _metric_deltas(metrics, baseline="r_bridge", candidate="python_native")

    gates = _acceptance_gates(
        boundary={"X": {"passed": True}},
        beta_compare=beta_compare,
        gamma_compare=gamma_compare,
        random_compare=random_compare,
        deltas=deltas,
        min_beta_corr=0.95,
        min_gamma_corr=0.95,
        min_association_corr=0.75,
        max_prediction_mae_delta=0.25,
    )

    assert deltas["prediction_mae"] == 0.10000000000000009
    assert all(gate["passed"] for gate in gates.values())


class _FakeFit:
    def __init__(self, eta: np.ndarray | None, lam: np.ndarray | None, assoc: np.ndarray | None):
        self._eta = eta
        self._lam = lam
        self._assoc = assoc

    def eta_samples(self, level=0):
        if self._eta is None or level != 0:
            raise ValueError("missing")
        return self._eta

    def lambda_samples(self, level=0):
        if self._lam is None or level != 0:
            raise ValueError("missing")
        return self._lam

    def species_association_samples(self, level=0, correlation=False):
        if self._assoc is None or level != 0:
            raise ValueError("missing")
        return self._assoc


def test_random_level_compare_reports_shapes_and_associations():
    eta = np.zeros((2, 3, 4, 1))
    lam = np.zeros((2, 3, 1, 5))
    assoc = np.arange(150, dtype=float).reshape(2, 3, 5, 5)

    compare = _random_level_compare(_FakeFit(eta, lam, assoc), _FakeFit(eta, lam, assoc + 0.01))

    assert compare["n_levels"] == 1
    level = compare["levels"][0]
    assert level["eta_shape_match"]
    assert level["lambda_shape_match"]
    assert level["association_compare"]["shape_match"]
    assert level["association_compare"]["mean_correlation"] > 0.99


def test_reuse_existing_python_native_requires_prior_outputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="reuse requested"):
        _load_existing_python_native(model=object(), root=tmp_path)
