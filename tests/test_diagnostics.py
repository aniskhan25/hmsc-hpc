import pandas as pd
import pytest

from pyhmsc import HmscModel
from pyhmsc.posterior import HmscFit


def test_diagnostics_require_or_use_arviz():
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
    )
    fit = HmscFit({"0": {"0": {"Beta": [[0.0], [1.0]]}}, "1": {"0": {"Beta": [[0.2], [0.8]]}}}, model)
    try:
        data = fit.to_arviz()
    except RuntimeError:
        pytest.skip("arviz not installed")
    assert "Beta" in data.posterior


def test_builtin_diagnostics_are_named_tables():
    model = HmscModel(
        Y=pd.DataFrame({"sp1": [1, 2]}),
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        x_formula="~ x",
    )
    posterior = {
        "0": {
            "0": {"Beta": [[0.0], [1.0]]},
            "1": {"Beta": [[0.1], [1.1]]},
            "2": {"Beta": [[0.2], [1.2]]},
        },
        "1": {
            "0": {"Beta": [[0.0], [1.0]]},
            "1": {"Beta": [[0.1], [1.1]]},
            "2": {"Beta": [[0.2], [1.2]]},
        },
    }
    fit = HmscFit(posterior, model)

    diagnostics = fit.diagnostics("Beta")
    overview = fit.diagnostics_overview("Beta", ess_threshold=2)

    assert list(diagnostics.columns) == ["covariate", "species", "mean", "sd", "rhat", "ess"]
    assert diagnostics["covariate"].tolist() == ["Intercept", "x"]
    assert diagnostics["species"].tolist() == ["sp1", "sp1"]
    assert diagnostics["rhat"].notna().all()
    assert diagnostics["ess"].gt(0).all()
    assert overview["n_parameters"] == 2
    assert overview["n_rhat_flagged"] == 0
