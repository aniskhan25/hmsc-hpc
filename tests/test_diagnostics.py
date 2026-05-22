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
