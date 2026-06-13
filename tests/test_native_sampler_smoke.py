import numpy as np
import pandas as pd
import pytest

from pyhmsc import HmscModel


@pytest.mark.parametrize(
    ("distr", "Y"),
    [
        (
            "normal",
            pd.DataFrame({"sp1": [1.2, 1.5, -0.2], "sp2": [0.2, 0.3, 1.1]}),
        ),
        (
            "poisson",
            pd.DataFrame({"sp1": [1, 2, 0], "sp2": [0, 1, 3]}),
        ),
        (
            "probit",
            pd.DataFrame({"sp1": [1, 1, 0], "sp2": [0, 1, 1]}),
        ),
    ],
)
def test_python_native_sampler_smoke(tmp_path, distr, Y):
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    model = HmscModel(Y=Y, X=X, x_formula="~ x", distr=distr)

    fit = model.sample(
        samples=1,
        transient=1,
        thin=1,
        chains=1,
        init="python-native",
        verbose=1,
        workdir=tmp_path / distr,
    )

    beta = fit.beta_mean()
    assert beta.shape == (2, 2)
    assert list(beta.index) == ["Intercept", "x"]
    assert np.isfinite(beta.to_numpy()).all()
    assert fit.output_file.exists()
    assert fit.output_file.suffix == ".h5"


def test_python_native_iid_random_intercept_sampler_smoke(tmp_path):
    Y = pd.DataFrame({"sp1": [1, 2, 0], "sp2": [0, 1, 3]})
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    study_design = pd.DataFrame({"plot": ["a", "b", "a"]})
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        study_design=study_design,
        random_levels={"plot": {"column": "plot", "type": "iid"}},
    )

    fit = model.sample(
        samples=1,
        transient=1,
        thin=1,
        chains=1,
        init="python-native",
        verbose=1,
        workdir=tmp_path / "iid",
    )

    assert fit.beta_mean().shape == (2, 2)
    assert np.isfinite(fit.beta_mean().to_numpy()).all()


def test_python_native_iid_random_slope_sampler_smoke(tmp_path):
    Y = pd.DataFrame({"sp1": [1, 2, 0, 2], "sp2": [0, 1, 3, 1]})
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    study_design = pd.DataFrame(
        {
            "plot": ["a", "b", "a", "b"],
            "elevation": [10.0, 20.0, 10.0, 20.0],
        }
    )
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        study_design=study_design,
        random_levels={"plot": {"column": "plot", "type": "iid", "x_formula": "~ elevation"}},
    )

    fit = model.sample(
        samples=1,
        transient=1,
        thin=1,
        chains=1,
        init="python-native",
        verbose=1,
        workdir=tmp_path / "iid-slope",
    )

    assert fit.beta_mean().shape == (2, 2)
    assert fit.lambda_samples(level=0).shape[-2:] == (2, 2)
    assert np.isfinite(fit.lambda_samples(level=0)).all()


def test_python_native_traits_sampler_smoke(tmp_path):
    Y = pd.DataFrame({"sp1": [1, 2, 0], "sp2": [0, 1, 3]})
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    traits = pd.DataFrame({"body": [1.0, 2.0]}, index=["sp1", "sp2"])
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        traits=traits,
        trait_formula="~ body",
    )
    fit = model.sample(
        samples=1,
        transient=0,
        thin=1,
        chains=1,
        init="python-native",
        verbose=1,
        workdir=tmp_path / "traits",
    )
    assert fit.beta_mean().shape == (2, 2)


def test_python_native_phylogeny_sampler_smoke(tmp_path):
    Y = pd.DataFrame({"sp1": [1, 2, 0], "sp2": [0, 1, 3]})
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    phylo = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["sp1", "sp2"], columns=["sp1", "sp2"])
    model = HmscModel(Y=Y, X=X, x_formula="~ x", distr="poisson", phylo_cov=phylo)
    fit = model.sample(
        samples=1,
        transient=0,
        thin=1,
        chains=1,
        init="python-native",
        verbose=1,
        workdir=tmp_path / "phylo",
    )
    assert fit.beta_mean().shape == (2, 2)


def test_python_native_spatial_full_sampler_smoke(tmp_path):
    Y = pd.DataFrame({"sp1": [1, 2, 0], "sp2": [0, 1, 3]})
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    study_design = pd.DataFrame(
        {"plot": ["a", "b", "c"], "xcoord": [0.0, 1.0, 0.0], "ycoord": [0.0, 0.0, 1.0]}
    )
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        study_design=study_design,
        random_levels={"plot": {"column": "plot", "type": "spatial_full", "coords": ["xcoord", "ycoord"]}},
    )
    fit = model.sample(
        samples=1,
        transient=0,
        thin=1,
        chains=1,
        init="python-native",
        verbose=1,
        workdir=tmp_path / "spatial",
    )
    assert fit.beta_mean().shape == (2, 2)


def test_python_native_spatial_full_random_slope_sampler_smoke(tmp_path):
    Y = pd.DataFrame({"sp1": [1, 2, 0, 1], "sp2": [0, 1, 3, 1]})
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    study_design = pd.DataFrame(
        {
            "plot": ["a", "b", "c", "d"],
            "elevation": [10.0, 20.0, 10.0, 30.0],
            "xcoord": [0.0, 1.0, 0.0, 1.0],
            "ycoord": [0.0, 0.0, 1.0, 1.0],
        }
    )
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        study_design=study_design,
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_full",
                "coords": ["xcoord", "ycoord"],
                "x_formula": "~ elevation",
            }
        },
    )
    fit = model.sample(
        samples=1,
        transient=0,
        thin=1,
        chains=1,
        init="python-native",
        verbose=1,
        workdir=tmp_path / "spatial-full-random-slope",
    )
    assert fit.beta_mean().shape == (2, 2)
    assert fit.eta_samples(level=0).shape[-2:] == (4, 1)
    assert fit.lambda_samples(level=0).shape[-3:] == (1, 2, 2)


def test_python_native_spatial_gpp_random_slope_sampler_smoke(tmp_path):
    Y = pd.DataFrame({"sp1": [1, 2, 0, 1], "sp2": [0, 1, 3, 1]})
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    study_design = pd.DataFrame(
        {
            "plot": ["a", "b", "c", "d"],
            "elevation": [10.0, 20.0, 10.0, 30.0],
            "xcoord": [0.0, 1.0, 0.0, 1.0],
            "ycoord": [0.0, 0.0, 1.0, 1.0],
        }
    )
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        study_design=study_design,
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_gpp",
                "coords": ["xcoord", "ycoord"],
                "n_knots": 2,
                "x_formula": "~ elevation",
            }
        },
    )
    fit = model.sample(
        samples=1,
        transient=1,
        thin=1,
        chains=1,
        init="python-native",
        verbose=1,
        workdir=tmp_path / "spatial-gpp-random-slope",
    )
    assert fit.beta_mean().shape == (2, 2)
    assert fit.eta_samples(level=0).shape[-2:] == (4, 1)
    assert fit.lambda_samples(level=0).shape[-3:] == (1, 2, 2)


def test_python_native_spatial_nngp_random_slope_sampler_smoke(tmp_path):
    Y = pd.DataFrame({"sp1": [1, 2, 0, 1], "sp2": [0, 1, 3, 1]})
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    study_design = pd.DataFrame(
        {
            "plot": ["a", "b", "c", "d"],
            "elevation": [10.0, 20.0, 10.0, 30.0],
            "xcoord": [0.0, 1.0, 0.0, 1.0],
            "ycoord": [0.0, 0.0, 1.0, 1.0],
        }
    )
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        study_design=study_design,
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_nngp",
                "coords": ["xcoord", "ycoord"],
                "n_neighbors": 2,
                "x_formula": "~ elevation",
            }
        },
    )
    fit = model.sample(
        samples=1,
        transient=1,
        thin=1,
        chains=1,
        init="python-native",
        verbose=1,
        workdir=tmp_path / "spatial-nngp-random-slope",
    )
    assert fit.beta_mean().shape == (2, 2)
    assert fit.eta_samples(level=0).shape[-2:] == (4, 1)
    assert fit.lambda_samples(level=0).shape[-3:] == (1, 2, 2)


def test_python_native_spatial_gpp_sampler_smoke(tmp_path):
    Y = pd.DataFrame({"sp1": [1, 2, 0, 1], "sp2": [0, 1, 3, 1]})
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    study_design = pd.DataFrame(
        {
            "plot": ["a", "b", "c", "d"],
            "xcoord": [0.0, 1.0, 0.0, 1.0],
            "ycoord": [0.0, 0.0, 1.0, 1.0],
        }
    )
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        study_design=study_design,
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_gpp",
                "coords": ["xcoord", "ycoord"],
                "n_knots": 2,
            }
        },
    )
    fit = model.sample(
        samples=1,
        transient=0,
        thin=1,
        chains=1,
        init="python-native",
        verbose=1,
        workdir=tmp_path / "spatial-gpp",
    )
    assert fit.beta_mean().shape == (2, 2)
    assert fit.eta_samples(level=0).shape[-2:] == (4, 1)


def test_python_native_spatial_nngp_sampler_smoke(tmp_path):
    Y = pd.DataFrame({"sp1": [1, 2, 0, 1], "sp2": [0, 1, 3, 1]})
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    study_design = pd.DataFrame(
        {
            "plot": ["a", "b", "c", "d"],
            "xcoord": [0.0, 1.0, 0.0, 1.0],
            "ycoord": [0.0, 0.0, 1.0, 1.0],
        }
    )
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        study_design=study_design,
        random_levels={
            "plot": {
                "column": "plot",
                "type": "spatial_nngp",
                "coords": ["xcoord", "ycoord"],
                "n_neighbors": 2,
            }
        },
    )
    fit = model.sample(
        samples=1,
        transient=0,
        thin=1,
        chains=1,
        init="python-native",
        verbose=1,
        workdir=tmp_path / "spatial-nngp",
    )
    assert fit.beta_mean().shape == (2, 2)
    assert fit.eta_samples(level=0).shape[-2:] == (4, 1)
