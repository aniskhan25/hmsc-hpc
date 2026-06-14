import numpy as np

from pyhmsc.simulate import (
    simulate_fixed_effect_data,
    simulate_spatial_effect_data,
    simulate_spatial_random_slope_effect_data,
)


def test_simulate_fixed_effect_data_shapes_and_truth():
    beta = np.array([[0.1, -0.1], [0.7, -0.7]])
    Y, X, truth = simulate_fixed_effect_data(n_sites=12, beta=beta, distr="normal", seed=4)
    assert Y.shape == (12, 2)
    assert X.shape == (12, 1)
    assert truth.loc["x", "sp1"] > 0
    assert truth.loc["x", "sp2"] < 0


def test_simulate_spatial_effect_data_is_deterministic_and_named():
    left = simulate_spatial_effect_data(n_sites=16, n_species=4, seed=42)
    right = simulate_spatial_effect_data(n_sites=16, n_species=4, seed=42)

    for left_frame, right_frame in zip(left[:3], right[:3]):
        assert left_frame.equals(right_frame)
    for key in left[3]:
        assert left[3][key].equals(right[3][key])

    Y, X, study_design, truth = left
    assert Y.shape == (16, 4)
    assert X.shape == (16, 1)
    assert list(X.columns) == ["env"]
    assert list(study_design.columns) == ["plot", "xcoord", "ycoord"]
    assert truth["beta"].shape == (2, 4)
    assert truth["site_effect"].shape == (16, 1)
    assert truth["lambda"].shape == (1, 4)
    assert truth["linear_predictor"].shape == (16, 4)
    assert truth["beta"].loc["env", "sp1"] > 0
    assert truth["beta"].loc["env", "sp4"] < 0
    assert truth["lambda"].loc["factor_0", "sp1"] > 0
    assert truth["lambda"].loc["factor_0", "sp4"] < 0


def test_simulate_spatial_effect_data_has_spatially_structured_truth():
    _Y, _X, study_design, truth = simulate_spatial_effect_data(n_sites=36, n_species=3, seed=7)
    coords = study_design[["xcoord", "ycoord"]].to_numpy()
    eta = truth["site_effect"]["eta"].to_numpy()
    distances = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    random_partner = (np.arange(len(eta)) + len(eta) // 2) % len(eta)
    nearest_difference = np.mean(np.abs(eta - eta[nearest]))
    far_difference = np.mean(np.abs(eta - eta[random_partner]))

    assert nearest_difference < far_difference


def test_simulate_spatial_random_slope_effect_data_is_deterministic_and_named():
    left = simulate_spatial_random_slope_effect_data(n_sites=25, n_species=4, seed=43)
    right = simulate_spatial_random_slope_effect_data(n_sites=25, n_species=4, seed=43)

    for left_frame, right_frame in zip(left[:3], right[:3]):
        assert left_frame.equals(right_frame)
    for key in left[3]:
        assert left[3][key].equals(right[3][key])

    Y, X, study_design, truth = left
    assert Y.shape == (25, 4)
    assert X.shape == (25, 1)
    assert list(study_design.columns) == ["plot", "slope_env", "xcoord", "ycoord"]
    assert truth["beta"].shape == (2, 4)
    assert truth["site_effect"].shape == (25, 1)
    assert list(truth["lambda"].index) == ["Intercept", "slope_env"]
    assert truth["lambda"].shape == (2, 4)
    assert truth["lambda"].loc["Intercept", "sp1"] > 0
    assert truth["lambda"].loc["Intercept", "sp4"] < 0
    assert truth["lambda"].loc["slope_env", "sp1"] < 0
    assert truth["lambda"].loc["slope_env", "sp4"] > 0
