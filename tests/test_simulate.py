import numpy as np

from pyhmsc.simulate import simulate_fixed_effect_data


def test_simulate_fixed_effect_data_shapes_and_truth():
    beta = np.array([[0.1, -0.1], [0.7, -0.7]])
    Y, X, truth = simulate_fixed_effect_data(n_sites=12, beta=beta, distr="normal", seed=4)
    assert Y.shape == (12, 2)
    assert X.shape == (12, 1)
    assert truth.loc["x", "sp1"] > 0
    assert truth.loc["x", "sp2"] < 0
