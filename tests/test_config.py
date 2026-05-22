from pyhmsc.config import model_from_config


def test_model_from_yaml_config():
    model, config = model_from_config("tests/fixtures/fixed_effect/model.yaml")
    assert config["distribution"] == "poisson"
    assert model.Y.shape == (6, 3)
    assert model.X.shape == (6, 2)
    assert model.x_formula == "~ forest_cover + elevation"
