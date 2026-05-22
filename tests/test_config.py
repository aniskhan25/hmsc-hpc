from pyhmsc.config import model_from_config


def test_model_from_yaml_config():
    model, config = model_from_config("tests/fixtures/fixed_effect/model.yaml")
    assert config["distribution"] == "poisson"
    assert model.Y.shape == (6, 3)
    assert model.X.shape == (6, 2)
    assert model.x_formula == "~ forest_cover + elevation"


def test_model_from_yaml_config_with_traits_and_phylogeny(tmp_path):
    model, config = model_from_config("tests/fixtures/fixed_effect/model_traits_phylo.yaml")
    assert config["phylo_cov"] == "phylo_cov.csv"
    assert model.traits.shape == (3, 2)
    assert model.phylo_cov.shape == (3, 3)
    compiled = model.compile(tmp_path / "traits-phylo", chains=1)
    assert compiled.init_json.exists()
