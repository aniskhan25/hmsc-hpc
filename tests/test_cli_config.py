import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_cli_compile_yaml_config(tmp_path):
    out = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "compile",
            "tests/fixtures/fixed_effect/model.yaml",
            "--output",
            str(out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    init_json = Path(result.stdout.strip())
    assert init_json.exists()
    assert init_json == out / "init.json"

    validate = subprocess.run(
        [sys.executable, "-m", "pyhmsc", "validate-init", str(init_json), "--strict"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "native_sampler_supported: passed" in validate.stdout


def test_cli_compile_yaml_config_chain_override(tmp_path):
    default_out = tmp_path / "default"
    override_out = tmp_path / "override"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "compile",
            "tests/fixtures/fixed_effect/model.yaml",
            "--output",
            str(default_out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "compile",
            "tests/fixtures/fixed_effect/model.yaml",
            "--chains",
            "4",
            "--output",
            str(override_out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    default_meta = json.loads((default_out / "init.json").read_text(encoding="utf-8"))
    override_meta = json.loads((override_out / "init.json").read_text(encoding="utf-8"))
    assert default_meta["dimensions"]["n_chains"] == 2
    assert override_meta["dimensions"]["n_chains"] == 4


def test_cli_validate_init_rejects_model_yaml():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "validate-init",
            "tests/fixtures/fixed_effect/model.yaml",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "compile MODEL.yaml" in result.stderr


def test_cli_predict_probit_uses_response_scale(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    x_path = tmp_path / "X.csv"
    output = tmp_path / "pred.csv"
    pd.DataFrame({"x": [1.0]}, index=["site_1"]).to_csv(x_path)
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset("Beta", data=[[[[0.0], [1.0]], [[0.0], [-1.0]]]])
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"covariates":["Intercept","x"],"species":["sp1"]},'
            '"formula":{"X":"~ x"},"distribution":"probit"}'
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "predict",
            str(posterior),
            "--X",
            str(x_path),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    pred = pd.read_csv(output, index_col=0)
    assert 0 <= pred.loc["site_1", "sp1"] <= 1
    assert abs(pred.loc["site_1", "sp1"] - 0.5) < 1e-8


def test_cli_predict_supports_separate_study_design_for_random_effects(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    x_train = tmp_path / "X_train.csv"
    y_train = tmp_path / "Y_train.csv"
    study_train = tmp_path / "study_train.csv"
    x_new = tmp_path / "X_new.csv"
    study_new = tmp_path / "study_new.csv"
    config = tmp_path / "model.json"
    output = tmp_path / "pred.csv"

    pd.DataFrame({"x": [0.0, 1.0]}, index=["site_a", "site_b"]).to_csv(x_train)
    pd.DataFrame({"sp1": [1.0, 2.0]}, index=["site_a", "site_b"]).to_csv(y_train)
    pd.DataFrame({"plot": ["a", "b"]}, index=["site_a", "site_b"]).to_csv(study_train)
    pd.DataFrame({"x": [0.0, 1.0]}, index=["new_a", "new_b"]).to_csv(x_new)
    pd.DataFrame({"plot": ["a", "b"]}, index=["new_a", "new_b"]).to_csv(study_new)
    config.write_text(
        json.dumps(
            {
                "response": y_train.name,
                "covariates": x_train.name,
                "study_design": study_train.name,
                "formula": {"X": "~ x"},
                "distribution": "normal",
                "random_levels": {"plot": {"column": "plot", "type": "iid"}},
            }
        ),
        encoding="utf-8",
    )
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset("Beta", data=[[[[0.0], [0.0]]]])
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset("Eta", data=[[[[0.2], [0.5]]]])
        level.create_dataset("Lambda", data=[[[[1.0]]]])

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "predict",
            str(posterior),
            "--X",
            str(x_new),
            "--model-config",
            str(config),
            "--study-design",
            str(study_new),
            "--include-random-effects",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    pred = pd.read_csv(output, index_col=0)
    assert list(pred.index) == ["new_a", "new_b"]
    assert abs(pred.loc["new_a", "sp1"] - 0.2) < 1e-8
    assert abs(pred.loc["new_b", "sp1"] - 0.5) < 1e-8


def test_cli_predict_supports_spatial_nearest_with_coords(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    x_train = tmp_path / "X_train.csv"
    y_train = tmp_path / "Y_train.csv"
    study_train = tmp_path / "study_train.csv"
    x_new = tmp_path / "X_new.csv"
    study_new = tmp_path / "study_new.csv"
    coords_new = tmp_path / "coords_new.csv"
    config = tmp_path / "model.json"
    output = tmp_path / "pred.csv"

    pd.DataFrame({"x": [0.0, 1.0]}, index=["site_a", "site_b"]).to_csv(x_train)
    pd.DataFrame({"sp1": [1.0, 2.0]}, index=["site_a", "site_b"]).to_csv(y_train)
    pd.DataFrame(
        {"plot": ["a", "b"], "xcoord": [0.0, 10.0], "ycoord": [0.0, 0.0]},
        index=["site_a", "site_b"],
    ).to_csv(study_train)
    pd.DataFrame({"x": [0.0]}, index=["new_site"]).to_csv(x_new)
    pd.DataFrame({"plot": ["new"]}, index=["new_site"]).to_csv(study_new)
    pd.DataFrame({"xcoord": [9.0], "ycoord": [0.0]}, index=["new_site"]).to_csv(coords_new)
    config.write_text(
        json.dumps(
            {
                "response": y_train.name,
                "covariates": x_train.name,
                "study_design": study_train.name,
                "formula": {"X": "~ x"},
                "distribution": "normal",
                "random_levels": {
                    "plot": {
                        "column": "plot",
                        "type": "spatial_nngp",
                        "coords": ["xcoord", "ycoord"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset("Beta", data=[[[[0.0], [0.0]]]])
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset("Eta", data=[[[[0.2], [0.8]]]])
        level.create_dataset("Lambda", data=[[[[1.0]]]])

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "predict",
            str(posterior),
            "--X",
            str(x_new),
            "--model-config",
            str(config),
            "--study-design",
            str(study_new),
            "--coords",
            str(coords_new),
            "--random-effects",
            "known",
            "--unseen-groups",
            "nearest",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    pred = pd.read_csv(output, index_col=0)
    assert abs(pred.loc["new_site", "sp1"] - 0.8) < 1e-8


def test_cli_predict_supports_full_spatial_conditional_prediction(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    x_train = tmp_path / "X_train.csv"
    y_train = tmp_path / "Y_train.csv"
    study_train = tmp_path / "study_train.csv"
    x_new = tmp_path / "X_new.csv"
    study_new = tmp_path / "study_new.csv"
    coords_new = tmp_path / "coords_new.csv"
    config = tmp_path / "model.json"
    output = tmp_path / "pred.csv"
    repeated_output = tmp_path / "pred_repeated.csv"

    pd.DataFrame({"x": [0.0, 1.0]}, index=["site_a", "site_b"]).to_csv(x_train)
    pd.DataFrame({"sp1": [0.0, 1.0]}, index=["site_a", "site_b"]).to_csv(y_train)
    pd.DataFrame(
        {"plot": ["a", "b"], "xcoord": [0.0, 1.0], "ycoord": [0.0, 0.0]},
        index=["site_a", "site_b"],
    ).to_csv(study_train)
    pd.DataFrame({"x": [0.0]}, index=["new_site"]).to_csv(x_new)
    pd.DataFrame({"plot": ["new"]}, index=["new_site"]).to_csv(study_new)
    pd.DataFrame({"xcoord": [0.5], "ycoord": [0.0]}, index=["new_site"]).to_csv(coords_new)
    config.write_text(
        json.dumps(
            {
                "response": y_train.name,
                "covariates": x_train.name,
                "study_design": study_train.name,
                "formula": {"X": "~ x"},
                "distribution": "normal",
                "random_levels": {
                    "plot": {
                        "column": "plot",
                        "type": "spatial_full",
                        "coords": ["xcoord", "ycoord"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    draws = 100
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset("Beta", data=np.zeros((1, draws, 2, 1)))
        level = handle.create_group("random_levels").create_group("0")
        eta = np.zeros((1, draws, 2, 1))
        eta[:, :, 1, 0] = 1.0
        level.create_dataset("Eta", data=eta)
        level.create_dataset("Lambda", data=np.ones((1, draws, 1, 1)))
        level.create_dataset("Alpha", data=np.ones((1, draws, 1), dtype=int))
        handle.attrs["pyhmsc_metadata"] = json.dumps(
            {
                "random_levels": [
                    {
                        "name": "plot",
                        "column": "plot",
                        "type": "spatial_full",
                        "coords": ["xcoord", "ycoord"],
                        "alphapw": [[1.0, 1.0]],
                    }
                ]
            }
        )

    base_command = [
        sys.executable,
        "-m",
        "pyhmsc",
        "predict",
        str(posterior),
        "--X",
        str(x_new),
        "--model-config",
        str(config),
        "--study-design",
        str(study_new),
        "--coords",
        str(coords_new),
        "--random-effects",
        "known",
        "--spatial-prediction",
        "conditional",
        "--seed",
        "23",
    ]
    subprocess.run(base_command + ["--output", str(output)], check=True, capture_output=True, text=True)
    subprocess.run(
        base_command + ["--output", str(repeated_output)],
        check=True,
        capture_output=True,
        text=True,
    )

    prediction = pd.read_csv(output, index_col=0)
    repeated = pd.read_csv(repeated_output, index_col=0)
    assert np.isfinite(prediction.to_numpy()).all()
    pd.testing.assert_frame_equal(prediction, repeated)


def test_cli_summarize_gamma_uses_metadata_names(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset("Gamma", data=[[[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]]])
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"covariates":["Intercept","TMG"],"traits":["Intercept","CN"]},'
            '"formula":{"X":"~ TMG"},"distribution":"probit"}'
        )

    result = subprocess.run(
        [sys.executable, "-m", "pyhmsc", "summarize", str(posterior), "--param", "Gamma", "--level", "0.5"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "covariate" in result.stdout
    assert "trait" in result.stdout
    assert "TMG" in result.stdout
    assert "CN" in result.stdout


def test_cli_diagnostics_writes_named_report(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    output = tmp_path / "diagnostics.txt"
    with h5py.File(posterior, "w") as handle:
        handle.create_dataset(
            "Beta",
            data=[
                [[[0.0], [1.0]], [[0.1], [1.1]], [[0.2], [1.2]]],
                [[[0.0], [1.0]], [[0.1], [1.1]], [[0.2], [1.2]]],
            ],
        )
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"covariates":["Intercept","x"],"species":["sp1"]},'
            '"formula":{"X":"~ x"},"distribution":"normal"}'
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "diagnostics",
            str(posterior),
            "--param",
            "Beta",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    text = output.read_text(encoding="utf-8")
    assert "diagnostics" in text
    assert "rhat_max" in text
    assert "ess_min" in text
    assert "Intercept" in text
    assert "sp1" in text


def test_cli_diagnostics_supports_random_level_parameters(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    eta_output = tmp_path / "eta_diagnostics.txt"
    lambda_output = tmp_path / "lambda_diagnostics.txt"
    eta = [
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.1, 1.1], [2.1, 3.1]],
            [[0.2, 1.2], [2.2, 3.2]],
        ],
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.1, 1.1], [2.1, 3.1]],
            [[0.2, 1.2], [2.2, 3.2]],
        ],
    ]
    lam = [
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.1, 1.1], [2.1, 3.1]],
            [[0.2, 1.2], [2.2, 3.2]],
        ],
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.1, 1.1], [2.1, 3.1]],
            [[0.2, 1.2], [2.2, 3.2]],
        ],
    ]
    with h5py.File(posterior, "w") as handle:
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset("Eta", data=eta)
        level.create_dataset("Lambda", data=lam)
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"species":["sp1","sp2"]},'
            '"random_levels":[{"levels":["plot_a","plot_b"]}]}'
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "diagnostics",
            str(posterior),
            "--param",
            "Eta",
            "--random-level",
            "0",
            "--output",
            str(eta_output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "diagnostics",
            str(posterior),
            "--param",
            "Lambda",
            "--random-level",
            "0",
            "--output",
            str(lambda_output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    eta_text = eta_output.read_text(encoding="utf-8")
    lambda_text = lambda_output.read_text(encoding="utf-8")
    assert "random_level: 0" in eta_text
    assert "plot_a" in eta_text
    assert "factor_1" in eta_text
    assert "random_level: 0" in lambda_text
    assert "sp2" in lambda_text


def test_cli_associations_writes_pair_table(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    output = tmp_path / "associations.csv"
    with h5py.File(posterior, "w") as handle:
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset(
            "Lambda",
            data=[
                [
                    [[1.0, 2.0, -1.0]],
                    [[2.0, 1.0, -2.0]],
                ]
            ],
        )
        handle.attrs["pyhmsc_metadata"] = '{"names":{"species":["sp1","sp2","sp3"]}}'

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "associations",
            str(posterior),
            "--level",
            "0.5",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    table = pd.read_csv(output)
    assert list(table.columns) == [
        "species_1",
        "species_2",
        "mean",
        "lower",
        "upper",
        "p_positive",
        "p_negative",
    ]
    assert table.shape[0] == 3
    assert table.loc[(table["species_1"] == "sp1") & (table["species_2"] == "sp3"), "p_negative"].iloc[0] == 1.0


def test_cli_diagnostics_supports_species_associations(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    output = tmp_path / "association_diagnostics.txt"
    with h5py.File(posterior, "w") as handle:
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset(
            "Lambda",
            data=[
                [
                    [[1.0, 2.0, -1.0]],
                    [[1.0, 2.0, -1.0]],
                ],
                [
                    [[-1.0, -2.0, 1.0]],
                    [[-1.0, -2.0, 1.0]],
                ],
            ],
        )
        handle.attrs["pyhmsc_metadata"] = '{"names":{"species":["sp1","sp2","sp3"]}}'

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "diagnostics",
            str(posterior),
            "--param",
            "Associations",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    text = output.read_text(encoding="utf-8")
    assert "param: Associations" in text
    assert "association: correlation" in text
    assert "sp1" in text
    assert "sp3" in text
    assert "n_rhat_flagged: 0" in text


def test_cli_diagnostics_aligns_latent_factors(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    output = tmp_path / "lambda_aligned_diagnostics.txt"
    with h5py.File(posterior, "w") as handle:
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset(
            "Eta",
            data=[
                [[[0.5]], [[0.5]]],
                [[[-0.5]], [[-0.5]]],
            ],
        )
        level.create_dataset(
            "Lambda",
            data=[
                [
                    [[1.0, 2.0]],
                    [[1.0, 2.0]],
                ],
                [
                    [[-1.0, -2.0]],
                    [[-1.0, -2.0]],
                ],
            ],
        )
        handle.attrs["pyhmsc_metadata"] = '{"names":{"species":["sp1","sp2"]}}'

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "diagnostics",
            str(posterior),
            "--param",
            "Lambda",
            "--align-factors",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    text = output.read_text(encoding="utf-8")
    assert "aligned: True" in text
    assert "n_rhat_flagged: 0" in text
    assert "sp1" in text
    assert "sp2" in text


def test_cli_summarize_random_level_parameters(tmp_path):
    import h5py

    posterior = tmp_path / "posterior.h5"
    with h5py.File(posterior, "w") as handle:
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset("Eta", data=[[[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]]])
        level.create_dataset("Lambda", data=[[[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]]])
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"species":["sp1","sp2"]},'
            '"random_levels":[{"levels":["plot_a","plot_b"]}]}'
        )

    eta = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "summarize",
            str(posterior),
            "--param",
            "Eta",
            "--level",
            "0.5",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    lam = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "summarize",
            str(posterior),
            "--param",
            "Lambda",
            "--level",
            "0.5",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "plot_a" in eta.stdout
    assert "factor_1" in eta.stdout
    assert "sp2" in lam.stdout
    assert "factor_1" in lam.stdout


def test_cli_sample_and_summarize(tmp_path):
    run_dir = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "compile",
            "tests/fixtures/fixed_effect/model.yaml",
            "--output",
            str(run_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    posterior = tmp_path / "posterior.h5"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "sample",
            str(run_dir / "init.json"),
            "--output",
            str(posterior),
            "--samples",
            "1",
            "--transient",
            "0",
            "--thin",
            "1",
            "--verbose",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "pyhmsc", "summarize", str(posterior), "--param", "Beta"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "covariate" in result.stdout
    assert "forest_cover" in result.stdout
    assert "sparrow" in result.stdout

    pred_out = tmp_path / "pred.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "predict",
            str(posterior),
            "--X",
            "tests/fixtures/fixed_effect/X.csv",
            "--random-effects",
            "none",
            "--output",
            str(pred_out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert pred_out.exists()

    ppc_out = tmp_path / "ppc.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "ppc",
            str(posterior),
            "--X",
            "tests/fixtures/fixed_effect/X.csv",
            "--Y",
            "tests/fixtures/fixed_effect/Y_poisson.csv",
            "--random-effects",
            "none",
            "--seed",
            "1",
            "--output",
            str(ppc_out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert ppc_out.exists()
    assert "sparrow" in ppc_out.read_text(encoding="utf-8")

    richness_ppc_out = tmp_path / "richness_ppc.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "ppc",
            str(posterior),
            "--X",
            "tests/fixtures/fixed_effect/X.csv",
            "--Y",
            "tests/fixtures/fixed_effect/Y_poisson.csv",
            "--scope",
            "site-richness",
            "--random-effects",
            "none",
            "--seed",
            "1",
            "--output",
            str(richness_ppc_out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert richness_ppc_out.exists()
    assert "observed_richness" in richness_ppc_out.read_text(encoding="utf-8")

    validate = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "validate",
            str(posterior),
            "--X",
            "tests/fixtures/fixed_effect/X.csv",
            "--Y",
            "tests/fixtures/fixed_effect/Y_poisson.csv",
            "--formula",
            "~ forest_cover + elevation",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "predictive_interval_contains_observed_mean" in validate.stdout
