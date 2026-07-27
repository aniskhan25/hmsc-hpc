from pathlib import Path


def test_scheduler_defaults_to_promoted_affine_with_scale_fallback():
    text = Path(
        "docs/lumi_neural_hmsc_predictive_deployment_smoke_sbatch.sh"
    ).read_text(encoding="utf-8")

    assert 'PREDICTIVE_MEAN_POLICY="${PREDICTIVE_MEAN_POLICY:-affine_branch}"' in text
    assert (
        'PREDICTIVE_MEAN_FALLBACK_POLICY="${PREDICTIVE_MEAN_FALLBACK_POLICY:-scale_only}"'
        in text
    )
    assert "Qualified Python MCMC: statistical reference only" in text
    assert "--policy \"${PREDICTIVE_MEAN_POLICY}\"" in text
    assert "--fallback-policy \"${PREDICTIVE_MEAN_FALLBACK_POLICY}\"" in text


def test_deployment_smoke_does_not_open_target_outcomes():
    text = Path(
        "examples/smoke_neural_hmsc_predictive_deployment.py"
    ).read_text(encoding="utf-8")

    assert "Y.csv" not in text
    assert '"mcmc_used_for_neural_prediction": False' in text
    assert '"target_response_opened": False' in text
