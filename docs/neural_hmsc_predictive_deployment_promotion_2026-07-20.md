# Neural Predictive Deployment Promotion

Date: 2026-07-20

## Policy

The default neural predictive-mean deployment policy is now
`affine_branch`. The explicit conservative fallback remains `scale_only`.
Qualified Python MCMC remains a statistical reference and is never a member of,
or an input to, neural ensemble prediction.

This promotion applies only to response-scale predictive means. It does not
replace coefficient posterior artifacts, Bayesian uncertainty semantics, or
Python-only/R-boundary HMSC inference.

## API

```python
from pyhmsc import load_predictive_mean_ensemble

# Promoted default: affine_branch
ensemble = load_predictive_mean_ensemble(manifest_dir, dataset="big_spatial")
probability = ensemble.predict_mean(X_new)

# Explicit conservative fallback
fallback = load_predictive_mean_ensemble(
    manifest_dir,
    dataset="big_spatial",
    policy="scale_only",
)
```

The deployment loader verifies member hashes and compatibility, predictive-only
semantics, outcome-independent selection, parity provenance, and ordered
qualified-MCMC reference provenance.

## Scheduler Smoke

- Job: `20032978`
- Partition: `dev-g`
- State: `COMPLETED`, exit code `0:0`
- Elapsed: `00:00:34`
- Run root:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_predictive_deployment_smoke_20260720`
- Decision: `predictive_deployment_smoke_passed`

The scheduler used its unmodified defaults:

- default policy: `affine_branch`;
- fallback policy: `scale_only`;
- qualified Python MCMC role: `statistical_reference_only`.

| Dataset | Prediction shape | Maximum default/fallback difference | Parity provenance | MCMC used for neural prediction |
|---|---:|---:|---|---|
| Whittaker | 12 x 75 | 0.000000 | qualified | no |
| Big Spatial | 360 x 75 | 0.040348 | qualified | no |

The smoke opened covariates only. It did not open target outcomes. The zero
Whittaker difference confirms identity fallback behavior, while the bounded Big
Spatial difference confirms that the promoted branch is active.

Local focused validation before submission: `47 passed`.
