# Predictive Ensemble API Requalification

Date: 2026-07-20

## Scope

This step packages the frozen three-member probability ensemble as a reusable,
predictive-only deployment artifact and requalifies it on Whittaker and Big
Spatial. The comparison uses:

- the matched three-member `scale_only` neural ensemble;
- the three-member `affine_branch` neural ensemble;
- the matched qualified Python MCMC response-probability ensemble as a
  statistical reference.

The MCMC comparator is not an ensemble member and does not alter the neural
promotion gate. This step does not claim that neural inference is equivalent
to Python-only or R-boundary HMSC inference.

## Implementation

`PredictiveProbabilityEnsemble` provides:

- ordered members with immutable SHA-256 hashes and seeds;
- a single predictive calibration role per artifact;
- distribution, formula, covariate, species, parameter, and artifact-role
  compatibility checks;
- acceptance, run-metadata, parity-metrics, and MCMC-reference provenance with
  independent SHA-256 validation;
- response-scale `predict_mean` aggregation by arithmetic probability mean;
- ordered seed subsets for leave-one-seed-out stability checks.

The deployment surface is:

```python
from pyhmsc import PredictiveProbabilityEnsemble

ensemble = PredictiveProbabilityEnsemble.from_manifest(manifest_path)
probability = ensemble.predict_mean(X_new)
```

The evaluator creates and reloads separate scale-only and affine manifests for
each dataset. Every full and leave-one-out neural prediction is produced through
the reloaded API. Target outcomes remain unopened until all neural and MCMC
predictions are frozen.

## LUMI Run

- Job: `20032745`
- Partition: `dev-g`
- State: `COMPLETED`, exit code `0:0`
- Elapsed: `00:01:19`
- Run root:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_probability_ensemble_api_requalification_20260720`
- Frozen source:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_source_transfer_realdata_sensitivity_20260720`
- Seeds: `20260721`, `20260722`, `20260723`

All four manifests reloaded with member, compatibility, and parity-provenance
hash validation. The decision was
`predictive_ensemble_api_requalification_passed`.

## Neural Result

All eight full and leave-one-out dataset rows passed Brier, log-loss, RMSE, and
richness-MAE no-degradation gates.

| Dataset | Ensemble | Brier ratio | Log-loss ratio | RMSE ratio | Richness-MAE ratio |
|---|---|---:|---:|---:|---:|
| Whittaker | Full | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Big Spatial | Full | 0.997524 | 0.994865 | 0.998761 | 0.985655 |
| Big Spatial | Leave out 20260721 | 0.999187 | 0.997267 | 0.999593 | 0.993119 |
| Big Spatial | Leave out 20260722 | 0.999409 | 0.997107 | 0.999705 | 0.992679 |
| Big Spatial | Leave out 20260723 | 0.994730 | 0.990819 | 0.997361 | 0.971293 |

Ratios are affine divided by matched scale-only; lower is better. Whittaker
uses the identity branch, while Big Spatial retains a small but stable gain.

## MCMC Reference

The full-ensemble proper scores were:

| Dataset | Model | Brier | Log loss | RMSE | Richness MAE |
|---|---|---:|---:|---:|---:|
| Whittaker | Neural affine | 0.075542 | 0.270509 | 0.274849 | 3.735133 |
| Whittaker | Qualified Python MCMC | 0.073968 | 0.261936 | 0.271971 | 3.362885 |
| Big Spatial | Neural affine | 0.051218 | 0.205443 | 0.226314 | 4.930266 |
| Big Spatial | Qualified Python MCMC | 0.047468 | 0.191443 | 0.217872 | 3.534128 |

Neural affine divided by MCMC was `1.0213` Brier and `1.0327` log loss on
Whittaker, and `1.0790` Brier and `1.0731` log loss on Big Spatial. MCMC also
remained stronger in every leave-one-out comparison.

## Decision

The reusable predictive-only ensemble artifact and API pass requalification.
The affine ensemble qualifies for promotion over the matched neural scale-only
deployment baseline because it is identity-safe on Whittaker and stably improves
Big Spatial. Qualified Python MCMC remains the stronger statistical reference,
especially for Big Spatial; the promotion must therefore be described as a
neural predictive deployment improvement, not HMSC posterior or predictive
parity.

Local focused validation before submission: `44 passed`.
