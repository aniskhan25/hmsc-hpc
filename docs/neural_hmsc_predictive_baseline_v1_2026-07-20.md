# Neural Predictive Baseline V1

Date: 2026-07-20

## Frozen Identifier

The promoted neural predictive deployment is frozen under the stable identifier:

`neural_predictive_affine_v1`

LUMI registry location:

`/scratch/project_462000131/anisrahm/hmsc-hpc-deployments/neural_predictive_affine_v1`

The registry entry is immutable: creation is atomic and a second freeze using
the same identifier is rejected. Consumers resolve it with:

```python
from pyhmsc import load_predictive_deployment_baseline

ensemble = load_predictive_deployment_baseline(
    registry_root,
    dataset="big_spatial",
    baseline_id="neural_predictive_affine_v1",
)
probability = ensemble.predict_mean(X_new)
```

The default is `affine_branch`; `policy="scale_only"` selects the explicit
fallback.

## Pinned Contents

The bundle pins four manifests:

| Dataset | Policy | Manifest SHA-256 |
|---|---|---|
| Whittaker | `affine_branch` | `ec14e540496da16a8990c580022ebbfa2371fe3b27d3cf218c533a7dda733aa2` |
| Whittaker | `scale_only` | `e0f08cd96f1727acf24cbc2009d132a56265ef2ceaf1905d6181be3711c770a0` |
| Big Spatial | `affine_branch` | `903f04b9ed66908f19c6dfd6c7f47c41bee2e7f75648373d0255fadb1dd9c51f` |
| Big Spatial | `scale_only` | `af3a0a202b3ecb7585c35818e38d09b853329e521443857e38b6cdbc4ef3aa54` |

It also pins the API requalification and default-wiring smoke JSON evidence by
hash. Bundle validation replays their required decisions and rejects changed
manifests or evidence.

## Frozen Competitor Contract

Every subsequent neural competitor must report against this exact baseline and
retain these gates:

1. coefficient SBC;
2. named OOD regimes;
3. rare-validation diagnostics;
4. Whittaker no degradation;
5. Big Spatial no degradation;
6. full and leave-one-out ensemble stability;
7. manifest and parity provenance.

Qualified Python MCMC remains reference-only. The frozen full-ensemble gaps are:

| Dataset | Neural/MCMC Brier ratio | Neural/MCMC log-loss ratio |
|---|---:|---:|
| Whittaker | 1.021279 | 1.032730 |
| Big Spatial | 1.078996 | 1.073132 |

The next competitor must reduce the Big Spatial gap without using real target
outcomes for fitting or selection and without changing coefficient-posterior
calibration.

## LUMI Freeze

- Job: `20033698`
- Partition: `dev-g`
- State: `COMPLETED`, exit code `0:0`
- Elapsed: `00:00:33`
- Run root:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_predictive_baseline_freeze_20260720`

The stable-ID smoke passed for Whittaker and Big Spatial. Whittaker remained
identity-equivalent to `scale_only`; Big Spatial retained maximum response
probability movement `0.040348`. MCMC was not used for neural prediction and
target outcomes were not opened.

Local focused validation before submission: `50 passed`.
