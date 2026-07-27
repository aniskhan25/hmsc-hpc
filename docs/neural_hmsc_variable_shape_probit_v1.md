# Neural-HMSC Variable-Shape Probit V1

Date: 2026-07-21

## Decision

Milestone 52 is qualified and promoted for a bounded variable-shape scope:

- fixed-effect probit `Beta` inference;
- exactly three ordered covariates: `Intercept`, `x1`, `x2`;
- 12-48 sites;
- 2-10 species;
- no traits, phylogeny, random effects, spatial effects, or detection submodel.

The immutable identifier is `neural_hmsc_variable_probit_v1`. Its durable LUMI
location is:

`/scratch/project_462000131/anisrahm/hmsc-hpc-deployments/neural_hmsc_variable_probit_v1`

This is a separate baseline. The fixed-shape `neural_hmsc_v0_1` release was not
modified.

| Record | SHA-256 |
| --- | --- |
| Baseline content inventory | `badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9` |
| `baseline.json` | `b9387efc147ecd9e3978c80cb2cc2a3ebdcddd5c68445c8b63ce8f37af61a2f1` |
| Multi-seed qualification | `0ba58fd9bc4d49710068881ef41d3d86010aeef988351a9c79b3b40c287e02ce` |
| Candidate checkpoint manifest | `cf46ebfdfc457e71a0da28f48f7709613f7e47b101b946553f711d5e1e4f47a5` |
| Candidate weights | `70ef4548eeb1dc3a0d9367cb8edaedb5a2030370179241f35b372aecd8d5c4cd` |
| Candidate calibration | `c3c8fd4ff50583ced5273c009e501ea0b6f400ff144a74f510513633edd7b771` |

## Representation

The model uses masked sufficient-statistic pooling and is invariant to site
order and equivariant to species order. A mask-aware penalized probit
IRLS/Laplace estimate supplies the posterior mean and scale anchor. The neural
head learns bounded posterior-mean and log-scale corrections around that
anchor. Padded sites are excluded from prevalence, information, loss, and
calibration calculations.

The packaged scalar scale calibration was fitted from an independent simulated
corpus across the predeclared shape range. It has coefficient-posterior
semantics and is not a predictive-only response calibration.

## Qualification

Three independent runs used 64 training, 32 calibration, and 32 test
communities each. Every run included both shape boundaries and used disjoint
train/calibration/test seeds. Two Python-MCMC comparisons per run were scored
on held-out sites, not the fitting sites.

| Metric | Three-seed result |
| --- | ---: |
| Mean 95% coefficient coverage | 0.9512 |
| Coverage range | 0.9451-0.9557 |
| Mean normalized rank mean | 0.4962 |
| Rank-mean range | 0.4818-0.5065 |
| Mean rank variance | 0.0845 |
| Mean neural/MCMC Brier ratio | 0.9997 |
| Worst neural/MCMC Brier ratio | 1.0339 |
| Mean neural/MCMC log-loss ratio | 1.0013 |
| Worst neural/MCMC log-loss ratio | 1.0297 |
| Candidate neural inference per dataset | 0.0291 seconds |
| Candidate MCMC time per fitted holdout comparison | 14.1 seconds |

All checkpoint round-trip, boundary-shape, coefficient coverage, rank,
IRLS/Laplace no-degradation, and held-out MCMC proper-score gates passed on all
three runs. Seed `20260730` was predeclared as the release candidate;
`20260731-20260732` were sensitivity evidence and did not select the candidate.

## Public API

```python
from pyhmsc import load_variable_shape_baseline

engine = load_variable_shape_baseline("/path/to/hmsc-hpc-deployments")
report = engine.check_compatibility(compiled_init_json)
fit = engine.infer(compiled_init_json, draws=1000, chains=2)
```

Run the complete stable-ID example:

```bash
python examples/run_neural_hmsc_variable_shape_baseline.py \
  --registry-root /private/tmp/neural_hmsc_variable_deployments \
  --n-sites 30 \
  --n-species 6 \
  --output /private/tmp/neural_hmsc_variable_shape_example
```

Inputs outside the declared ranges, changed covariate order/formula, changed
distribution, traits, or random effects fail before inference.

## Claim Boundary

The result qualifies a fast marginal `Beta` approximation within the declared
probit range. It does not establish joint-posterior MCMC equivalence, structural
HMSC equivalence, or support for Normal, Poisson, traits, associations, or
spatial effects. Qualified Python MCMC remains the statistical reference.

## Next Step

Milestone 53 trait-Gamma qualification subsequently ended in a preregistered
terminal failure. Under Milestone 54, generalize the fixed-effect probit
representation to variable design-column counts and ecology-scale species
counts. Keep both `neural_hmsc_v0_1` and
`neural_hmsc_variable_probit_v1` immutable regression baselines. See
`docs/neural_hmsc_post_m53a_scope_decision_2026-07-22.md`.
