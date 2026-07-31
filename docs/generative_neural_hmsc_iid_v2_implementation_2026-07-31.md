# Generative Neural-HMSC IID v2 Implementation Evidence

Date: 2026-07-31

Protocol: `generative_neural_hmsc_iid_probit_v2_orbit`

## Scope

This record closes the ordinary-fixture implementation step authorized by the
frozen v2 representation decision. No disposable, production, validation, or
reserved ledger seed was opened.

The implementation changes the variational representation and encoder only.
It imports and reuses the unchanged v1 state layout, prior density, and probit
likelihood. It contains no calibration, simulator-truth loss, MCMC teacher,
router, fallback, selector, or gate adjustment.

## Implemented Representation

The new module `pyhmsc/neural/generative_iid_v2.py` provides:

- a masked multivariate Student-t global posterior with diagonal-plus-rank-16
  scale and dataset-level degrees of freedom constrained to `[4, 30]`;
- a masked matrix-Normal latent base distribution with
  diagonal-plus-rank-16 row covariance;
- exact O(2) orbit sampling and log density through the frozen two-component
  modified-Bessel integral;
- conditional global-to-latent dependence through the six frozen
  alpha/log-tau/Beta moment summaries and a row-local FiLM mean;
- four edge-aware permutation-equivariant bipartite attention blocks with
  width 96, four heads, and feed-forward width 192;
- exactly four common-random first-order IWAE refinement steps with the frozen
  step sizes and at most three halvings;
- exact assembly back into the unchanged raw
  `[alpha, Beta, Eta, Lambda, log_tau]` state;
- analytic invariant `Beta`, `R`, and `C` moments without a dense state
  covariance;
- the unchanged eight-sample IWAE target and inherited outer training
  schedule.

The artifact module `pyhmsc/neural/generative_iid_v2_artifact.py` provides a
separate schema-v2 checkpoint kind, exact file inventory and hashes, frozen
architecture/refinement metadata, explicit absence of calibration and external
dependencies, v1 regression hashes, tamper rejection, and v1/v2 compatibility
rejection.

The experimental public namespace exposes the v2 model, posterior math,
artifact loader, objective, and training entry point without changing the
stable sampler API.

## Ordinary-Fixture Gates

All preregistered feasibility gates passed:

| Gate | Result |
| --- | --- |
| Student-t low-rank log density versus dense reference | pass |
| Matrix-Normal low-rank log density versus dense reference | pass |
| Exact orbit density versus 4096-angle quadrature over both O(2) components | pass, absolute delta below `1e-6` |
| Orbit, prior, likelihood, random-effect, and association invariance | pass |
| Four-step refinement gradients | all trainable gradients present and finite |
| Common-random refinement acceptance | every accepted score nondecreasing |
| Hidden-response isolation | pass |
| Site/species permutation equivariance | maximum delta below `2e-5` |
| Padding/batch-member invariance | maximum delta below `2e-5` |
| Raw-state target identity | exact array equality |
| Checkpoint roundtrip and tamper rejection | pass |
| v1 checkpoint rejection as v2 | pass |
| Immutable v1 source regression | exact hashes unchanged |
| Maximum `96 x 75` full refined inference | pass with rank-16 factors only |

The maximum-shape test runs the actual four-step, eight-draw refinement before
sampling and evaluating the joint density. It does not instantiate a
state-dimension-by-state-dimension covariance.

## Defects Found Before Freeze

Two implementation defects were caught and corrected before this record:

1. The unrefined checkpoint-build path initially did not instantiate the
   theta-conditioned latent mean, so that layer's weights were absent from the
   checkpoint. The model now builds every declared posterior component before
   save; exact conditional-mean roundtrip is tested.
2. TensorFlow's stateful gamma sampler interprets its shape as a sample prefix,
   unlike the full-shape stateless call. The Student-t radial sampler now uses
   the correct contract for both paths, and both sample shapes are tested.

A separate one-epoch ordinary-fixture training smoke completed after these
fixes with finite loss, IWELBO, and gradient norm.

## Verification

Commands and outcomes:

```text
pytest -q --runslow tests/test_neural_hmsc_generative_iid_v2.py
9 passed

pytest -q tests/test_neural_hmsc_generative_iid_v1.py -m 'not slow'
45 passed, 1 deselected

pytest -q tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_release.py -m 'not slow'
26 passed
```

The v2 source inventory at review time is:

```text
155292c9f8edc027ae64f1b1a0046998927308be24889662dab1b90bd48a4cbb  pyhmsc/neural/generative_iid_v2.py
2b950e83c6166352bbe7cc0e0e9baee709a6b157f016f18966539634ff2dac6d  pyhmsc/neural/generative_iid_v2_artifact.py
23c4794f418c82f75feaf3791a4b96db424fd9a62eb066e0af91b3b4a7c122f4  tests/test_neural_hmsc_generative_iid_v2.py
4a3125dea1733e4b386b3634a2c7f293bec828b37b6f18e628d612b04671468d  pyhmsc/neural/__init__.py
```

The immutable v1 regression inventory remains:

```text
a7885c9123ac4e52beb1ed366fd5c09857f132789e21cac540be6c96663b8d52  pyhmsc/neural/generative_iid.py
fb6429a5a58eee2caffcd1f33118847db269b53cfdcd4fc3556d9ae1ed523cac  pyhmsc/neural/generative_iid_artifact.py
```

## Decision Boundary

The representation is implementable and passes its mathematical and software
feasibility gates. It is not statistically qualified. In particular, this
record makes no claim about recovery, calibration, exact-MCMC comparison,
association learning, prediction, or real-data transfer.

The 593M-594M disposable smoke and every 511M-515M role remain sealed.

## Next Step

Freeze this implementation and evidence in one clean commit. Then implement
and review the sealed 593M-594M disposable harness using only the frozen
architecture, objective, gates, and seed roles. Run a token-free/no-seed
preflight before any separate smoke authorization.
