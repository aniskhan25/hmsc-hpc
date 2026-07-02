# Neural HMSC Conditional Coefficient Calibration

Date: 2026-07-01

## Implementation

Milestone 12 now includes a structured conditional scale head in
`pyhmsc/neural/conditional_calibration.py`. The head predicts one positive
multiplier per dataset, coefficient, and species from quantities available at
inference time:

- logit nonzero prevalence
- log expected diagonal design information
- log raw neural posterior standard deviation
- centered coefficient identity
- prevalence-by-coefficient interaction

Each continuous feature has a linear and positive-hinge term. The model is
initialized from the existing global scalar multiplier and fitted by minimizing
conditional Gaussian log score on simulated calibration truth. Ridge
regularization shrinks it toward the scalar baseline. A final scalar
normalization targets nominal marginal coverage while preserving conditional
scale ratios.

The calibrator does not alter posterior means. For a full-covariance posterior,
coefficient scales form a diagonal matrix `D` and each per-species Cholesky
factor becomes `D L`, giving covariance `D Sigma D`.

## Semantics

Conditional calibration metadata uses `semantics_version: 3` and method
`conditional_structured_scale`. It stores feature normalization, weights,
coefficient names, multiplier bounds, fitting hyperparameters, calibration
coverage, and scalar-versus-conditional log scores.

Coefficient and predictive calibration remain separate:

- coefficient artifacts may use the conditional version 3 calibrator
- predictive-only artifacts continue to use the scalar version 2 calibrator
- neither ecological dataset nor an MCMC posterior may be used to fit the
  conditional head

## Running

The dedicated entry point forwards all standard benchmark arguments and
selects conditional coefficient calibration:

```bash
python examples/run_neural_hmsc_conditional_calibration.py \
  --output run/conditional \
  --suite probit \
  --n-sites 40 \
  --n-species 75 \
  --train-datasets 512 \
  --calibration-datasets 128 \
  --sbc-datasets 128 \
  --sbc-draws 512 \
  --epochs 120
```

The general runner exposes the same mode through
`--coefficient-calibration conditional`. Optimization can be controlled with
`--conditional-calibration-epochs`,
`--conditional-calibration-learning-rate`, and
`--conditional-calibration-regularization`.

## Validation State

Unit coverage verifies conditional prevalence response, nominal calibration,
metadata round trips, domain rejection, unchanged means, and exact
full-covariance transformation. An end-to-end benchmark smoke test verifies
that coefficient artifacts carry version 3 metadata while predictive artifacts
remain on version 2.

The frozen five-seed in-domain/OOD comparison is recorded in
`docs/neural_hmsc_conditional_comparison_2026-07-02.md`. The implementation
fixed overall in-domain rank variance but failed rare-prevalence and OOD gates;
it is not qualified.
