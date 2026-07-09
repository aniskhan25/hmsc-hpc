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
initialized from the existing global scalar multiplier. Version 4 combines a
prevalence-weighted Gaussian log score with differentiable analytic SBC
rank-mean and rank-variance penalties. Rare and intermediate species receive
greater fitting weight. Ridge regularization shrinks the head toward the scalar
baseline, and a final scalar normalization targets nominal marginal coverage.

Version 4 also stores robust feature bounds and a regularized Mahalanobis
support radius from simulated calibration data. Conditional adjustments are
blended with the scalar multiplier in log space; trust decays beyond either
support boundary and reaches the scalar fallback under substantial feature
shift.

The calibrator does not alter posterior means. For a full-covariance posterior,
coefficient scales form a diagonal matrix `D` and each per-species Cholesky
factor becomes `D L`, giving covariance `D Sigma D`.

## Semantics

Anchored-model conditional calibration metadata uses `semantics_version: 5`
and method `conditional_rank_aware_anchor_scale`. It stores feature normalization,
weights, coefficient names, multiplier bounds, rank-objective settings,
feature-support geometry including posterior-mean magnitude, fitting
hyperparameters, calibration coverage, and
scalar-versus-conditional log scores and rank losses. Version 3
`conditional_structured_scale` and version 4
`conditional_rank_aware_scale` metadata remain loadable for reproducibility.

Coefficient and predictive calibration remain separate:

- coefficient artifacts may use the conditional version 5 calibrator
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
`--conditional-calibration-regularization`. Rank weighting is controlled by
`--conditional-calibration-rank-penalty-weight` and the three prevalence
weights. Support fallback is controlled by
`--conditional-calibration-support-quantile` and
`--conditional-calibration-fallback-strength`.

## Validation State

Unit coverage verifies conditional prevalence response, nominal calibration,
metadata round trips, version 3/4 compatibility, rank-loss improvement, scalar
fallback outside feature support, posterior-mean shift detection, domain
rejection, unchanged means, and exact full-covariance transformation. An
end-to-end benchmark smoke test verifies that anchored coefficient artifacts
carry version 5 metadata while predictive artifacts remain on version 2 and
SBC rows expose support-trust and mean-magnitude diagnostics.

The frozen five-seed in-domain/OOD comparison is recorded in
`docs/neural_hmsc_conditional_comparison_2026-07-02.md`. The implementation
fixed overall in-domain rank variance but failed rare-prevalence and OOD gates.
That result applies to the version 3 objective. The version 4 comparison is
recorded in `docs/neural_hmsc_rankaware_v4_comparison_2026-07-09.md`. Version 4
fixed rare coverage and recovered most OOD degradation, but prevalence
rank-mean and intercept rank-variance gates still failed. It is not qualified.
The IRLS/Laplace anchor and version 5 support extension are implemented but not
yet LUMI-qualified.
