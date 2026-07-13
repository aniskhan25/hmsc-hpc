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

Anchored-model conditional calibration metadata initially used
`semantics_version: 5` and method `conditional_rank_aware_anchor_scale`. The
OOD-aware update writes `semantics_version: 6` for the same method and adds
bounded support-excess uncertainty inflation after scalar fallback. The metadata
stores feature normalization, weights, coefficient names, multiplier bounds,
rank-objective settings, feature-support geometry including posterior-mean
magnitude, fitting hyperparameters, calibration coverage, and
scalar-versus-conditional log scores and rank losses. Version 3
`conditional_structured_scale`, version 4 `conditional_rank_aware_scale`, and
legacy version 5 metadata remain loadable for reproducibility.

The learned OOD-objective update writes `semantics_version: 7` for the same
method when held-out OOD calibration batches are supplied. Version 7 replaces
the fixed support-excess exponential with a learned bounded softplus curve. The
first version used support excess alone. The effect-size-aware revision keeps
legacy support-only v7 metadata loadable and fits a five-parameter curve over
both support excess and positive standardized posterior-mean magnitude. The
additional effect-size signal is intended to trigger under coefficient
magnitude shifts where covariate/support trust remains high. The curve is fit
only from simulated OOD calibration batches, penalizes OOD coefficient coverage
and rank-moment errors, and includes an in-domain gate penalty so in-domain SBC
acceptance remains a hard constraint. Posterior means remain fixed.

The gated effect-size update writes `semantics_version: 8` when the opt-in
`support_effect_gated_rank_coverage` objective is used. Version 8 serializes
`support_effect_gated_learned_softplus`, keeps the support-excess branch, and
multiplies the effect-size branch by a learned OOD-context gate over support
excess and effect-signal magnitude. Version 8 also adds a direct in-domain
extra-inflation penalty during OOD-objective fitting. Legacy support-only and
ungated effect-aware version 7 metadata remain loadable.

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

The learned OOD objective is opt-in:

```bash
python examples/run_neural_hmsc_conditional_calibration.py \
  --output run/conditional_ood_objective \
  --suite probit \
  --conditional-calibration-ood-objective support_excess_rank_coverage \
  --conditional-calibration-ood-datasets 128 \
  --conditional-calibration-ood-objective-epochs 200 \
  --conditional-calibration-ood-uncertainty-max-multiplier 8 \
  --ood-regimes covariate_shift effect_size_shift combined_shift
```

The OOD calibration batches are separate from the SBC batches. The same runner
still writes predictive-only artifacts with scalar version 2 calibration.

## Validation State

Unit coverage verifies conditional prevalence response, nominal calibration,
metadata round trips, version 3/4 compatibility, rank-loss improvement, scalar
fallback outside feature support, posterior-mean shift detection, domain
rejection, unchanged means, exact full-covariance transformation, and version 7
learned OOD-objective metadata/application, including legacy support-only v7
metadata, plus version 8 gated effect-size metadata/application. An
end-to-end benchmark smoke test verifies that anchored coefficient artifacts
carry version 5 metadata while predictive artifacts remain on version 2 and
SBC rows expose support-trust, mean-magnitude, and effect-size signal
diagnostics.

The frozen five-seed in-domain/OOD comparison is recorded in
`docs/neural_hmsc_conditional_comparison_2026-07-02.md`. The implementation
fixed overall in-domain rank variance but failed rare-prevalence and OOD gates.
That result applies to the version 3 objective. The version 4 comparison is
recorded in `docs/neural_hmsc_rankaware_v4_comparison_2026-07-09.md`. Version 4
fixed rare coverage and recovered most OOD degradation, but prevalence
rank-mean and intercept rank-variance gates still failed. It is not qualified.
The IRLS/Laplace anchor, version 6 support-excess inflation, and version 7
learned OOD objective are implemented. Version 7 still needs a five-seed LUMI
comparison against scalar, version 4, version 5 IRLS, version 6 default, and the
conservative version 6 sweep candidate.
