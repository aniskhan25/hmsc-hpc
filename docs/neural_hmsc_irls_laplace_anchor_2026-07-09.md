# Neural HMSC Probit IRLS/Laplace Anchor

Date: 2026-07-09

## Purpose

Conditional scale calibration versions 3 and 4 could not remove directional
SBC rank bias for rare and intermediate species. The fixed-shape probit
amortizer was anchored to a linear ridge regression of binary responses, so the
neural residual had to learn the probit-link transformation as well as
posterior correction.

New probit checkpoints replace that linear anchor with a deterministic,
penalized probit mode and Laplace information summary.

## Algorithm

For each simulated dataset and species, the anchor performs fixed-iteration
Fisher scoring:

```text
p = Phi(X beta)
w = phi(X beta)^2 / (p * (1 - p))
z = X beta + (y - p) / phi(X beta)
beta_next = solve(X' W X + prior_precision * I, X' W z)
```

Linear predictors and probabilities are clipped for numerical stability. The
inverse final penalized information matrix provides Laplace marginal standard
deviations. The encoder receives:

- the previous response/ridge/design summaries
- the probit mode
- log Laplace marginal standard deviations

The posterior mean is the probit mode plus the learned neural residual.
Calibration remains separate and cannot modify posterior means.

## Compatibility

- new checkpoints use `checkpoint_version: 0.3`
- version `0.2` checkpoints remain loadable and default to the exact legacy
  `ridge` anchor
- `probit_anchor: auto` resolves to `irls_laplace` for probit and `ridge` for
  Gaussian and Poisson models
- checkpoint manifests store anchor type, iterations, prior precision, and
  eta-clipping threshold

Calibration artifacts fitted to anchored models initially used
`semantics_version: 5` and method `conditional_rank_aware_anchor_scale`.
Version 5 extends support diagnostics with standardized
`log1p(abs(posterior_mean))` bounds so effect-size shift can trigger scalar
fallback. The subsequent OOD-aware uncertainty update writes
`semantics_version: 6` for the same method and adds bounded support-excess
inflation after scalar fallback. Versions 3, 4, and legacy 5 remain loadable.

## Benchmark Controls

The general and LUMI benchmark workflows expose:

```text
--probit-anchor
--probit-anchor-iterations
--probit-anchor-prior-precision
--probit-anchor-eta-clip
--conditional-calibration-ood-uncertainty-strength
--conditional-calibration-ood-uncertainty-max-multiplier
```

The next validation must train anchored and legacy candidates from the same
seeded simulation corpora. Frozen version 4 checkpoints remain reference
artifacts, but cannot be reused as the anchored candidate because the encoder
input and deterministic mean anchor changed.
