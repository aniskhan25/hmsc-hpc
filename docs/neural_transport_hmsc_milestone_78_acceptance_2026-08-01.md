# Neural-Transport HMSC Milestone 78 Acceptance

Date: 2026-08-01

Decision: `accept_exact_corrected_transport_direction`

## Reviewed Boundary

Milestone 78 was reviewed at clean source commit
`253e7802642192b0d72427b461bf9fc9cc30fa99`. The accepted design record is:

```text
docs/neural_transport_hmsc_go_no_go_2026-08-01.md
sha256 = 135adb8b2614d75f1aab17f2fbe0d2d379b9c971aec62e2fbfa5968acb6fc887
```

The four static decision tests passed before branch creation. The worktree was
clean and synchronized with `origin/feature/generative-neural-hmsc`.

## Accepted Direction

Accept preregistration for a bounded neural warm start and frozen affine
transport around the existing corrected HMC/Gibbs target. The neural component
may affect initialization and sampling efficiency. It may not define the
accepted posterior, remove the Metropolis correction, replace convergence
diagnostics, or silently handle unsupported inputs.

Reject a third standalone raw-state, orbit-IWAE, flow-only, diffusion-only, or
post-hoc calibrated neural posterior under this decision.

## Branch Boundary

The requested branch was created directly from the accepted commit:

```text
feature/neural-transport-hmsc
```

Milestone 79 authorizes documentation, static seed auditing, and static seal
tests only. It does not authorize model code, simulator execution, training,
MCMC generation, scheduler submission, or opening any seed block.

## Acceptance Conditions Carried Into Milestone 79

- first scope: probit, 40 sites, 12 species, two covariates, two latent factors,
  and one iid site-level random intercept;
- the existing HMSC target and priors remain the statistical authority;
- transformed HMC must include the exact transport Jacobian and retain its
  accept/reject correction;
- ordinary Python MCMC, identity transport, and warm-start-only controls are
  mandatory;
- posterior parity is evaluated before efficiency;
- unsupported or failed transport contexts fall back explicitly to ordinary
  MCMC;
- every production seed must be fresh relative to the complete local ledger;
  and
- failure to improve efficiency after exactness passes is still a candidate
  failure.

Decision: Milestone 78 is accepted. Proceed to the Milestone 79
preregistration and seed audit only.
