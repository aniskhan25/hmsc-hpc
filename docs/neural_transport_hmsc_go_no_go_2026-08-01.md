# Neural-Transport HMSC Go/No-Go Review

Date: 2026-08-01

Decision: `go_to_preregistration_for_exact_corrected_neural_transport`

Related closure audit:
`docs/generative_neural_hmsc_branch_closure_audit_2026-08-01.md`

## Decision

Do not build a third standalone amortized joint posterior. The v1 and v2
results show that asking one neural distribution to replace the HMSC posterior
is currently too fragile: v1 trained but collapsed structurally, while v2
represented the latent symmetry more carefully but failed finite
mixed-shape optimization.

Proceed instead to preregistration for a neural augmentation of the qualified
Python sampler:

`neural_transport_hmsc_iid_probit_v0_1`

The neural component will predict a warm start and an invertible,
data-conditioned transport for the existing HMSC state. Sampling will still
use the exact HMSC target with an HMC Metropolis correction and ordinary Gibbs
transitions. The network may improve initialization and geometry; it may not
define the accepted posterior.

This changes the success criterion from:

> Can a neural network approximate the complete HMSC posterior directly?

to:

> Can a neural network reduce warmup and improve effective samples per second
> while the corrected sampler retains the same target posterior?

## Why This Can Address The Failures

| Observed failure | Direct-posterior consequence | Transport-MCMC response |
|---|---|---|
| v1 latent path collapsed toward the no-latent ablation | Neural output was the posterior, so structural error was terminal | A poor transport lowers acceptance or speed; corrected MCMC still targets HMSC |
| v1 marginal and association undercoverage | Post-hoc calibration could not restore absent dependence | Posterior uncertainty comes from corrected MCMC, not the transport scale |
| v2 orbit-IWAE gradients became non-finite | Training failure prevented any checkpoint or inference | Start with a bounded affine bijector and supervised robust location/scale targets; no IWAE or orbit-density training |
| Raw Eta/Lambda are difficult to amortize across datasets | One global density had to represent local and global latent structure | Network supplies initialization and geometry only; existing conditional updates retain the structured state |
| OOD neural behavior is uncertain | Approximate posterior could silently be wrong | Compatibility/support checks can reject the transport and fall back to ordinary MCMC |

Metropolis correction preserves the intended stationary target for a valid
frozen proposal or transformed kernel. Finite runs still require ordinary
convergence diagnostics; this is not permission to shorten chains without
evidence.

## Competing Directions

| Direction | Posterior semantics | Main advantage | Main problem | Decision |
|---|---|---|---|---|
| Data-conditioned affine transport plus corrected HMC/Gibbs | Exact target asymptotically, subject to ordinary MCMC convergence | Failure affects efficiency rather than posterior validity; reuses current TensorFlow target | May produce little speedup | **Primary** |
| Neural warm start only | Exact target after convergence | Lowest implementation and mathematical risk | Can reduce burnin but not sampling autocorrelation | Required ablation and fallback |
| Normalizing-flow independent MH proposals | Exact target with valid proposal density and MH correction | Can add nonlocal global moves | High-dimensional acceptance may collapse; adaptive-MCMC controls required | Deferred until affine transport qualifies |
| Collapsed SBI over `Beta` and association invariants | Approximate unless followed by correction | Targets identifiable quantities and avoids raw factor rotations | Requires a tractable collapsed target or conditional reconstruction | Research alternative, not first |
| Conditional diffusion posterior over invariants | Approximate | Stable expressive neural posterior training | Density/correction is difficult and uncertainty would again require qualification | Do not pursue before transport result |
| Third raw-state/orbit amortized posterior | Approximate | Fast inference if successful | Repeats the failed v1/v2 objective | Rejected |

## Literature Basis

This direction adapts established deep-learning mechanisms rather than copying
an existing JSDM implementation:

- NeuTra HMC learns an invertible neural reparameterization to improve posterior
  geometry while retaining HMC:
  https://research.google/pubs/neutra-lizing-bad-geometry-in-hamiltonian-monte-carlo-using-neural-transport/
- adaptive independent Metropolis-Hastings with normalizing-flow proposals
  provides the relevant corrected-proposal framework:
  https://proceedings.mlr.press/v151/brofos22a.html
- adaptive Monte Carlo with flow-based nonlocal moves shows why the local
  corrected kernel should remain even when the learned map is imperfect:
  https://arxiv.org/abs/2105.12603
- lightweight inference compilation demonstrates learned proposals based on
  local graphical-model context:
  https://proceedings.mlr.press/v130/liang21a.html
- recent analysis of amortized variational inference reinforces that the
  amortization gap depends on model structure and is not automatically removed
  by a larger encoder:
  https://proceedings.mlr.press/v244/margossian24a.html

Conditional diffusion posterior estimation remains a plausible future
approximate-inference direction, but its improved training stability does not
provide the exact-correction path required here:
https://proceedings.mlr.press/v258/chen25d.html

## Frozen First Scope

The preregistration should cover only:

- probit occurrence response;
- fixed effects plus one iid site-level random intercept;
- two fixed covariates including the intercept;
- 40 sites and 12 species;
- exactly two latent factors;
- no traits, phylogeny, spatial effects, random slopes, detection, adaptive
  factor count, Gaussian, or Poisson response; and
- the existing Python-native HMSC target and priors unchanged.

The small fixed shape is deliberate. The first result must establish exact
target preservation and useful acceleration for one structural family before
variable-shape or ecological-scale generalization.

## Proposed Algorithm

### State and target

Reuse the current TensorFlow HMSC log target and state blocks for `Beta`,
`Eta`, `Lambda`, and the applicable positive shrinkage parameters. Existing
Gibbs updates remain the baseline and fallback.

### Neural context encoder

A permutation-aware site/species encoder consumes only compiled model inputs:

- `Y` and response mask;
- `X` and covariate mask;
- `Pi` and random-level dimensions; and
- declared model dimensions and prior hyperparameters.

It must not consume simulation truth, MCMC diagnostics, ecological holdout
outcomes, dataset identity, or gate results at inference time.

### Warm start

The encoder predicts robust locations for `Beta` and the random-effect product.
A deterministic rank-two factorization initializes `Eta` and `Lambda`. These
values initialize the chain only and are never emitted as posterior samples.

### Conditional affine transport

The first candidate is intentionally conservative:

```text
state = location(context) + positive_scale(context) * base_state
```

Location and positive diagonal or block-diagonal scales are frozen before
evaluation. For base state `z` and invertible transport `T`, transformed HMC
uses:

```text
log_target_z(z) = log_target_hmsc(T(z)) + log_abs_det_jacobian_T(z)
```

TFP HMC retains its accept/reject correction. A standard Gibbs transition is
interleaved at a frozen frequency so a weak transport cannot become the sole
route through state space.

### Training data and loss

Training may use only fresh simulated communities and Python-MCMC reference
draws assigned to the training role in the future preregistration. Use robust
posterior centers and scales as supervised transport targets. Do not optimize
ELBO, IWAE, SBC ranks, evaluation gates, or real ecological outcomes.

The warm-start-only network and identity transport are mandatory ablations.

## Implementation Milestones

### Milestone A: Preregistration and seed audit

- freeze target, state blocks, shape, priors, transform, training targets,
  fallback, baselines, gates, thresholds, and fresh seed roles;
- prove that the frozen transformed target includes the correct Jacobian and
  leaves the HMSC target invariant under the corrected kernel; and
- define a terminal stop rule before implementation.

No simulation generation occurs in this milestone.

### Milestone B: Ordinary-fixture implementation

- extract a reusable target/state adapter from `hmsc/updaters/updateHMC.py`;
- implement context encoder, warm-start adapter, and positive affine bijector;
- implement transformed HMC and explicit ordinary-Gibbs fallback;
- preserve the existing sampler API unless the transport is explicitly
  requested; and
- add checkpoint/provenance schema marked as sampler-acceleration-only.

Required ordinary tests:

- forward/inverse/Jacobian parity against dense references;
- transformed versus direct target equality;
- deterministic identity-transport equality;
- finite gradients at maximum declared shape;
- stationary-moment parity on exact Gaussian and small probit fixtures;
- permutation, padding, checkpoint, and compatibility rejection;
- forced transport rejection followed by successful Gibbs fallback; and
- immutable regression hashes for existing neural releases and MCMC behavior.

### Milestone C: Disposable qualification

Use fresh disposable seeds only after Milestones A and B pass. Compare:

1. ordinary Python MCMC from its existing initialization;
2. Python MCMC with neural warm start only; and
3. neural affine transport with corrected HMC/Gibbs.

The disposable gate must require all posterior-parity checks before considering
speed.

### Milestone D: Fixed validation and real data

Only a disposable pass may open fixed validation. Real ecological evaluation
is last and may measure runtime, convergence, posterior summaries, and proper
scores; it may not train or select the transport.

## Qualification Gates

### Exactness and parity gates

- direct and transformed log-target parity within numerical tolerance;
- no material difference from baseline MCMC in `Beta`, association, shrinkage,
  posterior predictive, and rank/coverage summaries beyond preregistered Monte
  Carlo error bounds;
- four-chain convergence thresholds for both baseline and candidate;
- no chain, stratum, or association-direction degradation hidden by aggregate
  metrics; and
- exact fallback behavior for unsupported or low-support contexts.

### Efficiency gates

Efficiency is evaluated only after every parity gate passes:

- at least 25% lower median time to the frozen convergence target;
- at least 1.25x median effective samples per second across primary identifiable
  summaries;
- no primary summary below 0.90x baseline ESS/second;
- transport overhead included in end-to-end time; and
- checkpoint training cost and break-even dataset count reported separately.

These thresholds are proposals for preregistration review, not yet frozen
qualification gates.

## Failure And Fallback Semantics

- A transport compatibility failure selects ordinary Python MCMC before
  sampling.
- Non-finite network output, transform output, Jacobian, or target evaluation
  aborts the transport path and records the reason; it cannot silently emit
  samples.
- Low acceptance or failed convergence is a candidate failure, not evidence
  for shortening baseline chains.
- Failure to improve efficiency closes the transport candidate even if
  posterior parity passes. Parity without acceleration is not a neural
  capability gain.
- The existing qualified fixed-effect neural releases remain unchanged.

## Expected Outcome

If successful, this work will not create a standalone neural posterior. It will
create an HMSC sampler whose **output remains MCMC**, but whose initialization
and geometry are informed by a neural network. Posterior semantics therefore
come from the corrected HMSC kernel, while the neural contribution is measured
as faster convergence and higher ESS/second.

This is the credible route to near-MCMC equivalence: not approximating MCMC
more aggressively, but accelerating MCMC without changing its target. It still
qualifies only one bounded iid-probit structural family; traits, phylogeny,
spatial effects, and broader likelihoods remain future work.

## Next Step

Create a new branch only after reviewing this decision. On that branch, write
and hash-freeze the complete Milestone A preregistration and unused-seed audit.
Do not implement the encoder, transport, or scheduler and do not generate any
simulation until that preregistration is accepted.
