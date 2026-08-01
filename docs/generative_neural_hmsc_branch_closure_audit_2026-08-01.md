# Generative Neural-HMSC Branch Closure Audit

Date: 2026-08-01

Branch: `feature/generative-neural-hmsc`

Decision: `close_current_branch_without_structural_neural_qualification`

## Executive Finding

This branch did not qualify a generative neural HMSC model. It produced useful
simulation, exact-reference, artifact, and sealed-evaluation infrastructure,
but both end-to-end iid latent-factor candidates failed before promotion.

The project still has two genuinely qualified neural capabilities inherited
from the earlier fixed-effect work:

1. bounded marginal fixed-effect probit coefficient inference; and
2. bounded response-probability prediction for named deployment workflows.

Neither capability is a joint HMSC posterior or an MCMC replacement. Qualified
Python MCMC remains the statistical and structural reference.

The defensible project description is:

> Python-native HMSC/MCMC with optional GPU-accelerated neural fixed-effect
> probit surrogates and predictive-only ensembles under explicit support
> boundaries.

The branch must not be described as generative Neural HMSC, neural iid HMSC,
joint-posterior HMSC inference, or near-equivalent to MCMC.

## Classification Rules

- **Qualified:** passed frozen simulation, artifact, compatibility, and required
  real-data or comparator gates for an explicit bounded scope.
- **Predictive-only:** qualified only for response-scale heldout prediction; no
  parameter-posterior or structural semantics.
- **Infrastructure-only:** tested and reusable engineering with no ecological or
  statistical model claim.
- **Failed:** opened its frozen qualification gate and did not pass it.
- **Unsupported:** never qualified for the neural path.

## Capability Matrix

| Capability | Classification | Evidence boundary | Allowed outcome |
|---|---|---|---|
| `neural_hmsc_v0_1` | **Qualified, marginal-only** | Fixed-effect probit `Beta`; exactly 40 sites, 75 species, and two ordered coefficients; calibrated marginal coverage and bounded Whittaker/Big Spatial proper-score ratios | Retain immutable release; repeated compatible inference only |
| `neural_hmsc_variable_probit_v1` | **Qualified, marginal-only** | Fixed-effect probit `Beta`; exactly three ordered coefficients, 12-48 sites, and 2-10 species; three-seed SBC and heldout MCMC gates passed | Retain immutable variable-shape baseline |
| Manifest-backed `affine_branch` ensemble | **Predictive-only** | Frozen Whittaker and Big Spatial response probabilities; full and leave-one-seed-out no-degradation gates | Retain for named workflows; keep `scale_only` fallback |
| Packaged `external_monotone` calibration | **Qualified only within v0.1** | Marginal coefficient calibration packaged and validated with each fixed-shape probit checkpoint | Retain unchanged; do not interpret as joint calibration |
| Gaussian and Poisson neural paths | **Experimental** | Implemented, but no retained production-shape release qualification | Keep non-default and unqualified |
| Generative iid v1 | **Failed** | 502M fixed validation: 26/65 gates passed; severe undercoverage, near-zero association recovery, and performance close to no-latent ablation | Archive model/checkpoints as negative evidence; never deploy |
| Generative iid v2 orbit posterior | **Failed** | Terminal 593M-594M smoke: 36/36 corpus fingerprints matched and zero Cholesky warnings, but training raised `non-finite v2 gradient` before checkpoint creation | Close representation permanently; never open 511M-515M |
| Trait/phylogeny, iid/spatial random effects, latent associations, detection, variable design columns, and broader likelihoods | **Unsupported by neural releases** | No neural structural family passed qualification; prior Gamma, variable-design, covariance, and Student-t attempts also failed | Route to qualified Python MCMC |
| Python-native HMSC/MCMC and direct R-boundary parity | **Qualified reference, not neural** | Fixed, trait/phylogeny, compact iid/spatial fixtures, and Big Spatial full-spatial requalification | Retain as statistical comparator and fallback |

## What Is Genuinely Working

The fixed-effect neural path is useful when all compatibility checks pass:

- amortized marginal `Beta` inference is fast after checkpoint training;
- fixed-shape v0.1 and bounded variable-shape v1 passed their frozen SBC and
  proper-score gates;
- immutable manifests reject unsupported distributions, formulas, dimensions,
  traits, and random effects;
- posterior output interoperates with `HmscFit`; and
- the predictive ensemble provides bounded response-scale deployment for the
  two named ecological workflows.

This is a real but narrow result. It is not evidence that the neural path
learns the HMSC joint posterior or residual species association.

## What Failed And Why

### Generative iid v1

The v1 raw-state joint Gaussian was trainable and operationally fast, but its
posterior was statistically wrong. Beta, association, alpha, and log-tau
coverage failed; association truth correlation was effectively zero; and the
candidate behaved like its no-latent ablation. A single Gaussian over raw,
rotationally non-identifiable Eta/Lambda coordinates collapsed the structural
path rather than learning it.

### Generative iid v2

The one permitted redesign replaced the raw Gaussian with a Student-t global
block, exact O(2)-orbit latent density, conditional dependence, attention, and
fixed semi-amortized IWAE refinement. Its isolated mathematical tests passed.

It did not establish end-to-end trainability on the frozen mixed-shape corpus.
The backend-specific Cholesky defect was repaired and the terminal run emitted
zero Cholesky warnings, but the unchanged objective still produced a
non-finite gradient before completing the first disposable checkpoint. The
immediate non-finite operation was not localized further because the terminal
stop rule prohibited another repair cycle. Scientifically, the representation
failed the minimum feasibility requirement: finite reproducible optimization
under its declared workload.

### Common Shortcoming

The desired capability was a calibrated joint posterior over fixed effects,
latent factors, loadings, and association. v1 was numerically trainable but
structurally collapsed; v2 represented the symmetry more carefully but was too
brittle to train end to end. Neither reached a checkpoint that passed both
trainability and statistical structural gates. Post-hoc calibration could not
repair either failure without changing the scientific target.

## Component Disposition

### Retain As Qualified Product

- immutable `neural_hmsc_v0_1` release;
- immutable `neural_hmsc_variable_probit_v1` baseline;
- manifest-backed affine predictive ensemble and explicit scale-only fallback;
- packaged v0.1 external-monotone coefficient calibration; and
- Python-native HMSC/MCMC, R-boundary parity fixtures, and real-data reference
  workflows.

### Retain As Reusable Infrastructure

- fixed-effect and iid generative simulators;
- exact iid log joint and exact-model MCMC comparator;
- masking, padding, batching, permutation, and compatibility fixtures;
- posterior HDF5 and `HmscFit` interoperability;
- immutable artifact schemas and content-hash validation;
- seed ledgers, token-free preflights, one-shot authorization, sharded recovery,
  freeze, and post-freeze validation patterns; and
- SBC, rank, association, invariant, predictive, and real-data gate evaluators.

These components lower the cost of testing another model. They do not inherit
qualification from the models they evaluated.

### Archive As Research-Only

- generative iid v1 model, checkpoints, evaluator, and 501M/502M evidence;
- generative iid v2 model, artifact loader, schedulers, and 593M/594M evidence;
- prior failed Gamma, variable-design, covariance-overlay, Student-t-overlay,
  teacher, router, and post-hoc calibration families; and
- every negative-result report and frozen failure artifact.

Research-only modules may remain importable from the experimental
`pyhmsc.neural` namespace for reproducibility, but must not be promoted through
the stable top-level API or deployment registry.

### Retire Without Opening

The closed v2 production and evaluation ledgers `511M-515M` remain sealed.
They are retired with this representation and must not be reused to make a new
candidate appear independent. Any future family requires fresh seed blocks.

## Final Claim Boundary

The current neural result is a fixed-effect marginal and predictive surrogate.
It is not:

- a neural implementation of full HMSC;
- an iid or spatial latent-factor HMSC posterior;
- calibrated joint species-association inference;
- support for traits, phylogeny, detection, or structural random effects; or
- near-equivalent to MCMC.

Users requiring those capabilities must use qualified Python MCMC. Unsupported
neural inputs must reject or use an explicitly requested MCMC fallback; they
must never silently receive fixed-effect neural posterior semantics.

## Branch Decision

Close `feature/generative-neural-hmsc` for model-family development. Preserve
its commits, frozen artifacts, and negative evidence. Do not modify either
qualified neural baseline in place, do not retry v1 or v2, and do not open
their retired seed blocks.

This audit supersedes the new-branch recommendation in
`docs/neural_hmsc_branch_closure_audit_2026-07-27.md`: that recommendation was
executed by this branch, and the resulting two generative candidates failed.
The earlier fixed-effect qualification evidence remains valid.

## Next Decision, Not Yet Authorized

The next roadmap step is a bounded no-seed go/no-go review for a fundamentally
different generative approach. It must not begin implementation or allocate
seeds. A new branch and preregistration are justified only if a design can
address both observed failure modes:

1. represent identifiable structural targets without collapsing to the
   no-latent model;
2. avoid the numerically brittle all-state orbit-IWAE/refinement path;
3. demonstrate finite mixed-shape optimization on ordinary stress fixtures;
4. define direct association, marginal, joint, predictive, and MCMC gates;
5. use fresh train, validation, and reserved seed ledgers; and
6. retain a one-redesign terminal stop rule and explicit non-equivalence claim.

If no design satisfies those conditions on paper, the generative neural HMSC
research path should remain closed and the project should focus on Python MCMC
plus the two qualified fixed-effect neural surrogates.
