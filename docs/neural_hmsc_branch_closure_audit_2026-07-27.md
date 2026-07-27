# Neural-HMSC Feature Branch Closure Audit

Date: 2026-07-27

## Executive Decision

The branch did not deliver a neural replacement or summary-level
near-equivalent of HMSC/MCMC. It delivered two qualified neural fixed-effect
probit approximations, a qualified predictive-only deployment artifact for two
named ecological datasets, and substantial reusable validation and
Python-HMSC parity infrastructure.

The defensible description of the result is:

> A GPU-accelerated neural fixed-effect probit JSDM surrogate with calibrated
> marginal coefficient uncertainty, bounded predictive deployment, strict
> applicability limits, and Python-MCMC reference behavior.

It must not be described as full Neural HMSC, structural HMSC equivalence, a
joint-posterior approximation to HMSC, or an MCMC replacement.

Model development on this branch is closed. Safety and maintenance work may
continue, but another neural structural attempt requires a new branch, a
genuinely different generative representation, fresh preregistration, and
unused evidence.

## Original Goal

The original research goal was a GPU-accelerated neural joint species
distribution model capable of learning from environmental, species, trait,
spatial, temporal, and community-response information while retaining useful
HMSC structure and uncertainty.

A successful broad path would eventually require:

- variable site, species, and design dimensions;
- trait and phylogeny mediation;
- latent site/species factors and residual association;
- iid and spatial random effects;
- calibrated marginal and joint posterior summaries;
- reliable heldout prediction;
- explicit uncertainty and applicability semantics;
- quantitative comparison against qualified Python MCMC.

The branch qualified only a bounded subset of the fixed-effect portion of that
goal.

## Qualified Outcomes

### Fixed-shape probit v0.1

`neural_hmsc_v0_1` is release-qualified only for:

- fixed-effect probit `Beta`;
- exactly 40 sites, 75 species, and two ordered coefficients;
- the frozen formula and coefficient boundary;
- calibrated marginal coefficient uncertainty;
- repeated amortized inference.

Its three production-shape SBC runs achieved 95% coverage between `0.952865`
and `0.956198`. Whittaker and Big Spatial predictive proper-score ratios
remained within the frozen `1.10` MCMC tolerance, but qualified Python MCMC was
better on every reported real-data Brier and log-loss score.

The release explicitly excludes traits, phylogeny, random effects, latent
associations, spatial effects, detection models, and joint-posterior
equivalence.

### Variable-shape probit v1

`neural_hmsc_variable_probit_v1` is qualified for:

- fixed-effect probit `Beta`;
- exactly three ordered coefficients;
- 12-48 sites;
- 2-10 species.

Three independent runs achieved mean 95% coefficient coverage `0.9512`.
Mean neural/MCMC Brier and log-loss ratios were `0.9997` and `1.0013`; worst
ratios were `1.0339` and `1.0297`.

This result supports bounded variable site/species inference. It does not
support variable design-column counts, ecology-scale species counts, traits,
associations, spatial effects, or joint-posterior equivalence.

### Predictive-only deployment

The manifest-backed affine ensemble is qualified only as a response-scale
predictive artifact for the frozen Whittaker and Big Spatial workflows. It
passed full and leave-one-seed-out no-degradation checks against the matched
neural scale-only ensemble.

It is not a posterior calibration, HMSC structural result, or evidence of
predictive superiority over MCMC. Neural/MCMC Brier and log-loss ratios were
approximately `1.0213`/`1.0327` on Whittaker and `1.0790`/`1.0731` on Big
Spatial.

## Unqualified Or Failed Capabilities

| Capability | Result | Consequence |
| --- | --- | --- |
| Gaussian and Poisson public neural paths | implemented but not release-qualified | remain experimental |
| Trait-mediated Gamma | terminal Milestone 53A failure | Python MCMC is the only qualified trait-Gamma path |
| iid Eta/Lambda structural family | not opened after Gamma prerequisite failed | unsupported |
| Variable design-column and ecology-scale shape | terminal Milestone 54 v2.1 failure | variable-v1 remains the endpoint |
| Learned within-species covariance overlay | terminal Milestone 56 failure | no covariance or joint-posterior claim |
| Joint bivariate Student-t posterior | terminal Milestone 57 failure | no reserved MCMC or real-data evaluation |
| Spatial neural random effects | prototype only; never structurally qualified | unsupported |
| Cross-species latent association posterior | not qualified | unsupported |
| Full or summary-level MCMC equivalence | not achieved | must not be claimed |

The final Student-t candidate passed 173 of 176 fixed-validation gates and
improved location and proper scores substantially, but failed the frozen width,
tail-saturation, and strong-effect radial-rank gates. The stop rule correctly
prevented a favorable aggregate score from hiding a joint-calibration defect.

## Why The Broad Neural-HMSC Goal Was Not Reached

### Representation drift

Development converged on amortized `Beta` estimation around
IRLS/Laplace-derived anchors. This is useful for fixed-effect approximation,
but it is not the HMSC generative structure.

### Missing structural variables

The qualified models do not jointly represent latent site factors, species
loadings, trait-mediated coefficients, phylogenetic dependence, study-design
random levels, or spatial random effects. These are central HMSC quantities,
not optional calibration layers.

### Local correction was asked to solve global posterior structure

Scalar scales, monotone calibration, response-affine corrections, covariance
heads, and the Student-t head adjusted local posterior summaries. They could
not create structural dependence absent from the underlying representation.

### Training targets were narrower than the desired claim

Known simulated coefficients and response proper scores can qualify marginal
and predictive behavior. They do not by themselves identify a full structured
posterior over HMSC parameters and latent states.

### Transfer and support remained representation-dependent

Repeated failures concentrated in rare prevalence, low design support,
variable design dimensions, tail behavior, and joint ranks. Selector and
calibration changes moved these failures but did not eliminate their
architectural cause.

## Reusable Assets

The following work should be retained:

- qualified immutable `neural_hmsc_v0_1` and
  `neural_hmsc_variable_probit_v1` baselines;
- compiled-artifact compatibility checks and explicit rejection behavior;
- posterior HDF5 adapters and `HmscFit` interoperability;
- fixed, variable-shape, trait, iid, and spatial simulation scaffolding;
- deterministic seed ledgers and one-shot authorization barriers;
- SBC, rank, coverage, OOD, proper-score, and stratum diagnostics;
- Whittaker and Big Spatial frozen real-data workflows;
- direct R/Python HMSC boundary and preprocessing parity fixtures;
- qualified Python-MCMC comparison path;
- immutable artifact manifests, hash inventories, and release registries;
- LUMI train/evaluate/freeze/post-freeze harness patterns;
- negative-result artifacts and stop-rule discipline.

These assets materially reduce the cost of evaluating a new representation.
They do not make the current model structurally complete.

## Retired Research Families

The following families must not receive further tuning on this branch:

- trait-Gamma v1 and its calibration variants;
- variable-design v1/v2.1;
- M56 correlation-only covariance overlay;
- M57 bivariate Student-t overlay;
- globally or conditionally applied post-hoc OOD/effect/combined-shift heads
  that failed their frozen gates;
- source-only predictive-mean calibration variants that failed cross-dataset
  transfer gates;
- MCMC-teacher residual variants that failed compact or real-context support.

The promoted predictive affine ensemble and external monotone scale
calibration remain frozen deployment artifacts. Their qualification does not
authorize further posterior or structural claims.

## Branch Disposition And Claim Boundary

This branch should be retained as the implementation history and evidence for:

`neural fixed-effect JSDM approximation and validation infrastructure`

It should be closed for new neural model-family development after:

1. retaining the two qualified neural baselines;
2. retaining the predictive-only deployment artifact;
3. retaining terminal negative artifacts and reports;
4. adding an explicit applicability/fallback policy;
5. ensuring public documentation uses the narrowed claim.

The public name `Neural-HMSC` may remain as an experimental project namespace,
but every user-facing claim must include the fixed-effect probit boundary.
Unqualified inputs must reject or use an explicitly requested Python-MCMC
fallback; they must never silently receive neural posterior semantics.

## Decision On A New Structured Branch

A new branch is justified only if the project still intends to pursue the
original research goal and accepts that this is a new model, not Milestone 57
repair.

Recommended branch scope:

`feature/generative-neural-hmsc`

The first candidate should be a structured fixed-effect-plus-iid-latent-factor
probit model, because latent site/species factors directly address residual
species association. Traits, phylogeny, and spatial effects should remain
sealed until that first structural family qualifies.

The representation must be different in kind:

- an explicit HMSC-like generative likelihood;
- exchangeable site and species encoders;
- learned latent site factors and species loadings;
- a posterior family that owns fixed effects, latent factors, loadings, and
  their dependence together;
- variable site/species support by construction;
- simulation-based neural posterior estimation or variational inference
  trained end to end;
- no frozen v0.1 posterior as the candidate mean/scale anchor;
- no post-hoc scalar calibration as the primary uncertainty mechanism.

Normalizing flows, structured variational families, or conditional neural
posterior estimators are plausible implementation choices. The architecture
choice must be made before seed allocation and must be justified by the
posterior factorization it can represent.

## Entry Gate For New Work

Do not open the new branch until a short design preregistration fixes:

- the exact generative model and posterior factorization;
- the first supported structural family;
- parameter identifiability constraints;
- simulation factorial and prior-predictive checks;
- unused train, validation, and reserved seed ledgers;
- marginal, joint, association, predictive, and MCMC comparison gates;
- real-data boundary and target-outcome prohibition;
- runtime budget and one-redesign stop rule;
- the claim that success would and would not support.

At minimum, the first structural candidate must beat fixed-effect neural
baselines on association-aware simulation metrics, remain within fixed MCMC
proper-score tolerances, pass joint calibration diagnostics, and recover
latent association direction where simulation truth is known.

## Outcome Of Following The Revised Roadmap

Completing applicability and fallback work on the current branch produces a
safe, useful fixed-effect neural surrogate. It does not produce near-MCMC HMSC
equivalence.

Completing a successful new structured branch would produce evidence for one
bounded neural structural HMSC family, initially fixed effects plus iid latent
association. It still would not establish full HMSC or universal MCMC
equivalence. Traits, phylogeny, spatial effects, detection, and broader
likelihoods would each require separate qualification.

The immediate next step is to freeze this audit in the roadmap and prepare the
new-branch design preregistration. Applicability/fallback implementation should
then be treated as maintenance of the current qualified product, not as
completion of the original Neural-HMSC research claim.
