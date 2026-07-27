# Neural-HMSC Post-M53A Scope Decision

Date: 2026-07-22

## Decision

Retain qualified Python MCMC as the only trait-Gamma path and close neural
trait-Gamma v1. Do not begin iid Eta/Lambda qualification. Resume neural
development only inside the already-qualified fixed-effect probit family.

The next bounded milestone is variable-design fixed-effect probit v2. It will
replace the current variable-shape model's fixed three-column design with a
coefficient-wise masked representation while preserving the two immutable
regression baselines:

- `neural_hmsc_v0_1`, content SHA-256
  `affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8`;
- `neural_hmsc_variable_probit_v1`, content SHA-256
  `badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9`.

## Why This Scope

The qualified v1 variable-shape path already has strong coefficient
calibration and MCMC proper-score evidence, but accepts exactly
`Intercept, x1, x2` and only 2-10 species. That is now the clearest usability
constraint. Whittaker and Big Spatial both use 40 sites, 75 species, and two
design columns, so they cannot use the promoted variable-shape checkpoint even
though they remain within the same fixed-effect probit statistical family.

Trait-Gamma failed a genuinely independent preregistered gate after its allowed
representation redesign and calibration-only requalification. Revisiting its
scale, threshold, or seeds would invalidate the stop rule. Starting iid latent
effects before qualifying Gamma would also discard the agreed structural
ordering. Gaussian and Poisson are separate likelihood-qualification projects
and do not address the present fixed-effect usability constraint.

## Milestone 54 Boundary

Build one target-agnostic fixed-effect probit checkpoint supporting:

- 12-128 sites;
- 2-100 species;
- 2-8 ordered numerical design columns, including one leading intercept;
- variable compiled covariate names and formula provenance;
- no traits, phylogeny, random effects, spatial effects, detection model, or
  target-outcome-conditioned selection.

The representation must use site, species, and covariate masks; a shared
coefficient head; coefficient-local sufficient statistics; and a probit
IRLS/Laplace anchor. It must be invariant to site order and equivariant to
species order and non-intercept covariate order. Unsupported or poorly
identified designs must be rejected explicitly rather than silently projected
into the training domain.

## Evidence Contract

Training, coefficient-posterior calibration, and fixed evaluation use disjoint
seed blocks. Evidence spans all dimensional boundaries and strata for
prevalence, coefficient magnitude, and design conditioning. One candidate and
two independent sensitivities must all pass:

- exact checkpoint and public-API parity;
- padding and mask parity;
- site/species/covariate permutation properties;
- overall 95% coefficient coverage in `[0.925, 0.975]`;
- normalized rank mean and variance errors no greater than `0.025`;
- no posterior-mean degradation versus the IRLS/Laplace anchor;
- held-out Brier and log-loss ratios no greater than `1.10` versus qualified
  Python MCMC;
- explicit non-failing coverage/rank reports by covariate count, coefficient
  role, dimensional boundary, and design-information stratum.

Whittaker and Big Spatial are opened only after simulated qualification. They
remain evaluation-only and may not choose parameters. Both must stay within
`1.10` of qualified MCMC and within `0.02` of the applicable frozen neural
baseline on each proper-score ratio.

## Outcomes

Passing freezes a separate `neural_hmsc_variable_design_probit_v2` baseline and
retains variable-probit v1 as fallback. This expands the usable fixed-effect
domain; it does not establish a joint posterior or structural HMSC equivalent.

Failure after one representation-level correction and one fresh independent
evaluation closes v2, leaves v1 as the qualified fixed-effect endpoint, and
does not start another calibration search.

## Immediate Next Step

The variable-design tensor/model skeleton and deterministic tests are complete.
The new path has `covariate_mask`, coefficient-local features, a shared
coefficient head, nonzero-head padding/permutation parity, compatibility and
checkpoint-hash rejection, and exact local hash regression for both existing
baselines. It is explicitly untrained and remains experimental.

The fixed qualification harness is preregistered and implemented under protocol
`neural_hmsc_variable_design_m54_v1_1`. Its disposable 91M-93M smoke passed and
left the untouched 101M-109M production blocks unopened. The initial 61M-69M
draft blocks are retired because a unit test generated one training block while
checking balance; no model training, calibration, or evaluation occurred.

Next, run only candidate `train-calibrate` on 101M/102M with the exact
confirmation token. Validate that freeze before separately authorizing the 103M
reserved evaluation block.
