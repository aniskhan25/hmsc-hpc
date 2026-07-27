# Neural-HMSC Milestone 54 v2.1 Production Freeze

Date: 2026-07-22

Status: production train/auxiliary/calibration freeze validated; reserved 115M
evaluation remains sealed.

## Authorized Run

Exact authorization:
`GENERATE_M54_V2_1_TRAIN_AUX_CALIBRATION`

LUMI dev-g job `20144482` completed successfully in `00:08:08` with exit code
`0`. The harness training timer was `409.900557` seconds. The run root is:

`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_m54_v2_1_train_calibration_20144482`

The run opened only the preregistered production roles:

- 111000001-111000243: coefficient-posterior training
- 112000001-112000243: predictive auxiliary contexts
- 113000001-113000243: independent predictive heldout partners
- 114000001-114000243: coefficient calibration
- model initialization: `111900001`

The reserved 115000001-115000243 evaluation block was not generated or
opened. No evaluation command is present in the submitted sbatch path.

## Validation

The job-local post-freeze validator and an independent local replay of
`validate_freeze()` both passed. Validation confirmed:

- exact seed roles and disjointness;
- no intersection between opened roles and 115M;
- 81 factorial cells with exactly three communities per cell;
- balanced site, species, covariate, design-condition, prevalence, and effect
  marginals for training, auxiliary-context, and calibration corpora;
- independent paired heldout designs with shared coefficient truth;
- checkpoint schema `0.2` load and calibration roundtrip;
- exact immutable fixed-shape v0.1 and variable-design v1 hashes;
- exact frozen preregistration hash;
- `reserved_evaluation_opened=false`.

The freeze and validation hashes are:

- freeze report:
  `bb32afd655db277064c5c6fcbdf53e2d89a9f42c24a0690c50a494967f46d816`
- post-freeze validation:
  `81a1126bf28662ace32b0e0f30869e3b56a5acbf58b2c6b48bd6ddd70f775d53`
- checkpoint manifest:
  `20938b1cf4a55ebde79adee092cd27af999113b94d208753b17d95a16a50b678`
- checkpoint weights:
  `788d801777a12844a5936856e933914c322d35e708ea47487d96557dd1621f7c`
- coefficient calibration:
  `d680781064cea1f040b0934b957564117bdd629f3815d9478a322c52f1c580ac`
- preregistration:
  `900af8719fc73947cd7addf3b7dc9fe2f233eadbbd2bf9f37bac1286fc15e54d`

Immutable baseline hashes remained:

- fixed-shape v0.1:
  `affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8`
- variable-design v1:
  `badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9`

## Frozen Training Summary

The final epoch recorded total loss `0.388648`, Beta RMSE `0.297158`,
predictive Brier `0.155632`, predictive log loss `0.475887`, and mean support
gate `0.819710`. The independent 114M coefficient calibration contains 49,410
coefficients and froze a scalar Beta-scale multiplier of `0.863175` for 95%
coefficient-posterior coverage semantics.

These training diagnostics are not qualification evidence. Only the sealed
115M evaluation can determine whether the preregistered aggregate, stratum,
proper-score, genuine-improvement, support-ordering, and MCMC comparison gates
pass.

## Resolved Sealed Action

The separately authorized one-shot evaluation completed as LUMI job
`20179655`. The wrapper pinned this freeze SHA-256 and revalidated the
checkpoint, calibration, manifest, seed roles, immutable baselines, and
preregistration before opening 115M. The immutable result was
`variable_design_v2_1_terminal_failure`; see
`docs/neural_hmsc_m54_v2_1_terminal_result_2026-07-23.md`. Milestone 54 is
closed, and `neural_hmsc_variable_probit_v1` remains the qualified endpoint.
