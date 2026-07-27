# Generative Neural-HMSC iid v1 Production Authorization Review

Date: 2026-07-27

Branch: `feature/generative-neural-hmsc`

Protocol: `generative_neural_hmsc_iid_probit_v1`

## Decision

The implementation and disposable evidence were frozen in clean baseline
commit `c90aab13806248d9bc339ba921b30201ba870d81`.

A separate production harness and LUMI wrapper now implement the 501M
candidate-training boundary. They have not opened a production seed. The
current production-harness changes must receive their own clean commit before
501M can be authorized.

The 502M fixed-validation block remains sealed and deliberately
non-executable. The preregistration requires a complete multi-comparator
qualification, not a candidate validation-loss calculation. Opening 502M
before that evaluator is complete would consume the one-shot block without
being able to reach a valid decision.

## 501M Training Contract

The new production command is:

```text
examples/run_generative_neural_hmsc_iid_v1_production.py train-candidate
```

It requires:

- environment variable
  `OPEN_GENERATIVE_IID_501M_TRAINING=GENERATE_501M_CANDIDATE_TRAINING_ONLY`;
- an explicit full `--expected-source-commit`;
- repository `HEAD` equal to that commit;
- a completely clean Git worktree;
- unchanged preregistration, seed-audit, and design-review hashes;
- a new empty output root.

Only after those checks may it generate 501000001-501000324. The block is the
exact 324-owner factorial:

- sites: 24, 40, 96;
- species: 12, 36, 75;
- covariate shapes: normal, right-skewed;
- loading strata: weak, medium, strong;
- prevalence strata: rare, moderate, common;
- two owning-context replicates per cell.

Each owner produces response realizations zero and one from identical X and
Theta, giving 648 training realizations. Training is fixed to:

- model seed 501900001;
- 200 epochs;
- batch size four;
- eight IWAE samples;
- the preregistered AdamW, cosine schedule, warm-up, weight decay, and
  gradient clipping implemented by the frozen model code;
- final-epoch weights, without validation selection.

The output binds:

- exact clean source commit and source-file hashes;
- runtime environment;
- training corpus metadata hashes;
- checkpoint manifest, content, and weight hashes;
- finite final optimization diagnostics;
- explicit false flags for 502M, 503M-505M, and 511M-515M.

`validate-training` independently checks these bindings without regenerating
any simulation.

## Scheduler Boundary

The LUMI wrapper is:

`docs/lumi_generative_neural_hmsc_iid_v1_training_sbatch.sh`

It uses `dev-g`, requests one GPU and 24 hours, validates the full expected
source commit and frozen document hashes, refuses a dirty repository or reused
run root, and repeats read-only post-freeze validation after training.

No job was submitted in this step.

## 502M Boundary

`preflight-fixed-validation` validates a completed 501M freeze and an explicit
checkpoint content hash without reading a 502M seed. It also requires the 502M
confirmation variable to remain unset during preflight.

The harness intentionally has no executable 502M evaluation command yet.
Even an exact
`OPEN_GENERATIVE_IID_502M_FIXED_VALIDATION=EVALUATE_502M_FIXED_VALIDATION_ONCE`
token reaches a hard refusal before simulation generation.

The following preregistered components must be implemented, tested, reviewed,
and hash-frozen first:

1. candidate 256-draw posterior diagnostics;
2. fixed no-latent ablation trained on the same 501M corpus;
3. exact-model MCMC on the fixed 36-context subset;
4. qualified Python HMSC-HPC on that subset;
5. immutable `neural_hmsc_v0_1` on the matched cell;
6. permutation and padding invariance;
7. masked-cell and new-site proper scores;
8. posterior-predictive richness and prevalence intervals;
9. runtime and peak-device-memory measurements;
10. every preregistered aggregate and stratum gate.

This list is machine-readable in the production harness.

## Verification

The focused generative suite reports `19 passed, 1 skipped`. The skipped test
is the optional exact-MCMC execution test when its runtime dependency is not
available. Python bytecode compilation and Bash syntax validation pass.

The no-seed seal command reports:

- candidate 501M training opened: false;
- fixed 502M validation opened: false;
- reserved 503M-505M opened: false;
- redesign 511M-515M opened: false.

An explicit 501M-token dry run against the currently dirty implementation was
rejected by the clean-worktree check before its output root was created. No
production simulation was generated.

## Next Barrier

Commit the production harness, scheduler wrapper, tests, review, and roadmap
update. Then implement the complete 502M comparator evaluator using only
ordinary non-ledger test fixtures. Do not authorize 501M training until that
evaluator and its gate report are complete and reviewed; this avoids freezing
a costly checkpoint before the one-shot validation decision is executable.
