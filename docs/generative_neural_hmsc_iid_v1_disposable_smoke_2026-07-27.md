# Generative Neural-HMSC iid v1 Disposable Smoke

Date: 2026-07-27

Branch: `feature/generative-neural-hmsc`

Protocol: `generative_neural_hmsc_iid_probit_v1`

## Decision

The authorized 591M-592M disposable smoke passed its plumbing, optimization,
artifact-integrity, exact-target, and seed-seal checks. This result permits the
implementation to advance to a clean-source production authorization review.
It does not qualify the model statistically and does not authorize any
501M-515M seed block.

The accepted disposable artifact is:

`/private/tmp/generative_neural_hmsc_iid_v1_disposable_20260727_retry1`

An earlier artifact at
`/private/tmp/generative_neural_hmsc_iid_v1_disposable_20260727` is
superseded. Inspection of that first run showed that its manifest recorded the
base Git commit but did not identify the uncommitted implementation bytes.
Before rerunning, the artifact schema and independent validator were changed
to require the exact source-file inventory, source hashes, branch, worktree
state, and runtime versions.

## Authorized Seed Use

The disposable confirmation was:

`OPEN_GENERATIVE_IID_DISPOSABLE_SMOKE=GENERATE_591M_592M_DISPOSABLE_ONLY`

The accepted run used exactly:

- training: 18 contexts, seeds 591000001-591000018
- validation: 18 contexts, seeds 592000001-592000018
- epochs: 2
- evidence role: `plumbing_and_optimization_only`

The independent validator confirmed:

- `production_seed_ranges_opened = false`
- `reserved_seed_ranges_opened = false`
- no 501M-515M seed was opened

## Independent Validation

The independent validation status is
`independent_disposable_validation_passed`.

| Check | Result |
| --- | ---: |
| Final training loss | 8776.780603027344 |
| Recomputed validation loss | 5716.5771484375 |
| Recomputed validation IWELBO | -5716.5771484375 |
| Recomputed exact truth log joint | -175.4917007215717 |
| Maximum weight change from seeded initialization | 0.0017552822828292847 |
| All stored smoke metrics finite | true |
| Checkpoint roundtrip and compatibility validation | pass |
| Freeze-to-report hash validation | pass |
| Exact disposable seed-range validation | pass |

The nonzero maximum weight change demonstrates that optimization updated the
seeded model. The independently recomputed validation objective and exact
truth log joint match the stored run evidence. These checks establish
execution and artifact integrity only; their absolute values are not
qualification gates.

## Frozen Artifact Hashes

| File or content | SHA-256 |
| --- | --- |
| Checkpoint content | `e827df53e27b239f082166c5760f7e39625ce464bdd9d94961ec499737ce5609` |
| `checkpoint/generative_iid_checkpoint.json` | `0925051235431cbceb1d2d5f024243d3cfdf4a1ec45d08a9a6d8cd6ebb983fd4` |
| `checkpoint/weights.weights.h5` | `13cde23b67528567bb7207754ae0d2b27833293cb2bc60abe535ba33ff762939` |
| `disposable_smoke_report.json` | `0f9e38831ae9f6dd085bf34f598506a09fd2786d4086cc7faafc24a552c03cf9` |
| `freeze.json` | `a5fa96cc517dd189bc4754dbee2497929f77168166a7febf8768e197f06e52ef` |
| `independent_validation.json` | `63bf6a51076e6000beeca4e7f4d9a5ab9ab266980c19ae89da2add6d11525a48` |

The checkpoint records commit
`fa6b5e03846ba4de9655e28ce68cf537c7d59063`, branch
`feature/generative-neural-hmsc`, and `worktree_dirty = true`. The exact
implementation, harness, preregistration, seed-audit, and design-review file
hashes are embedded in its `source_provenance` inventory. The runtime inventory
records Python 3.12.0, TensorFlow 2.21.0, TensorFlow Probability 0.25.0, NumPy
1.26.4, and macOS arm64.

Recording a dirty worktree is sufficient to identify this disposable artifact
exactly, but it is not an acceptable production freeze. Production training
must be pinned to a clean commit containing the reviewed implementation and
this disposable evidence.

## Scope

This smoke establishes that the frozen simulator, tensor path, generative
posterior, IWAE training loop, exact-model target check, checkpoint schema, and
sealed harness execute together on all 18 preregistered disposable factorial
cells.

It does not establish:

- posterior calibration or SBC rank behavior
- recovery of Beta, Eta, Lambda, or residual species association
- agreement with exact-model MCMC
- predictive superiority over the immutable neural or Python MCMC baselines
- generalization to real ecological data
- readiness to open production or reserved evaluation seeds

## Next Barrier

Freeze the implementation and disposable evidence in a clean source commit,
then implement and review a separate production authorization path for 501M
training and 502M fixed validation. That path must pin the clean commit and all
preregistered document hashes, revalidate seed roles before generation, and
leave 503M-515M sealed. No production seed should be opened as part of that
implementation or review.
