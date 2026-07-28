# Generative Neural-HMSC iid v1 501M Validation

Date: 2026-07-28

Job: `20301852`

Source: `fc2ac5aff84f2fbed2c3604f3001f3647618fdc0`

## Decision

Accept the frozen 501M candidate and no-latent ablation artifacts. Do not
retrain or reopen 501M. The artifacts are eligible for a separate 502M
authorization decision, but this validation does not itself authorize 502M.
Blocks 502M-515M remain sealed.

The machine-readable evidence is
`docs/generative_neural_hmsc_iid_v1_501m_validation_2026-07-28.json.md`.

## Failure Classification

Slurm reported `FAILED` after 11:03:29 with exit code `1:0`. Both 200-epoch
models had already completed, and all checkpoint, corpus, report, and freeze
files had been written.

The failure occurred in the first read-only validation call. The generated
corpus manifest used `fixed_validation_opened`, while the validator required
`fixed_validation_seed_ranges_opened`. Both names represent the same false
seal state. Freeze, report, and checkpoint manifests used the canonical key
and also recorded false.

The validator now accepts either exact key, requires every present alias to be
false, and still rejects missing, true, or conflicting values. Future corpus
manifests use the canonical key. This is a validation-schema correction; it
does not change the simulator, representation, objective, optimizer,
checkpoint weights, comparator, gate, threshold, or seed role.

## Independent Validation

The complete run root was downloaded to an independent local directory.
Validation recomputed and checked:

- `freeze.json` SHA-256 and its sidecar;
- preregistration, seed-audit, and design-review bindings;
- candidate manifest, content, weight hash, size, and exact file set;
- ablation manifest, content, weight hash, size, and exact file set;
- identical clean source provenance for both checkpoints;
- the 324-context, 648-realization corpus and factorial contract;
- every report-to-checkpoint, report-to-corpus, and freeze-to-report binding;
- finite candidate and ablation optimization metrics;
- false 502M, reserved, and redesign seed flags.

Both checkpoints loaded successfully. On ordinary non-ledger seed `881501001`,
all 90 weight tensors per model were finite, posterior means and low-rank
factors were finite, and all diagonal posterior scales were positive.

The corrected focused suite reported `36 passed, 1 skipped`. The generated
`postfreeze_validation.json` has SHA-256
`0f6ac100df4497d7df8636962cf5c67a76dbefcb4915dcd77a4df6446c3c87c6`.

## Frozen Hashes

Candidate:

- content: `d36dd3b23ccdba36041792716b9fb2cb21a437265870e686cdef1f01b9d05e30`
- manifest: `48a6bfb95cc9c93dbf4770aca013a8d552a1d18a1ed5f087667113141aabb45d`
- weights: `43b4eded085b0213f53ffa795e5bf91f367a2dc86cd17a2915da7e404f8043c7`

No-latent ablation:

- content: `691f8c992ec709ac241af32ea0fd7e94e43c3ed9d79c768e01a23a4a1e8193bc`
- manifest: `ba7c809117798559ba5a74cc30881f23fb1feda5856fadef2ae5b56332193a16`
- weights: `1ab01e332b7b23609fb0bdb7a41e978a29c3f237c94eb02c0ab0276bb541232d`

Training evidence:

- freeze: `93f11221c9bbbd3b8ced541888397541ab61f0b88ae23eebc3431e969512ae39`
- corpus: `aeba904e8c047cf2952f1f2a3e61482f2d358f4644ecaa2385282e2b34ae8697`
- report: `07ac63f9295d82c9e1aed5d43a0af89faa580b4ee2bbc8b352e7a52e7646524c`

## Next Barrier

Commit the validator correction and this independent evidence. Then run the
read-only 502M preflight pinned to the hashes above and explicitly decide
whether to authorize the one-shot 502M fixed-validation block. Do not open
502M during this correction or validation step; keep 503M-515M sealed.
