# Generative Neural-HMSC iid v1: 502M timeout-recovery authorization

Date: 2026-07-30

## Frozen implementation

- Evaluator commit:
  `d32093b367e0af40a9d9bd583d0812419b83667f`
- Recovery evaluator:
  `generative_iid_v1_502_sharded_recovery_v1`
- Training source:
  `fc2ac5aff84f2fbed2c3604f3001f3647618fdc0`
- Training freeze:
  `93f11221c9bbbd3b8ced541888397541ab61f0b88ae23eebc3431e969512ae39`
- Candidate content:
  `d36dd3b23ccdba36041792716b9fb2cb21a437265870e686cdef1f01b9d05e30`
- No-latent ablation content:
  `691f8c992ec709ac241af32ea0fd7e94e43c3ed9d79c768e01a23a4a1e8193bc`
- Timeout report:
  `a64884b09da18ae85b7076682949fcd09f65cb312502621be24b0b50b190ac89`

## Validation

The ordinary-fixture recovery tests and unchanged fixed-gate tests passed with
`49 passed, 1 skipped`. Python compilation, scheduler shell syntax, and
`git diff --check` passed before the evaluator commit was created.

Token-free LUMI preflight job `20430583` completed on `dev-g` in 31 seconds
with exit code `0:0`. Its evidence artifact has SHA-256
`8d07dfdcbf37a14ecfc1ebbcf3e45cecd8f1375662e5da22ab1de0c021ce17f3`.
It validated all frozen source and checkpoint bindings, all 36 unchanged
502M shard owners, the immutable timeout-report hash, and the 11-file partial
inventory. The partial attempt is excluded wholesale. Every 502M-515M opening
flag remained false during preflight.

## Authorization decision

The same-seed 502M comparator-shard recovery is explicitly authorized with:

`OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY=RECOVER_502M_TIMEOUT_SHARDS_ONLY`

LUMI array job `20430754` was submitted on `standard-g` with indices `0-35`
and concurrency limit 12. The output root is
`generative_neural_hmsc_iid_v1_502m_recovery_shards_d32093b`.

This authorization applies only to the 36 exact-MCMC/Python-HMSC comparator
shards. The finalizer token remains unset and is not authorized by this
record. Blocks 503M-515M remain sealed.

## Next step

Monitor all 36 array tasks. After they finish, independently validate exact
seed ownership, both comparator inventories, all shard result/freeze hashes,
and absence of partial-attempt reuse. Only a complete 36-of-36 validation may
support a separate finalizer authorization.
