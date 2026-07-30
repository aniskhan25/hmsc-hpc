# Generative Neural-HMSC iid v1: 502M finalizer authorization

Date: 2026-07-30

## Completed shard execution

LUMI recovery array job `20430754` completed all 36 one-seed comparator
shards with exit code `0:0`. Individual task runtimes ranged from 1:12:02 to
4:28:24. The frozen evaluator remained
`d32093b367e0af40a9d9bd583d0812419b83667f`.

## Independent validation

A standalone validator, separate from the recovery harness, re-hashed every
shard freeze, result, exact-MCMC artifact, and Python-HMSC artifact. Its
SHA-256 is
`56c3d9db25c467e76e4d901472c6090f2866892f1e7a7e875ea7814036031dcc`.

Token-free LUMI validation job `20461335` completed on `dev-g` in 2:57 with
exit code `0:0`. The validation evidence is
`generative_neural_hmsc_iid_v1_502m_recovery_shards_d32093b_validation_v2.json`,
SHA-256
`b19c1b5e45cdc27b9e7cc41bacdbef07af15e35b22125c908085ee5cf2a5b623`.

Validated inventory:

- shard count: 36;
- exact-MCMC files: 36, totaling 312,291,181 bytes;
- Python-HMSC files: 108, totaling 466,943,120 bytes;
- shard-binding SHA-256:
  `4d2ec028f3eef8d60e75d40fa91d8f47fbbebd8ae52ac2b42d21cb263e50df98`;
- partial attempt reused: false;
- 503M-505M reserved blocks opened: false;
- 511M-515M redesign blocks opened: false.

The first CPU validation attempt, job `20460640`, stopped at Python parsing
because LUMI's system Python did not support the future-annotations import. It
created only a zero-byte output placeholder and did not inspect a shard.
Pending duplicate `20460874` was cancelled. Job `20461323` then stopped at its
no-overwrite check on that zero-byte path. The successful validator used a new
immutable output path; none of these infrastructure attempts changed a shard.

## Authorization decision

The complete 36-of-36 evidence satisfies the frozen recovery boundary.
The one-shot finalizer is therefore separately authorized with:

`OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY_FINALIZER=FINALIZE_502M_TIMEOUT_RECOVERY_ONCE`

LUMI job `20461616` was submitted on `dev-g` using the same evaluator,
training freeze, candidate, ablation, shard root, release registry, metrics,
gates, and thresholds. The shard authorization token was unset. Blocks
503M-515M remain sealed.

## Next step

Monitor job `20461616`. On completion, independently validate `freeze.json`,
`fixed_validation_report.json`, `context_metrics.json.gz`, reconstructed
comparator inventories, shard bindings, immutable v0.1 provenance, and every
unchanged 502M gate. Open no later seed unless the complete final report passes.
