# Generative Neural-HMSC iid v1 502M Timeout

Date: 2026-07-30

Job: `20351142`

Training source: `fc2ac5aff84f2fbed2c3604f3001f3647618fdc0`

Evaluator source: `b3d6cd10b045dd52d5513b80519b04220d614f07`

## Decision

Job `20351142` is an incomplete infrastructure attempt, not a 502M
qualification result and not a candidate failure. It produced no
`fixed_validation_report.json`, `context_metrics.json.gz`, `freeze.json`, or
read-only validation output. No 503M-515M block may open.

The original partial run root must remain immutable. A recovery may reuse only
the already-authorized deterministic 502M seeds and frozen candidate,
ablation, comparator, gate, and threshold definitions. It must not select
between the partial and recovery outputs. The partial exact-MCMC samples are
excluded wholesale from the final decision.

## Scheduler Result

Slurm reported:

- state: `TIMEOUT`;
- elapsed: `1-00:00:11`;
- batch state: `CANCELLED`;
- batch exit code: `0:15`;
- maximum resident memory: `5,847,616 KiB`;
- requested time limit: 24 hours.

LUMI `standard-g` permits at most 48 hours. The first 11 serial exact-MCMC
contexts consumed almost the complete 24-hour allocation, implying roughly
75 hours for all 36 contexts before the Python-HMSC comparator and final
aggregation. Extending the monolithic job cannot provide a reliable bounded
completion.

## Partial Artifacts

Eleven of the preregistered 36 exact-MCMC contexts were written:

`502000001`, `502000005`, `502000013`, `502000017`, `502000055`,
`502000059`, `502000067`, `502000071`, `502000073`, `502000077`, and
`502000085`.

All eleven are members of the frozen subset and pass ZIP archive-integrity
checks. The remaining 25 exact-MCMC contexts, all Python-HMSC comparator
contexts, candidate/ablation metric files, immutable-v0.1 comparison,
aggregate gates, report, and freeze were not written.

The partial files and SHA-256 values are:

- `502000001.npz`: `8875293025fb05af6bbb5b045653e0f142828824ab5a3357ab64dfb930af6b3b`
- `502000005.npz`: `e30041030352cf02f9cc3c4ec5a8ef470e0038994e0f81aecd3dd6274091db99`
- `502000013.npz`: `9190456cae18eb1dcc91dc51d15fcff758c6c4d0330a84c34eb9b93219128a9f`
- `502000017.npz`: `0aa2377386bf936dbef0abfde2a24da97c837217a488fe6abc69c6521d39f07b`
- `502000055.npz`: `4b582b2004acdd25a41d73ca6893ccb3b998aae443be1f9bff3fe7996c27d8c3`
- `502000059.npz`: `a79954a70caa57592d037c82150eea166ba2344818a102e93bedabbe1e2b2317`
- `502000067.npz`: `2079f71fe27398b2a571855b9acfda21489bfe5182132c840098a996b98430b3`
- `502000071.npz`: `ef1edbbce6357d1f57afa2e1a0d3d15409a84647772a0fdf1dc818c72d7bc3ad`
- `502000073.npz`: `5c82848a6716d5830ce9ecb89034ab1ba9ddf94351804b446d15db26ad152d82`
- `502000077.npz`: `15be5ffcfa1faac5ce73381de9c03f90e2a8e9610f82c6f0fc9b21d84a973ed0`
- `502000085.npz`: `e86f02c3ea3523c77896596abae49712c675a9fb776080bf9cb6ea42172e7d45`

## Recovery Boundary

The recovery implementation must:

1. retain the accepted 501M freeze and both checkpoint hashes unchanged;
2. retain evaluator version, simulator, fixed 502M seed ownership, MCMC
   warm-up/draw/chain/continuation rules, Python-HMSC settings, immutable-v0.1
   comparator, all metrics, all gates, and all thresholds unchanged;
3. run the 36 fixed comparator contexts as independently hash-frozen shards;
4. give each shard an exact seed/index ownership manifest and refuse duplicate
   or missing ownership;
5. checkpoint exact-MCMC rows, diagnostics, samples, elapsed time,
   Python-HMSC rows, elapsed time, and complete artifact inventories;
6. aggregate only after all 36 shard manifests independently validate;
7. compute candidate, ablation, v0.1, invariance, and final gates once in a
   separately authorized finalizer;
8. bind the final freeze to every shard hash and both source commits;
9. use a fresh recovery run root and preserve job `20351142` unchanged;
10. keep 503M-515M sealed regardless of recovery progress.

Run only ordinary non-ledger fixtures while implementing and testing the
recovery. Require a new explicit timeout-recovery confirmation before any
502M shard is executed.

## Next Step

Implement and test the sealed sharded 502M recovery harness and scheduler
using ordinary fixtures. Then run a token-free LUMI preflight and explicitly
decide whether to authorize the same-seed 502M timeout recovery. Do not submit
another monolithic evaluator.
