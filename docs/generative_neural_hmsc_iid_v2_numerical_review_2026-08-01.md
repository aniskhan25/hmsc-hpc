# Generative Neural-HMSC IID v2 Bounded Numerical Review

Date: 2026-08-01

Protocol: `generative_neural_hmsc_iid_probit_v2_orbit`

Decision: `implementation_repair_accepted_keep_511m_515m_sealed`

## Scope

This review followed disposable retry job `20518775`, which generated the
authorized corpus but failed with `FloatingPointError: non-finite v2 gradient`
after repeated batched Cholesky failures. This review opened no simulation,
training, validation, or evaluation ledger seed. It used only ordinary fixture
seeds `983001-983018` and the already frozen model-initialization seed.

No posterior representation, density, IWAE objective, refinement step,
architecture, optimization schedule, gate, threshold, simulator, or seed role
was changed.

## Root Cause

The prior feasibility suite had a coverage gap. It tested:

- dense posterior-density parity;
- analytic orbit-density parity;
- one-community end-to-end gradients; and
- maximum-shape forward sampling and density.

It did not test two-epoch training over a padded mixed-shape batch spanning the
full 18-cell factorial. That mixed workload repeatedly evaluates rank-16
Woodbury systems during refinement.

The failure was not a general MI250X or TensorFlow failure. On ordinary data,
the unchanged CPU path was finite, and LUMI TensorFlow 2.16 completed training.
However, LUMI's GPU Cholesky emitted 16 rejected decompositions for both
float32 and explicitly symmetric float64 GPU factorization. Moving only the
small symmetric rank-16 factorization to float64 CPU arithmetic eliminated all
warnings. The CPU and GPU float64 modes returned identical losses, IWELBOs, and
gradient norms on the accepted ordinary run.

Thus the bounded diagnosis is a data/conditioning-sensitive ROCm batched
Cholesky implementation defect. The represented covariance remains

```text
diag(exp(2 log_scale)) + factor factor'
```

and the Woodbury density is unchanged.

## Repair

`pyhmsc/neural/generative_iid_v2.py` now:

1. forms the rank-16 Woodbury system in float64;
2. symmetrizes it as `(A + A') / 2`, an exact no-op in real arithmetic;
3. computes only its Cholesky factor on CPU; and
4. returns inverse, factor, Cholesky, and log-determinant terms in the model
   dtype.

No jitter, clipping, fallback, calibration, altered loss, or altered posterior
parameterization was introduced.

Repaired source SHA-256:

```text
87828857ee1718a8825a1a15e7af99abe49a86ee4d179f6cbce6591162aa71bc  pyhmsc/neural/generative_iid_v2.py
```

The no-ledger diagnostic and corrected scheduler are pinned at:

```text
e3ca2876bdad56ce6c42d45720f88435742567e8977c292c530316cdd32e973b  examples/diagnose_generative_neural_hmsc_iid_v2_numerics.py
860f1e088a00a86e16cc4b1f09048773427dd68609a44efb2a72616b22fc1632  docs/lumi_generative_neural_hmsc_iid_v2_numerical_review_sbatch.sh
```

## Evidence

LUMI job `20519145` failed before ordinary simulation because its first
external-script launcher resolved the installed `pyhmsc`; it opened no seed.
The corrected `runpy` launcher was used thereafter.

LUMI job `20519231` compared the original source on the ordinary corpus:

| Mode | Finite | Failed-Cholesky warnings |
|---|---:|---:|
| frozen float32 GPU | yes | 16 |
| symmetric float64 GPU | yes | 16 |
| symmetric float64 CPU | yes | 0 |

LUMI job `20519352` validated the actual repaired source in an isolated tree:

| Metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| loss | 16993.924609375 | 17043.474609375 |
| IWELBO | -16993.924609375 | -17043.474609375 |
| gradient norm | 293723.29609375 | 367159.28046875 |

The repaired frozen mode reported `all_finite = true`,
`ledger_seeds_opened = false`, and zero failed-Cholesky warnings. Its metrics
were exactly equal to both explicit float64 comparison modes.

Downloaded result hashes include:

```text
0ebba99ef9876e53ea98a74b87036b2140fcb7f505b19acd3751b9ab05aa61a2  frozen.json
cd67dac15bc939821d8aca974ce8ea6d1d5c9d017cf1e64f4888b123e5b5a0bc  frozen.err
87de074540f375f4ecdbe102a08a30b7626469c0ad135470ce12e038d76fe2a0  exit_codes.txt
```

The complete local v2 implementation, harness, authorization, numerical, and
slow maximum-shape/mixed-training suite passed `29/29` tests.

## Decision Boundary

The numerical defect is repairable without changing the frozen statistical
candidate, so the generative iid v2 family is not closed by this review.
However, the repair does not retroactively pass the disposable smoke and does
not authorize another run. Blocks 511M-515M remain sealed.

Before another disposable decision, freeze this repair and evidence in a clean
commit, update the candidate source inventory used by the sealed harness, and
rerun token-free/no-seed preflight. A later 593M-594M verification requires a
new explicit authorization. No production opening may precede a complete
disposable artifact pass.
