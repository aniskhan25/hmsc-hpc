# Generative Neural-HMSC iid v1 502M Evaluator Review

Date: 2026-07-27

Branch: `feature/generative-neural-hmsc`

Protocol: `generative_neural_hmsc_iid_probit_v1`

Evaluator: `generative_iid_v1_502_evaluator_v1`

## Decision

The complete fixed-validation evaluator and its separate authorization path
are implemented. The implementation uses only ordinary non-ledger fixtures in
local tests. No 501M-515M seed was generated or opened.

The evaluator is ready to be committed and hash-frozen. Production training
must still be separately authorized before 502M can be considered.

## Training Freeze Extension

The 501M workflow now trains and freezes two independent final-epoch models
from the same 324 owning contexts and 648 response realizations:

1. the generative iid candidate using the frozen eight-sample IWAE objective;
2. a same-architecture R=0 likelihood ablation.

The ablation retains the encoder and joint posterior head, but Eta@Lambda is
set to zero in its response likelihood. Eta and Lambda remain
prior-regularized nuisance outputs and cannot explain Y. It is trained with
the same 200 epochs, batch size four, optimizer schedule, warm-up, and model
seed as the candidate. Its checkpoint, content hash, weight hash, training
role, and clean-source provenance are independently frozen.

The production training scheduler now uses `standard-g`. Training two
200-epoch models over 648 realizations is not a short `dev-g` workload.

## Candidate Metrics

For every one of the 324 fixed-validation contexts, the evaluator emits 256
joint posterior draws and computes:

- 50%, 80%, 90%, and 95% marginal intervals;
- Beta, R, alpha, and log(tau) truth coverage;
- normalized SBC rank mean and variance;
- 16 protocol-hash-owned Beta projections;
- 16 protocol-hash-owned R projections;
- 16 protocol-hash-owned off-diagonal C projections;
- invariant 48-dimensional energy-score draws;
- posterior-mean association correlation and RMSE;
- random-effect RMSE;
- masked-cell Brier score and log loss;
- deterministic new-site Brier score and log loss with fresh prior Eta;
- site-richness and species-prevalence posterior-predictive coverage;
- inference time, peak device memory, low-rank condition bound, and posterior
  density finiteness.

The same state-draw evaluator is used for the candidate and exact-model MCMC.

## Comparators

### No-Latent Ablation

The separately trained R=0 checkpoint is evaluated on all 324 contexts. Its
random-effect recovery and masked-cell proper scores are used for the frozen
weak and medium/strong comparison gates.

### Exact-Model MCMC

The fixed 36-context subset follows the preregistered shape, loading, and
prevalence selection. Each fit uses four chains, 1,000 warm-up iterations,
1,000 retained draws, and target acceptance 0.85.

If split R-hat exceeds 1.05 or bulk ESS is below 200, one continuation starts
from the retained chain states with the same owning chain seed and adds 1,000
adaptation and 1,000 retained iterations. Diagnostics are recomputed over the
combined retained draws. A second failure remains a gate failure.

Full exact-MCMC state draws and diagnostics are retained as compressed,
hash-inventoried artifacts.

### Qualified Python HMSC-HPC

Python-native HMSC-HPC is fitted to the same 36 masked contexts with:

- probit response;
- formula `~ x1`;
- one iid site random level;
- `nf = nfMin = nfMax = 2`;
- four chains;
- 1,000 transient and 1,000 retained draws;
- thin one;
- distinct owning-seed child streams.

The evaluator records known-site masked predictions, marginal new-site
predictions, and posterior-mean species associations. Every compiled and
posterior artifact is hash-inventoried.

### Immutable v0.1

The validated `neural_hmsc_v0_1` release is loaded only for the 40-site,
75-species, two-coefficient cells. The adapter preserves the immutable
checkpoint's historical `Intercept`/`TMG` coefficient names and formula while
passing the unchanged numeric simulated design. This is necessary for its
qualified calibration compatibility.

## Gate Engine

The gate engine implements all preregistered operational, calibration, joint,
association, predictive, posterior-predictive, and runtime decisions as named
booleans. There is no implicit pass for missing comparator values.

It validates exact ownership of:

- 324 candidate contexts;
- 324 no-latent contexts;
- the fixed 36-context exact-MCMC subset;
- the same 36 Python-HMSC contexts;
- every matched 40-by-75 v0.1 context.

It requires all comparator outputs to be finite and all exact-MCMC diagnostics
to pass. Aggregate and registered-stratum gates are conjunctive. The immutable
report decision is derived only from `all(gates.values())`.

A failed 502M report records `stop_before_reserved_evaluation`. Only a complete
pass records `eligible_to_authorize_503m_505m`; the evaluator itself never
opens reserved seeds.

## Artifacts

The one-shot output contains:

- `context_metrics.json.gz`;
- `fixed_validation_report.json`;
- `freeze.json`;
- compressed exact-MCMC state files;
- Python HMSC-HPC compiled and posterior files;
- file counts, byte sizes, and SHA-256 inventories;
- the immutable v0.1 release content hash;
- candidate, ablation, training-freeze, source-commit, and evaluator-version
  bindings.

Read-only validation rechecks the report, gate decision, seed flags, and every
comparator artifact hash without regenerating simulation.

## Scheduler Boundary

The new LUMI wrapper is:

`docs/lumi_generative_neural_hmsc_iid_v1_fixed_validation_sbatch.sh`

It uses `standard-g`, requires exact source, training-freeze, candidate, and
ablation hashes, runs preflight with the 502M token unset, restores the token
only for the one-shot evaluator, and leaves 503M-515M sealed.

Both production wrappers verify the Git commit and clean worktree in the host
shell before entering CSC's TensorFlow container. They pass a strict commit,
branch, and clean-status attestation into the container because that image
does not provide Git. Python accepts this fallback only when Git execution is
unavailable, the commit is a full lowercase SHA-1, and the host clean flag is
exactly one. Frozen document and production source-file hashes are still
validated inside the container.

The later 501M scheduler submission is documented separately in
`docs/generative_neural_hmsc_iid_v1_501m_validation_2026-07-28.md`.

## Verification

Local ordinary-seed verification reports:

- focused tests: `36 passed, 1 skipped`;
- complete synthetic 324-cell gate fixture: pass;
- exact-MCMC adapter with tiny chains/draws and continuation: pass;
- Python-native HMSC-HPC adapter with tiny chains/draws: pass;
- immutable v0.1 adapter against a validated local release: pass;
- read-only 501M preflight and opening-token refusal paths: pass;
- container host-source attestation acceptance and fail-closed rejection:
  pass;
- missing-token 501M and 502M refusal before output creation: pass;
- Python bytecode compilation: pass;
- both scheduler Bash syntax checks: pass;
- `git diff --check`: pass.

The skipped focused test remains the optional exact-MCMC test under
environments lacking a usable TensorFlow Probability runtime. The explicit
ordinary-seed exact adapter smoke passed in the current environment.

These checks establish evaluator completeness and API wiring, not candidate
quality or qualification.

## 501M Post-Training Validation Correction

Job `20301852` completed both frozen 200-epoch training paths and wrote the
full artifact bundle, then exited during read-only validation because the
corpus manifest used the earlier false seal key `fixed_validation_opened`
while the validator expected `fixed_validation_seed_ranges_opened`.

The validator now accepts either exact key, requires every present alias to be
false, and rejects missing, true, or conflicting values. Future generated
corpus manifests use the canonical key. Independent validation accepted the
immutable candidate and ablation checkpoints without retraining and retained
502M-515M as unopened. This correction changes no model, objective,
checkpoint, comparator, gate, threshold, or seed role.

## Next Barrier

Commit the validator correction and independent 501M evidence. Then run the
read-only 502M preflight pinned to the accepted candidate, ablation, and
training-freeze hashes and explicitly decide whether to authorize 502M. Keep
502M-515M sealed until that decision.
