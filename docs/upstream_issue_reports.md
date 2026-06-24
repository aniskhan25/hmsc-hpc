# Upstream Issue Reports

This document contains maintainer-ready reports for issues observed while
building and validating the Python-native workflow. These reports are scoped to
the original `hmsc-hpc` sampler behavior, not to wrapper-only `pyhmsc` features.

The public upstream issue tracker is:

```text
https://github.com/hmsc-r/hmsc-hpc/issues
```

As of 2026-06-24, the tracker already has related open issues for NNGP
reconsideration and species-trait failures. Prefer commenting on those existing
threads where the scope matches instead of creating duplicates.

## Report 1: `updateBetaLambda` failure for trait/phylogeny models with random levels

Suggested destination:

- New upstream issue, or a comment on the existing trait-related failure thread
  if maintainers prefer consolidating trait-model failures.

Suggested title:

```text
updateBetaLambda fails for trait/phylogeny-structured models with random levels
```

Suggested body:

````markdown
While validating hmsc-hpc on LUMI, I found that models combining species traits
and/or phylogeny with latent random levels reach an unsupported
`updateBetaLambda` path.

The observed failure was:

```text
NameError: name 'nfSum' is not defined
```

The failing model class is:

- probit response
- species traits with a trait formula
- phylogenetic covariance matrix
- iid site-level random intercept

The same validation workflow succeeds when the model is split into supported
submodels:

- fixed trait/phylogeny model: runs successfully
- environment-only iid random-intercept model: runs successfully

The combined trait/phylogeny + random-level model is currently guarded in my
Python-native wrapper before TensorFlow starts, because it reliably reaches this
upstream updater path.

Validation context:

- system: LUMI, `dev-g`, TensorFlow 2.16
- real-data validation: Whittaker plant data
- held-out split: 40 training sites, 12 TMG-stratified held-out sites
- successful fixed trait/phylogeny run: Brier 0.0742, log loss 0.2648,
  macro AUC 0.5518
- successful environment-only iid marginal run: Brier 0.0734, log loss 0.2607,
  macro AUC 0.5495

The important point is not the prediction score; it is that the component model
families run independently while the combined trait/phylogeny + random-level
state fails in `updateBetaLambda`.

Expected behavior:

`updateBetaLambda` should support models where the species-response structure is
trait/phylogeny-aware and the model also has latent random levels, or fail early
with an explicit unsupported-model error.

Current workaround:

Fit the trait/phylogeny model without random levels, or fit an environment-only
random-level model. This loses the combined HMSC model that users would expect
to be valid.
````

Local references:

- Guard: `pyhmsc/validation.py`
- Whittaker held-out workflow:
  `examples/generate_whittaker_holdout_validation.py`,
  `examples/analyze_whittaker_holdout_validation.py`,
  `docs/lumi_whittaker_holdout_validation_sbatch.sh`
- Validation docs: `docs/roadmap.md`,
  `examples/projects/whittaker_plants_hmsc_book/README.md`

## Report 2: NNGP Eta update runtime bottleneck

Suggested destination:

- Comment on upstream issue #29, `Potential for NNGP Revisit?`, because this is
  directly about NNGP behavior and performance.

Suggested title if maintainers prefer a new issue:

```text
NNGP Eta update is much slower than full spatial and GPP on compact real-data benchmark
```

Suggested body:

````markdown
I ran a compact real-data spatial hold-out benchmark on LUMI comparing fixed,
full spatial, GPP, and NNGP probit models on the same 400-site, 40-species plant
community subset.

Benchmark setup:

- data: compact big-spatial plant validation subset
- split: deterministic spatial block split, 319 train sites and 81 held-out
  sites
- models: fixed, full spatial, GPP, NNGP
- response: probit
- chains: 2
- saved draws: 250
- transient: 250
- thin: 5
- platform: LUMI `dev-g`, TensorFlow 2.16, MI250X GPU

Sampler-only elapsed times:

| model | sampler time |
| --- | ---: |
| fixed | 5.1 s |
| full spatial | 31.6 s |
| GPP | 15.7 s |
| NNGP | 613.1 s |

Peak process RSS was similar across models, about 1.9-2.3 GB, so the observed
issue looks primarily like an Eta-update runtime bottleneck rather than memory
pressure.

Predictive metrics were finite and reasonable, so this was not a failed run:

| model | Brier | log loss | macro AUC |
| --- | ---: | ---: | ---: |
| fixed | 0.070486 | 0.247399 | 0.730839 |
| full spatial | 0.072041 | 0.254337 | 0.724529 |
| GPP | 0.069072 | 0.243408 | 0.732161 |
| NNGP | 0.074632 | 0.263303 | 0.716890 |

The benchmark suggests that the current NNGP Eta update path may scale poorly
relative to full spatial and GPP on this workload. I am not proposing a wrapper
optimization here; this likely belongs in the core sampler/updater.

Reproducibility context:

- LUMI job: 19435459
- total benchmark wall time: 14 min 32 s
- validation script:
  `docs/lumi_big_spatial_holdout_validation_sbatch.sh`
- generator/analyzer:
  `examples/generate_big_spatial_holdout_validation.py`
  `examples/analyze_big_spatial_holdout_validation.py`
````

Local references:

- Runtime docs: `docs/roadmap.md`
- Project docs: `examples/projects/big_spatial_plants_validation/README.md`
- LUMI script: `docs/lumi_big_spatial_holdout_validation_sbatch.sh`

## Report 3: Multi-factor NNGP Eta prior precision ordering fix

Suggested destination:

- New upstream issue or pull request. This is a correctness fix candidate, not
  only a performance report.

Suggested title:

```text
Multi-factor NNGP Eta prior precision uses inconsistent vector ordering
```

Suggested body:

````markdown
While validating NNGP spatial random effects with more than one latent factor, I
found an ordering mismatch in the NNGP Eta prior precision construction.

The previous construction assembled the spatial precision in factor-major order,
while the likelihood and `mu0` vector use unit-major ordering. The fix is to map
the sparse precision indices by random-effect unit first and latent factor
second:

```python
row = iW.row * nf + h
col = iW.col * nf + h
```

This affected the multi-factor NNGP Eta updater path. Single-factor behavior is
unchanged by construction.

Validation:

- added focused updater coverage in `hmsc/test/updateEta_test.py`
- added a deterministic multi-factor NNGP Eta validation project with `nf=2`
- LUMI validation run `spatial_multifactor_eta_validation_real`, job `19276714`
- sampler completed in 6 min 8 s
- beta signs recovered: 8 / 8
- species PPC coverage: 8 / 8
- site richness PPC coverage: 64 / 64
- raw/aligned Eta mean truth correlation: 0.220537 / 0.856748
- raw/aligned Lambda mean truth correlation: 0.776261 / 0.916068
- association truth correlation: 0.981125

The raw factor means remain affected by latent-factor sign/permutation
non-identifiability, so aligned factor summaries and association summaries are
the useful validation targets.
````

Local references:

- Candidate fix commit: `b80c2b0`
- Affected file: `hmsc/updaters/updateEta.py`
- Test file: `hmsc/test/updateEta_test.py`
- Validation project:
  `examples/projects/simulated_spatial_multifactor_eta_validation/`
- Analyzer: `examples/analyze_spatial_multifactor_eta_validation.py`
- Existing register: `docs/upstream_hmsc_changes.md`

## Submission Notes

- The local environment does not currently have the GitHub CLI available, so
  these reports were prepared but not submitted directly from this session.
- The reports above are ready to paste into GitHub issues or comments.
- Recommended submission order:
  1. Comment on upstream #29 with Report 2.
  2. Open Report 1 as a new issue unless maintainers prefer using the existing
     trait-model thread.
  3. Open Report 3 as either a new issue or a focused pull request with only the
     `updateEta.py` fix and the associated test.
