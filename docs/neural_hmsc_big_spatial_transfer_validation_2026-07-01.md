# Frozen Neural-HMSC Big Spatial Plant Transfer

This validation applies the qualified Whittaker Neural-HMSC checkpoint to an
independent real plant community dataset without updating neural weights or
calibration parameters.

## Design

- source: repository Big Spatial Plant community data
- training sites: 40 selected by coordinate maximin, without response data
- held-out sites: remaining 360 sites
- species: 75 selected by training prevalence only
- response: presence/absence with probit link
- environmental projection: standardized `Max_temp_smooth` mapped to `TMG`
- reference: target-specific fixed-effect MCMC using `presence ~ TMG`
- frozen source: Whittaker LUMI job `19637813`

The workflow records SHA-256 fingerprints for the checkpoint, source
coefficient artifact, calibration metadata, and source acceptance report.
Target holdout responses are used only for final metrics. Passing the transfer
gate establishes predictive transfer under this fixed-shape projection; it
does not establish target-domain coefficient-posterior calibration.

## Production Result

LUMI job `19638224` completed in 1 minute 17 seconds with exit code `0:0` and
peak RSS of approximately 3.16 GB. It supersedes label-only run `19638197` by
recording predictive transfer and coefficient calibration as separate
decisions. Frozen artifact fingerprints were:

- checkpoint: `f24aa6e8657877652eaa173a327b7e4cb815d6387a690f7266e7d0c48c545cb2`
- coefficient source: `48131c5bd495594a0ed758f2ae85a1e97d4392acbb389051ed202251c2d9066f`
- calibration: `4c58c379434d6d625cbe76dc85d8d31068e9f566cd3724e8eccae42da5822f16`
- source acceptance: `4177480c674a3744e6b4e0ba4193f0fea1373af23fe903333c9ce86a57742832`

The manifest records `weights_updated: false` and
`calibration_updated: false`.

| Model | Brier | Log loss | Macro AUC | Prevalence MAE | Richness MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neural uncalibrated | 0.05698 | 0.22917 | 0.61419 | 0.07715 | 5.6834 |
| Neural coefficient-calibrated | 0.10019 | 0.36479 | 0.58663 | 0.21427 | 16.0985 |
| Neural predictive-only | 0.05672 | 0.22808 | 0.61250 | 0.07592 | 5.5721 |
| MCMC fixed | 0.04749 | 0.19161 | 0.64714 | 0.03999 | 3.5466 |

The frozen predictive-only artifact passed all predefined target criteria. Its
Brier and log-loss ratios versus uncalibrated neural prediction were `0.9953`
and `0.9952`; ratios versus MCMC were `1.1943` and `1.1903`. Prevalence and
richness MAE ratios versus uncalibrated prediction were `0.9840` and `0.9804`.
The explicit `predictive_transfer_acceptance_passed` decision was true.

The coefficient posterior requires a separate caveat. Its mean correlation
with MCMC was `0.9322`, but its 95% interval overlap was only `0.1648`, and its
coefficient-SD RMSE against MCMC was `1.6369`. The Whittaker coefficient scale
therefore remains an inherited source-domain uncertainty correction, not a
validated target-domain posterior calibration. Real target data do not provide
coefficient truth, so target coefficient calibration is recorded as not
assessable rather than passed.

Frozen neural inference took 0.160 seconds and MCMC sampling took 31.8 seconds,
for a 198.7x inference-only speedup.

## Qualified Comparator Rerun

Date: 2026-07-19

Final LUMI job: `20001432`

Remote run root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_big_spatial_transfer_qualified_retry2_20260719`

This rerun used the newly completed Whittaker `external_monotone` artifact as
the frozen source:

```text
/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_whittaker_extmono_qualified_20260719
```

It attached the passed Big Spatial direct R/Python parity metrics:

```text
/scratch/project_462000131/anisrahm/hmsc-hpc-runs/direct_r_python_big_spatial_full_parity_20260719/big_spatial_plants_validation_model_spatial_full/direct_r_python_parity_metrics.json
```

The first dependent submission, job `20000925`, failed because the transfer
runner still assumed that the frozen coefficient posterior carried the scalar
predictive-only multiplier. The promoted Whittaker run stores coefficient
calibration in `neural_posterior.h5` and predictive calibration in
`neural_predictive_distribution.h5`. Retry job `20001335` exposed one stale
variable name in the posterior writer. Job `20001432` completed successfully
after the transfer runner was corrected to load the two calibration artifacts
separately.

Acceptance:

| gate | result |
| --- | --- |
| inherited source SBC acceptance | pass |
| target predictive acceptance | pass |
| frozen predictive transfer acceptance | pass |
| reference parity qualification | pass |
| target coefficient calibration assessable | false |

Held-out metrics:

| model | Brier | log loss | macro AUC | prevalence MAE | richness MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| neural uncalibrated | `0.051242` | `0.203917` | `0.640447` | `0.051664` | `4.566353` |
| neural coefficient-calibrated | `0.052362` | `0.211971` | `0.640800` | `0.064194` | `5.288344` |
| neural predictive-only | `0.052811` | `0.211215` | `0.632069` | `0.059852` | `5.148121` |
| qualified Python MCMC fixed | `0.047494` | `0.191519` | `0.647862` | `0.039840` | `3.545930` |

Reference parity diagnostics from the attached Big Spatial direct parity run:

| diagnostic | value |
| --- | ---: |
| `Beta` mean correlation | `0.984128` |
| `Gamma` mean correlation | `0.999570` |
| random-level association correlation | `0.749967` |

Frozen neural inference took `0.132` seconds, and qualified Python MCMC sampling
took `36.349` seconds, for a `274.442x` inference-only speedup.

This confirms the frozen Big Spatial transfer gate with the comparator now
explicitly qualified against the original R+Python HMSC-HPC boundary. It still
does not establish target-domain coefficient posterior calibration because the
real Big Spatial target has no coefficient truth.

## Next Decision

The bounded three-seed real-data sensitivity check completed as LUMI job
`20001710`; see
`docs/neural_hmsc_realdata_sensitivity_2026-07-19.md`. Big Spatial
frozen-transfer passed in all three seeds with the Big Spatial direct parity
metrics attached. The qualified Python MCMC comparator retained the proper-score
advantage in all three Big Spatial seeds: mean Brier ratio versus MCMC was
`1.0945`, mean log-loss ratio was `1.0866`, and mean macro AUC delta was
`-0.0167`.

The next roadmap step is to resume simulated neural competitor development,
using Big Spatial transfer as a frozen real-data gate rather than adding another
dataset-specific calibration adjustment.
