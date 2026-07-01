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
