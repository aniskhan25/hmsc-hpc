# Neural-HMSC Trait-Gamma Milestone 53A Terminal Result

Date: 2026-07-22

## Decision

`trait_gamma_probit_terminal_failure`

Milestone 53A used the frozen candidate weight SHA-256
`bc869b8a92e7d9ea0bf11acb565e571816a68dcff220f0a003f22d2d753cdcac`
and the preregistered finite-sample conformal multiplier
`1.3018141270106574`. No model, calibration, threshold, or gate changed after
the reserved evaluation was opened.

## Reserved Simulation Results

| Block | Coverage | Rank mean | Rank variance | Anchor RMSE ratio | MCMC Gamma RMSE ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 41000001 | 0.961240 | 0.546822 | 0.067586 | 1.018490 | 0.539924 | pass |
| 42000001 | 0.968992 | 0.534112 | 0.066833 | 1.003556 | 0.415765 | pass |
| 43000001 | 0.967054 | 0.542704 | 0.065928 | 0.985744 | 1.426715 | fail |

Block `43000001` failed only `mcmc_gamma_rmse`, whose fixed acceptance maximum
was `1.25`. Its MCMC Brier and log-loss ratios were `1.027125` and `1.039925`,
and all other simulation gates passed.

## Whittaker Replays

| Seed | Gamma MAE versus MCMC | Brier ratio | Log-loss ratio | Result |
| --- | ---: | ---: | ---: | --- |
| 44000001 | 0.320812 | 1.023183 | 1.011314 | pass |
| 44000002 | 0.230321 | 1.027909 | 0.993956 | pass |
| 44000003 | 0.332950 | 1.023835 | 0.999490 | pass |

## Provenance

- Calibration freeze SHA-256:
  `51d0d4fbf4486e3af2f1c890d9696d75ba1812047f9aafcaf7a15eebf274bb09`
- Evaluation artifact SHA-256:
  `af55e54172465893b3dbfde4a04a392cbdb55a9f875646f0aacd7bb30b0a467b`
- Existing immutable neural baselines modified: false
- Reserved evaluation opened: true
- Terminal rule applies: true

## Consequence

Neural trait-Gamma v1 is not qualified and may not be recalibrated or rerun.
Python MCMC remains the only qualified trait-Gamma path. iid Eta/Lambda remains
blocked because its structural prerequisite did not qualify. The bounded
post-failure scope decision returns development to variable-design fixed-effect
probit work under Milestone 54; see
`docs/neural_hmsc_post_m53a_scope_decision_2026-07-22.md`.
