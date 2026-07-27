# Trait-Gamma Calibration Decision Review

Date: 2026-07-22

## Decision

Milestone 53A completed with the preregistered terminal-failure decision. Do
not accept the neural trait-Gamma v1 candidate for deployment, and do not
proceed to iid Eta/Lambda qualification. Python MCMC remains the only qualified
trait-Gamma path.

This decision does not reopen neural representation or posterior-mean model
selection. The exact candidate weights are frozen by SHA-256:

`bc869b8a92e7d9ea0bf11acb565e571816a68dcff220f0a003f22d2d753cdcac`

Both qualified regression baselines remain immutable.

## Evidence Review

The predeclared candidate run covered 119 of 128 Gamma coefficients (`0.929688`).
The intended sensitivity run covered 113 of 128 (`0.882812`). Pooled coverage
was 232 of 256 (`0.906250`). The corresponding 95% Wilson intervals are broad:

| Evidence | Coverage | 95% Wilson interval |
| --- | ---: | ---: |
| Candidate | 0.929688 | 0.871765-0.962570 |
| Intended sensitivity | 0.882812 | 0.815634-0.927683 |
| Pooled | 0.906250 | 0.864298-0.936190 |

Mean quality is not the blocker. Both runs passed Gamma rank, Gamma mean versus
MCMC, held-out simulated proper-score, and Whittaker proper-score gates. The
failed run had negligible signed Gamma bias.

## Independence Defect

The intended sensitivity run incremented the base seed by one. Because each
corpus used consecutive seeds, it overlapped the candidate evidence:

| Corpus | Candidate seeds | Intended sensitivity seeds | Overlap |
| --- | --- | --- | ---: |
| Training | 20260801-20260864 | 20260802-20260865 | 63/64 |
| Calibration | 20360801-20360832 | 20360802-20360833 | 31/32 |
| Test | 20460801-20460864 | 20460802-20460865 | 63/64 |

The second result is useful evidence of calibration sensitivity, but it is not
the untouched independent evaluation required by the stop rule. Therefore the
roadmap cannot conclude that a fresh independent evaluation failed. It also
cannot promote from the single passing candidate run.

## Milestone 53A Preregistration

### Frozen Components

- Candidate weights remain byte-identical to the SHA-256 above.
- Model architecture, bounded joint Gamma anchor, Beta anchor, formulas,
  dimensions, simulation priors, and posterior means are frozen.
- No additional neural training, mean correction, routing, caps, or
  target-dataset selection is allowed.
- Existing candidate and intended-sensitivity outcomes may not choose any
  parameter of the new calibration.

### Calibration Corpus

- Use 384 new communities, twelve times the original 32-community corpus.
- Use six disjoint 64-community seed blocks beginning at `31000001`,
  `32000001`, `33000001`, `34000001`, `35000001`, and `36000001`.
- Balance the full 3 by 3 factorial of Gamma scale (`0.55`, `0.80`, `1.05`)
  and Beta residual scale (`0.10`, `0.20`, `0.35`).
- Fit one global split-conformal scalar to standardized absolute Gamma
  residuals. Use the finite-sample order statistic
  `ceil((n + 1) * 0.95)` divided by the standard-Normal 95% quantile.
- The calibration remains coefficient-posterior uncertainty only. It must not
  alter Beta, Gamma means, or predictive probabilities.

### Untouched Evaluation

Run three independent 258-community evaluations with seed blocks beginning at
`41000001`, `42000001`, and `43000001`. Each evaluation is balanced across the
same 3 by 3 factorial. None of these seeds may appear in training,
calibration, prior qualification, MCMC, or development diagnostics before the
calibration artifact is frozen.

For each evaluation block, require all of the following:

- overall Gamma 95% coverage between `0.90` and `0.99`;
- coverage of both Intercept and TMG Gamma coefficients at least `0.90`;
- coverage in every Gamma-scale by residual-scale cell at least `0.85`;
- normalized rank mean between `0.40` and `0.60`;
- normalized rank variance between `0.06` and `0.11`;
- Gamma posterior-mean RMSE no worse than `1.05` times the frozen analytical
  anchor;
- simulated neural/MCMC Gamma RMSE ratio no greater than `1.25`;
- held-out simulated Brier and log-loss ratios no greater than `1.05`.

Run the frozen Whittaker holdout with three MCMC RNG seeds. Every replay must
retain Brier and log-loss ratios no greater than `1.05` and Gamma mean MAE to
MCMC no greater than `0.35`. Whittaker outcomes remain evaluation-only.

### Terminal Rule

Promotion requires every gate in all three untouched simulation blocks and all
three Whittaker replays to pass. A failure in any block permanently ends the
v1 neural trait-Gamma path: Python MCMC remains the only qualified trait-Gamma
implementation, no baseline is frozen, and iid Eta/Lambda work remains blocked
pending a separate roadmap decision.

If all gates pass, freeze `neural_hmsc_trait_gamma_probit_v1`, expose its loader
through the stable API, and retain qualified Python MCMC as the statistical
reference. Passing establishes only the declared Beta/Gamma marginal and
predictive scope, not coupled joint-posterior equivalence.

## Milestone 53A Result

The Milestone 53A calibration and fixed-evaluation harness is implemented in
`examples/run_neural_hmsc_trait_gamma_m53a.py`. Its disposable smoke used only
the `51000001` and `52000001` blocks and passed. It preserved the frozen weight
SHA-256 byte-for-byte, balanced all nine factorial cells, packaged a
finite-sample conformal multiplier of `1.427889`, completed simulated MCMC and
Whittaker paths, and recorded `reserved_seed_opened=false`. Smoke observations
are plumbing evidence only and do not change production gates or parameters.

The production `calibrate` command has now frozen the preregistered
384-community artifact from the six `31000001` through `36000001` seed blocks.
The finite-sample conformal multiplier is `1.3018141270106574`; the nine
factorial cells contain 42 or 43 communities each. Independent validation
confirmed calibration artifact SHA-256
`3ab539e117827a73718a03b19cee3e1c1191484038d2c67b3b58a5f7746f40a9`,
checkpoint manifest SHA-256
`aed26718e224fea37c29a8701249f7c149fc98eccb17a2fe1bbbf91ae8612554`,
and the unchanged frozen weight SHA-256
`bc869b8a92e7d9ea0bf11acb565e571816a68dcff220f0a003f22d2d753cdcac`.
The freeze entered evaluation in status `frozen_before_reserved_evaluation`
with `reserved_evaluation_opened=false`.

The authorized one-shot evaluation then opened all three reserved simulation
blocks and all three Whittaker replays without changing the model, calibration,
thresholds, or gates. Blocks `41000001` and `42000001` passed every gate. Block
`43000001` passed coverage, coefficient, cell, rank, anchor, MCMC predictive,
and held-out predictive gates, but failed the preregistered MCMC Gamma-RMSE
gate: its neural-to-MCMC Gamma RMSE ratio was `1.426715144159916`, above the
fixed `1.25` maximum. All three Whittaker replays passed.

The final evaluation artifact SHA-256 is
`af55e54172465893b3dbfde4a04a392cbdb55a9f875646f0aacd7bb30b0a467b`.
Its decision is `trait_gamma_probit_terminal_failure` and
`terminal_rule_applies=true`. Under the preregistered rule, neural trait-Gamma
v1 is closed without tuning or rerun. The next roadmap action is a bounded
scope decision: retain Python MCMC as the only trait-Gamma implementation and
either return to already-qualified fixed-effect neural work or separately
propose a representation-level structural-family milestone. iid Eta/Lambda
qualification remains blocked.
