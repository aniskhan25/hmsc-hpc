# R Parity Checks

The Python-native workflow does not require R at runtime. This optional
one-time validation checks that selected compiled Python-native model artifacts
match base R formula and study-design encodings.

Run from the repository root:

```bash
python examples/run_r_parity_checks.py --output run/r_parity_checks
```

If R may not be installed, use:

```bash
python examples/run_r_parity_checks.py \
  --output run/r_parity_checks \
  --skip-if-r-missing
```

The default cases cover:

- fixed-effect Poisson: `tests/fixtures/fixed_effect/model.yaml`
- fixed-effect traits and phylogeny:
  `tests/fixtures/fixed_effect/model_traits_phylo.yaml`
- environment-only iid random intercept:
  `examples/projects/iid_random_intercept/model.yaml`

The script compiles each model with `pyhmsc`, asks base R to build the
equivalent `model.matrix` and factor encodings, and compares:

- fixed-effect design matrix `X`
- trait design matrix `T`, when present
- ordered phylogenetic covariance matrix `C`, when present
- study-design random-level integer codes, when present
- compiled native validation status

This is not a replacement for statistical validation or long MCMC runs. It is a
targeted compatibility check for the model-construction boundary where Python
replaces R.

Known limitation: the check intentionally avoids the guarded
trait/phylogeny-plus-random-level combination until the upstream `hmsc`
`updateBetaLambda` path supports that model family.

## Archived Result

The one-time default parity check was run on LUMI on 2026-06-23 and all default
cases passed. See:

- [R parity check archive: 2026-06-23](r_parity_checks_2026-06-23.md)
