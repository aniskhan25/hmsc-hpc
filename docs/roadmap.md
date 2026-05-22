# Python API Roadmap

## Milestone 1: Python wrapper over R initialization

Implemented initial package skeleton:

- `pyhmsc.HmscModel`
- R bridge that writes CSV inputs and generates `make_init.R`
- runner for `python -m hmsc.run_gibbs_sampler`
- `pyhmsc.HmscFit` with `beta_mean()`, `beta_ci()`, `summary("Beta")`, and
  fixed-effect Poisson prediction
- example: `examples/simple_birds_r_bridge.py`

This milestone still requires R plus the R packages `Hmsc` and `jsonify`.

## Milestone 2: Native fixed-effect path

- Use `docs/hmsc_hpc_input_schema.md` as the working schema reference.
- Use `pyhmsc compile` / `HmscModel.compile()` to create the Python-native
  `init.json` + `init_arrays.h5` artifact for fixed-effect models.
- `hmsc.run_gibbs_sampler --input run/init.json --output run/posterior.json`
  now loads the Python-native fixed-effect artifact directly.
- Validate the path with pure-Python schema tests and sampler smoke tests for
  Gaussian, Poisson, and Probit models.

## Milestone 3: Python-native fixed-effect initializer

- Harden fixed-effect Gaussian, Poisson, probit, and Bernoulli models with
  simulation-based validation.
- Compare posterior summaries against known simulated coefficients and posterior
  predictive checks.
- Add JSON/HDF5 posterior storage for larger fixed-effect runs, then Zarr.

## Later milestones

- iid random intercepts by study-design group
- trait design matrices and trait-related posterior summaries
- phylogenetic covariance matrix support
- spatial random effects after the simpler model classes are stable
