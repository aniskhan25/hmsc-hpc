# Generative Neural-HMSC IID v2 Representation Decision

Date: 2026-07-31

## Decision

Use the single permitted representation redesign.

The selected candidate is:

`generative_neural_hmsc_iid_probit_v2_orbit`

It replaces the v1 raw-state joint Gaussian with:

- a joint low-rank multivariate Student-t block for alpha, Beta, and log(tau);
- an exact O(2)-orbit-symmetrized matrix-Normal block for Eta and Lambda;
- conditional global-to-latent dependence;
- a permutation-equivariant bipartite attention encoder; and
- four fixed objective-only semi-amortized refinement steps.

The frozen preregistration is:

`docs/generative_neural_hmsc_iid_v2_orbit_preregistration_2026-07-31.md`

SHA-256:

`a2eaee0441833167f707f7cb9ae6b1162ba4e118ee3dfc1a245983cc9ada24c2`

The seed reaudit is:

`docs/generative_neural_hmsc_iid_v2_seed_reaudit_2026-07-31.json.md`

SHA-256:

`9a463943508651e74855701cdbd9870961efd3fd3c07a444674da36a67d49344`

This decision authorizes implementation and ordinary non-ledger feasibility
tests only. It does not authorize 593M-594M or 511M-515M.

## Why a Redesign Is Still Justified

The 502M result was not ambiguous. It demonstrated:

- near-zero recovery of association direction;
- R coverage of 0.306 and R/exact width ratio of 0.220;
- log(tau) coverage of 0.025;
- Beta coverage of 0.674 and biased Beta ranks;
- performance effectively equal to the no-latent ablation;
- proper scores materially worse than exact MCMC and Python HMSC.

The exact-MCMC chains, comparator artifacts, runtime, and evaluator all passed
their operational checks. The failure therefore isolates the candidate
representation rather than the target, simulator, or evidence pipeline.

The original design review explicitly identified a single Gaussian in raw
factor coordinates as the primary accepted risk. The observed collapse is the
predicted failure mode. Testing one symmetry-correct representation is
scientifically distinct from continuing the earlier calibration loop.

## Alternatives Considered

### Close the iid family immediately

This is the lowest-cost option and remains the fallback. It was not selected
now because one fresh redesign was explicitly reserved for this exact failure,
the seed ledger remains unused, and an exact two-factor orbit density is
tractable.

The redesign closes automatically if its ordinary-fixture mathematics fails
or if fresh fixed validation fails.

### Apply a normalizing flow or mixture in raw Eta/Lambda coordinates

Rejected.

A generic flow or finite Gaussian mixture would spend capacity approximating
arbitrary factor rotations. It would not enforce continuous O(2) invariance,
would make permutation behavior harder to guarantee, and could still average
association products toward the no-latent solution. More expressivity in the
wrong coordinates is not a controlled response to the evidence.

### Infer only R and A/C directly

Rejected for this bounded redesign.

R and A are the correct inferential summaries, but a direct posterior over
rank-two R and positive-semidefinite A would require a new manifold measure,
an induced-prior Jacobian, constrained support, and a new mapping back to
raw Eta/Lambda for the unchanged generative density. That is closer to a new
statistical model than a posterior/encoder redesign and risks invalidating the
exact-target comparison.

### Orbit-symmetrized block posterior

Selected.

Concatenating Eta and transpose(Lambda) makes the full non-identifiable action
a right multiplication by O(2). With factor-isotropic matrix-Normal covariance,
the orbit integral has a closed two-component Bessel form. Sampling and log q
can therefore respect the full continuous symmetry without orientation labels,
teachers, or calibration.

Separating the global state into a Student-t block directly addresses broad
marginal underdispersion. Fixed semi-amortized refinement addresses the
amortization gap and mean bias using only the unchanged IWAE objective.

## Why This Is One Representation Change

The components form one posterior construction:

```text
q(theta, Z | data) = q_student(theta | data)
                     q_orbit(Z | theta, data).
```

The attention network and refinement steps parameterize that distribution.
They do not create a second candidate, selector, fallback, or calibration
layer. The simulator, model, prior, likelihood, factor count, objective class,
data support, comparators, gates, and thresholds remain unchanged.

## Risks and Fail-Closed Controls

### Orbit-density algebra is wrong or unstable

Control:

- dense and numerical-quadrature parity on ordinary float64 fixtures;
- explicit O(2) density and invariant-product tests;
- no disposable seed until those tests pass.

### Student-t tails inflate intervals without improving ranks

Control:

- no coverage-only acceptance;
- unchanged SBC rank, exact-width, energy-score, and proper-score gates remain
  conjunctive.

### Refinement becomes hidden per-dataset tuning

Control:

- exactly four fixed steps;
- only IWAE gradients;
- no truth, comparator, ecological outcome, or gate feedback;
- identical procedure in training, validation, and deployment;
- fixed runtime gate.

### Attention breaks exchangeability

Control:

- no IDs or positional encodings;
- shared site/species parameters;
- float64 invariant reductions;
- unchanged permutation and padding gates.

### Implementation scope expands after smoke

Control:

- architecture, posterior math, refinement, seeds, and sequence are frozen
  before implementation;
- disposable results cannot change them;
- any mathematical redesign requires closure, not amendment.

## Bounded Outcome

There are only three allowed outcomes:

1. ordinary-fixture feasibility fails and the iid family closes without a
   redesign seed;
2. implementation passes, but fresh 512M validation fails and the iid family
   closes permanently; or
3. every fixed and reserved gate passes, after which the unchanged Whittaker
   boundary is evaluated.

There is no third representation attempt and no calibration phase.

## Authorized Next Step

Implement the v2 posterior-math and encoder skeleton using ordinary non-ledger
fixtures only:

1. masked low-rank Student-t block;
2. O(2)-orbit matrix-Normal sample and log density;
3. conditional joint posterior assembly;
4. four-step refinement;
5. checkpoint schema and v1 incompatibility;
6. dense/quadrature, gradient, permutation, padding, and immutable-hash tests.

Run no disposable or production workflow until all frozen feasibility gates
pass and the implementation is reviewed from a clean commit.
