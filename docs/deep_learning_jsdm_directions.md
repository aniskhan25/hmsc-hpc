# Deep Learning Directions for HMSC/JSDM Research

This note records possible research directions for adding deep learning ideas to
the `hmsc-hpc` / `pyhmsc` ecosystem. It is intentionally not a roadmap. The
goal is to preserve candidate ideas for later selection and design work.

The main constraint is that a plain neural network JSDM is not enough. Similar
models already exist in the literature: multi-output species distribution
models, neural species embeddings, heterogeneous graph SDMs, biodiversity
foundation models, and VAE/GAN-style ecological generators. The more useful
direction is to borrow deep learning mechanisms that are mature elsewhere but
are not yet standard in HMSC-like joint species distribution modeling.

## Framing

HMSC provides more than prediction. It gives posterior samples, interpretable
fixed effects, trait and phylogeny structure, latent random effects, species
associations, posterior predictive checks, and uncertainty semantics.

A deep learning extension should therefore be judged by whether it contributes
one of the following:

- faster approximate inference for HMSC-shaped models
- better transfer to new regions, sites, or rare species
- nonlinear environment-trait response surfaces while retaining ecological
  structure
- calibrated predictive uncertainty or community-level uncertainty
- richer spatial random effects
- high-order community generation beyond pairwise residual association

## Candidate Directions

### 1. Amortized Neural-HMSC Posterior

Rooted in:

- normalizing flows
- simulation-based inference
- variational inference
- amortized Bayesian inference

Core idea:

Train a neural posterior approximator for HMSC-like parameters:

```text
q_phi(Beta, Gamma, Eta, Lambda, Sigma | Y, X, traits, phylogeny, space)
```

The neural model is not the ecological model. It is a fast approximate posterior
engine for an HMSC-shaped model.

Why this is interesting:

- preserves HMSC targets such as `Beta`, `Gamma`, `Eta`, `Lambda`, associations,
  and posterior predictive summaries
- can be trained on simulated datasets generated from the existing `pyhmsc`
  simulation and validation machinery
- can be calibrated against existing MCMC output from `hmsc-hpc`
- could make repeated model fitting much faster once trained

Possible implementation shape:

```text
compiled model arrays
  -> permutation-aware encoder over sites/species
  -> posterior head for fixed effects
  -> posterior head for random effects
  -> posterior head for latent loadings
  -> flow or low-rank Gaussian posterior sampler
```

Main risks:

- factor non-identifiability for `Eta` and `Lambda`
- posterior calibration may be poor without simulation coverage
- amortized inference can fail out of distribution
- credible intervals need careful validation against MCMC

Useful references:

- Variational Inference with Normalizing Flows: https://arxiv.org/abs/1505.05770
- Neural Spline Flows: https://arxiv.org/abs/1906.04032

### 2. Conditional Neural Process for Community Ecology

Rooted in:

- Conditional Neural Processes
- Neural Processes
- few-shot probabilistic learning
- meta-learning

Core idea:

Treat observed site-species responses as a context set, then infer predictions
for unobserved site-species pairs, new sites, new regions, or rare species.

```text
context:
  (site_i, species_j, y_ij, X_i, traits_j, space_i)

target:
  (site_k, species_l, X_k, traits_l, space_k)

output:
  p(y_kl | context, target)
```

Why this is interesting:

- targets missingness, sparse communities, rare species, and transfer rather
  than just dense matrix prediction
- naturally supports different numbers of sites and species
- can be trained across many simulated or empirical communities
- could provide fast uncertainty for new ecological surveys

Possible implementation shape:

```text
context encoder over observed responses
target encoder over requested predictions
latent global community variable
decoder for Bernoulli, Poisson, negative-binomial, or zero-inflated likelihood
```

Main risks:

- context selection matters
- uncertainty can be under-dispersed
- ecological interpretability is weaker than direct HMSC parameters unless
  explicit HMSC-shaped heads are added

Useful references:

- Conditional Neural Processes: https://arxiv.org/abs/1807.01613
- Neural Processes: https://arxiv.org/abs/1807.01622

### 3. Permutation-Equivariant Site-Species Transformer

Rooted in:

- Set Transformer
- Perceiver IO
- sparse attention
- permutation-equivariant learning

Core idea:

Represent the community dataset as a set of site tokens, species tokens, and
observed site-species response edges rather than as a fixed rectangular matrix.

```text
site tokens:
  X_i, coordinates, time, sampling design

species tokens:
  traits_j, taxonomy_j, phylogeny_j

edge tokens:
  y_ij, observed/missing flag, effort/detection metadata
```

The model should be invariant or equivariant to row and column ordering.

Why this is interesting:

- most neural SDMs assume a fixed species list and output ordering
- exchangeability is closer to the statistical structure of community data
- supports missing data and variable species sets naturally
- can be extended with interpretable heads for associations or latent factors

Possible implementation shape:

```text
site set encoder
species set encoder
sparse cross-attention between observed site-species pairs
decoder for target site-species pairs or full community vectors
```

Main risks:

- attention cost can be high for large site by species matrices
- summaries such as variance partitioning are not automatic
- requires careful validation to avoid just learning prevalence

Useful references:

- Set Transformer: https://arxiv.org/abs/1810.00825
- Perceiver IO: https://arxiv.org/abs/2107.14795

### 4. Conditional Diffusion Model for Whole Communities

Rooted in:

- diffusion models
- discrete diffusion
- tabular diffusion
- generative modeling

Core idea:

Generate entire community response vectors conditional on environment, space,
traits, and possibly observed partial communities:

```text
Y_i ~ diffusion_model(Y | X_i, space_i, traits, phylogeny)
```

This shifts the target from independent site-species predictions to
community-level generation.

Why this is interesting:

- can capture high-order co-occurrence patterns beyond pairwise covariance
- naturally supports posterior-predictive-style simulation
- could generate counterfactual communities under environmental scenarios
- could model richness and composition jointly

Possible implementation shape:

```text
condition encoder for site covariates and spatial context
species/trait conditioning block
binary/count/tabular diffusion process over community vector
community-level calibration and PPC diagnostics
```

Main risks:

- discrete ecological responses need careful likelihood design
- interpretation is weaker than HMSC unless diagnostic summaries are built
- diffusion may overfit prevalence patterns without ecological constraints

Useful references:

- TabDDPM: https://arxiv.org/abs/2209.15421
- Deep Generative AI for species coexistence: https://arxiv.org/abs/2107.06020

### 5. Phylogeny-Aware Neural Priors

Rooted in:

- graph neural networks
- graph attention
- hierarchical embeddings
- neural empirical Bayes

Core idea:

Use phylogeny or taxonomy as a neural prior over species parameters instead of
only as a covariance matrix.

```text
phylogeny graph -> GNN -> prior mean/scale for Beta_j, Lambda_j, Gamma_j
```

Why this is interesting:

- related species can share information nonlinearly
- rare species can borrow strength from taxonomic or phylogenetic neighbors
- retains HMSC-like species coefficients
- can complement existing phylogenetic covariance support

Possible implementation shape:

```text
tree/taxonomy graph encoder
species trait encoder
prior parameter head for species-specific coefficients
HMSC-like likelihood with neural prior regularization
```

Main risks:

- phylogenetic signal may be weak or confounded with traits
- tree uncertainty is usually ignored
- neural priors need careful checks to avoid overpowering data

Useful references:

- Graph Attention Networks: https://arxiv.org/abs/1710.10903
- Graph Neural Networks review: https://arxiv.org/abs/1812.08434

### 6. Neural Spatial Random Effects and Neural Operators

Rooted in:

- neural fields
- deep spatial models
- Fourier Neural Operators
- coordinate networks

Core idea:

Replace or augment spatial `Eta` with a learned continuous spatial process:

```text
Eta(s) = neural_field(s, environmental context)
```

or learn an operator from observed spatial community structure to latent fields:

```text
observed spatial community field -> neural operator -> predicted latent field
```

Why this is interesting:

- can model nonstationary spatial dependence
- can interpolate latent community structure to new coordinates
- can scale to dense spatial grids or environmental rasters
- could augment full, GPP, or NNGP spatial random effects

Possible implementation shape:

```text
coordinate/environment encoder
spatial latent field network
species loading matrix
HMSC-like likelihood
optional GP-style or NNGP-style regularization
```

Main risks:

- neural spatial fields can extrapolate badly
- uncertainty is not automatic
- spatial confounding with environmental covariates needs explicit diagnostics

Useful references:

- Fourier Neural Operator: https://arxiv.org/abs/2010.08895
- Learning nonlinear operators via DeepONet: https://www.nature.com/articles/s42256-021-00302-5

### 7. Conformal Ecological Uncertainty Layer

Rooted in:

- conformal prediction
- conformalized quantile regression
- distribution-free uncertainty

Core idea:

Wrap neural JSDM predictions with calibrated prediction intervals or prediction
sets.

```text
neural predictor + calibration residuals -> species/site/community coverage
```

Why this is interesting:

- neural uncertainty is often poorly calibrated
- conformal calibration could provide empirical coverage for held-out species,
  sites, richness, or community composition
- works as an add-on to multiple neural architectures
- can be evaluated with existing pyhmsc holdout workflows

Possible implementation shape:

```text
fit neural JSDM
reserve calibration sites or species
estimate nonconformity scores
emit calibrated intervals/sets for response, richness, or composition
```

Main risks:

- exchangeability assumptions may fail under spatial blocking
- coverage target must be defined carefully
- intervals may be too wide if the base model is weak

Useful references:

- Conformalized Quantile Regression: https://arxiv.org/abs/1905.03222

## Most Promising Starting Points

The strongest candidates for this repository are:

1. Amortized Neural-HMSC posterior
2. Conditional Neural Process for community ecology
3. Conditional diffusion model for whole communities

The first is closest to the current codebase because `pyhmsc` already has
compiled artifacts, simulation projects, posterior storage, diagnostics,
associations, and posterior predictive checks. It also preserves the central
scientific value of HMSC: interpretable posterior structure rather than only
black-box prediction.

The second is attractive if the main scientific question becomes transfer:
rare species, sparse observations, new regions, new surveys, or adaptive
sampling.

The third is attractive if the focus becomes community assembly simulation,
composition-level counterfactuals, and high-order co-occurrence beyond
`Lambda.T @ Lambda`.

## Evaluation Principles

Any selected direction should be compared against the existing HMSC path using:

- held-out prediction log likelihood
- calibration of predicted occurrence/count probabilities
- species-level posterior predictive checks
- site richness posterior predictive checks
- residual spatial autocorrelation
- recovery of simulated `Beta`, `Gamma`, `Eta`, and `Lambda` where applicable
- species association recovery from identifiable association matrices
- behavior under rare species and sparse observation regimes
- behavior under spatially blocked holdout splits

## Working Hypothesis

The most defensible research contribution is not "deep learning replaces
HMSC." It is:

```text
Deep learning can amortize, regularize, or extend HMSC-like joint species
models while preserving ecological structure and testable uncertainty targets.
```

That framing leaves room for predictive neural baselines, but it keeps the
primary research goal aligned with what makes HMSC scientifically useful.
