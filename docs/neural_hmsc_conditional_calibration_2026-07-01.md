# Neural HMSC Conditional Coefficient Calibration

Date: 2026-07-01

## Implementation

Milestone 12 now includes a structured conditional scale head in
`pyhmsc/neural/conditional_calibration.py`. The head predicts one positive
multiplier per dataset, coefficient, and species from quantities available at
inference time:

- logit nonzero prevalence
- log expected diagonal design information
- log raw neural posterior standard deviation
- centered coefficient identity
- prevalence-by-coefficient interaction

Each continuous feature has a linear and positive-hinge term. The model is
initialized from the existing global scalar multiplier. Version 4 combines a
prevalence-weighted Gaussian log score with differentiable analytic SBC
rank-mean and rank-variance penalties. Rare and intermediate species receive
greater fitting weight. Ridge regularization shrinks the head toward the scalar
baseline, and a final scalar normalization targets nominal marginal coverage.

Version 4 also stores robust feature bounds and a regularized Mahalanobis
support radius from simulated calibration data. Conditional adjustments are
blended with the scalar multiplier in log space; trust decays beyond either
support boundary and reaches the scalar fallback under substantial feature
shift.

The calibrator does not alter posterior means. For a full-covariance posterior,
coefficient scales form a diagonal matrix `D` and each per-species Cholesky
factor becomes `D L`, giving covariance `D Sigma D`.

## Semantics

Anchored-model conditional calibration metadata initially used
`semantics_version: 5` and method `conditional_rank_aware_anchor_scale`. The
OOD-aware update writes `semantics_version: 6` for the same method and adds
bounded support-excess uncertainty inflation after scalar fallback. The metadata
stores feature normalization, weights, coefficient names, multiplier bounds,
rank-objective settings, feature-support geometry including posterior-mean
magnitude, fitting hyperparameters, calibration coverage, and
scalar-versus-conditional log scores and rank losses. Version 3
`conditional_structured_scale`, version 4 `conditional_rank_aware_scale`, and
legacy version 5 metadata remain loadable for reproducibility.

The learned OOD-objective update writes `semantics_version: 7` for the same
method when held-out OOD calibration batches are supplied. Version 7 replaces
the fixed support-excess exponential with a learned bounded softplus curve. The
first version used support excess alone. The effect-size-aware revision keeps
legacy support-only v7 metadata loadable and fits a five-parameter curve over
both support excess and positive standardized posterior-mean magnitude. The
additional effect-size signal is intended to trigger under coefficient
magnitude shifts where covariate/support trust remains high. The curve is fit
only from simulated OOD calibration batches, penalizes OOD coefficient coverage
and rank-moment errors, and includes an in-domain gate penalty so in-domain SBC
acceptance remains a hard constraint. Posterior means remain fixed.

The gated effect-size update writes `semantics_version: 8` when the opt-in
`support_effect_gated_rank_coverage` objective is used. Version 8 serializes
`support_effect_gated_learned_softplus`, keeps the support-excess branch, and
multiplies the effect-size branch by a learned OOD-context gate over support
excess and effect-signal magnitude. Version 8 also adds a direct in-domain
extra-inflation penalty during OOD-objective fitting. The stratified-gate
revision expands the in-domain OOD gate from prevalence-only rank groups to
prevalence, design-information, and coefficient groups, and includes
per-stratum coverage penalties. The constrained-branch revision adds a
backward-compatible ninth curve parameter, `effect_high_design_suppression`,
which suppresses the effect-size branch for high-design, support-close
coefficient contexts and adds a matching in-domain cap penalty during fitting.
The stratum-conditioned revision further adds learned gate offsets for
prevalence strata, design-information strata, and coefficient identity, plus
per-stratum in-domain extra-inflation caps.
The residual in-domain revision adds two mechanisms outside the OOD inflation
gate: an optional prevalence-by-coefficient mean-bias correction, which remains
serialized but zero by default after failed local transfer, and active
stratum-specific base log-scale offsets learned in the main conditional scale
objective for version 8 gated calibration. A held-out monotone rank-centering
mechanism is also serialized and applicable, but automatic fitting remains
disabled after local SBC transfer worsened rare-prevalence rank error.
Legacy support-only and ungated effect-aware version 7 metadata remain
loadable, as do earlier seven- and eight-parameter version 8 curves with zero
high-design suppression and earlier version 8 curves without stratum offsets.

The training-time rank-mean penalty is separate from coefficient calibration.
It is an opt-in neural training objective that evaluates rare-prevalence rank
means on held-out simulation batches and records its history in benchmark
metadata. It remains experimental and is not enabled by default because local
sanity runs improved rare-prevalence rank error but did not clear the full
stratified in-domain/OOD gate. A later redesign added prevalence-weighted rank
centering, delayed activation, and an optional design-information coverage
guard; local tuning still failed the rare-prevalence rank gate and did not
clear the stricter design-coverage guard, so it also remains opt-in only. A
signed posterior-mean variant with medium/high design-information mean guards
was also tested; it improved some coverage metrics but transferred rare-rank
mean in the wrong direction on independent SBC data, so it remains
experimental and should not be submitted to LUMI. A crossfit/multi-holdout
rank-training variant added fold-stability gating for the signed correction,
but local SBC still missed the rare-rank and design-information gates; it also
remains experimental and should not be submitted to LUMI. A guarded
rare-balanced calibration head can fit rare-only residual mean corrections from
additional rare-prevalence simulations, but local validation selected zero
shrinkage across tested rare pools, so it is currently diagnostic only. The
rare-head metadata now records candidate offsets, shrinkage-grid validation
scores, rare-pool prevalence summaries, and rare-pool residual/rank summaries
to guide the next simulation-design revision. A stratified rare-calibration
pool with intercept-shift, low-detection, and small-sample regimes can now
produce nonzero rare-head offsets, but independent SBC rare-rank error
worsened locally. Nonzero rare-head offsets therefore now require an
independent rare-head validation gate before application. The benchmark runner
exposes `--rare-validation-datasets`; failed independent validation keeps the
candidate offsets in diagnostics but resets the selected rare-head shrinkage
and applied offsets to zero. The first local independent-gate run rejected the
stratified rare-head candidate because the independent rare-validation pool did
not clear absolute coverage floors, so this path remains local-only and should
not be submitted to LUMI. A subsequent scale-side correction stores
independent rare-validation design-stratum multipliers under
`rare_validation_scale`. This correction is separate from mean offsets and is
accepted only when the independent rare-validation pool clears coverage floors
without material rank degradation. The first local run fixed the
rare-validation coverage failure but over-inflated in-domain uncertainty, so it
also remains local-only until an in-domain overcoverage/rank-variance guard is
added. The constrained follow-up added a support-excess activation and
in-domain overcoverage/rank-variance guard. That guard correctly rejected the
always-on scale behavior, but the support-excess-only activation was not
discriminative enough to recover rare-validation coverage before the in-domain
guard failed. The next revision should use a stronger low-detection or
small-sample regime proxy rather than a global design-stratum scale. The next
local revision added such a proxy by combining support excess,
rare/intermediate prevalence by design-information stratum, and low community
occupancy from the observed response matrix. This proxy identified the
rare-validation stress context, but the same positive design-stratum multiplier
still over-inflated in-domain high-design coefficients, so selected shrinkage
remained zero. Future work should change the correction shape, not just the
activation. The follow-up changed the shape to thresholded support-excess and
low-community activations that are zero at the in-domain thresholds and grow
only outside them. That selected a nonzero rare-validation scale locally while
preserving the in-domain guard, but OOD coverage remained below target, so the
next work item is OOD refitting/interaction with the learned OOD objective. An
OOD-focused local check showed that simply raising the OOD inflation cap and
using more OOD calibration batches did not improve effect-size or combined
coverage. Pure effect-size shift barely activates the low-community scale, so
the next OOD revision should make the learned OOD objective final-multiplier
aware and explicitly gate effect-size/combined-shift coverage. That
final-multiplier-aware refinement is now implemented as a second OOD fitting
pass after rare-validation scale selection, but the local sanity gate still
fails: effect-size coverage is `0.8256` and combined-shift coverage is
`0.8036`. Final-multiplier/effect-gate diagnostics are now recorded by OOD
regime under `ood_objective.final_multiplier_diagnostics`. The production-like
diagnostic run showed that pure effect-size shift under-inflates the middle
effect-signal stratum, while combined shift still under-covers high-effect
coefficients despite large final multipliers. The next revision should replace
the current global support/effect curve with a more discriminative
effect-shift-specific scale head or domain-classifier-gated multiplier. An
experimental context-gated effect-shift head is now implemented with separate
pure-effect and combined-shift log-scale components plus differentiable
effect-quantile coverage losses. The production-like local gate showed partial
improvement, raising effect-size OOD coverage from `0.8256` to `0.8656` and
combined-shift coverage from `0.8036` to `0.8333`, but it still does not meet
the `0.90` OOD floor and increases in-domain extra-inflation penalties. The
effect-shift head is now constrained with fixed branch-specific log-amplitude
caps, a pure high-effect taper, and a combined-shift support-excess activation
gate. The constrained variant retained partial OOD gains but did not qualify:
effect-size OOD coverage was `0.8621`, combined-shift coverage was `0.8293`,
and in-domain extra-inflation penalties remained high. The next revision should
use post-fit selection or shrinkage to accept head offsets only when OOD
effect-quantile gains clear explicit thresholds without breaching in-domain
gate limits. That post-fit selector is now implemented: it evaluates a head
amplitude shrinkage grid, stores the candidate decisions in
`ood_objective.final_multiplier_diagnostics.effect_shift_head_selection`, and
the production-like local sanity gate selected shrinkage `0.0`. The selected
candidate still failed OOD floors, with effect-size coverage `0.8481` and
combined-shift coverage `0.8168`. The next revision separated the post-fit
selector into independent pure-effect and combined-shift branches. Pure-effect
shrinkage is accepted only against the `effect_size_shift` validation domain;
combined-shift shrinkage is accepted only against the `combined_shift`
validation domain. The selector records branch-specific shrinkage, branch
acceptance flags, domain coverage gains, and in-domain gate deltas under
`effect_shift_head_selection` with kind
`post_fit_independent_effect_shift_head_selection`. A tiny metadata smoke run
confirmed the independent selector shape. The production-like local sanity run
accepted the pure-effect branch at shrinkage `0.5` but rejected the
combined-shift branch, yielding held-out coverage `0.8507` for effect-size
shift and `0.8175` for combined shift. This is still below the `0.90` OOD
floor, so the candidate remains local-only. The next revision should stop
tuning post-fit selection and redesign the combined-shift calibration path
itself. That redesign is now implemented as a separate serialized
`combined_shift_scale` head. It applies a bounded support-excess-by-effect
log-scale multiplier after the learned OOD curve and rare-validation scale, and
accepts nonzero amplitude only when held-out `combined_shift` coverage reaches
`0.90`, improves by at least `0.005`, and preserves in-domain gate deltas. A
tiny metadata smoke run confirmed the block and selected zero amplitude when no
combined-shift gain was available. The production-like local sanity run also
selected zero amplitude: nonzero candidates could improve diagnostic
combined-shift coverage, and large amplitudes could exceed the `0.90` combined
coverage floor, but every nonzero candidate violated the in-domain gate. This
path remains local-only; the next revision should make the combined-shift
correction more selective rather than globally applied across the combined
regime. That selective revision is now implemented by adding low-design and
low-community gates to the combined-shift scale activation while keeping the
same held-out combined-coverage and in-domain gate checks. A tiny metadata
smoke confirmed the new activation fields. The production-like local sanity run
still selected zero amplitude: the selective gate reduced in-domain penalty
growth, but the strongest candidate reached only `0.8561` diagnostic
combined-shift coverage and still violated gate limits. This path remains
local-only; the next revision should target combined-shift effect quantiles
directly rather than using one selective scalar amplitude.

Coefficient and predictive calibration remain separate:

- coefficient artifacts may use the conditional version 5 calibrator
- predictive-only artifacts continue to use the scalar version 2 calibrator
- neither ecological dataset nor an MCMC posterior may be used to fit the
  conditional head

## Running

The dedicated entry point forwards all standard benchmark arguments and
selects conditional coefficient calibration:

```bash
python examples/run_neural_hmsc_conditional_calibration.py \
  --output run/conditional \
  --suite probit \
  --n-sites 40 \
  --n-species 75 \
  --train-datasets 512 \
  --calibration-datasets 128 \
  --sbc-datasets 128 \
  --sbc-draws 512 \
  --epochs 120
```

The general runner exposes the same mode through
`--coefficient-calibration conditional`. Optimization can be controlled with
`--conditional-calibration-epochs`,
`--conditional-calibration-learning-rate`, and
`--conditional-calibration-regularization`. Rank weighting is controlled by
`--conditional-calibration-rank-penalty-weight` and the three prevalence
weights. Support fallback is controlled by
`--conditional-calibration-support-quantile` and
`--conditional-calibration-fallback-strength`.

The learned OOD objective is opt-in:

```bash
python examples/run_neural_hmsc_conditional_calibration.py \
  --output run/conditional_ood_objective \
  --suite probit \
  --conditional-calibration-ood-objective support_excess_rank_coverage \
  --conditional-calibration-ood-datasets 128 \
  --conditional-calibration-ood-objective-epochs 200 \
  --conditional-calibration-ood-uncertainty-max-multiplier 8 \
  --ood-regimes covariate_shift effect_size_shift combined_shift
```

The OOD calibration batches are separate from the SBC batches. The same runner
still writes predictive-only artifacts with scalar version 2 calibration.

## Validation State

Unit coverage verifies conditional prevalence response, nominal calibration,
metadata round trips, version 3/4 compatibility, rank-loss improvement, scalar
fallback outside feature support, posterior-mean shift detection, domain
rejection, unchanged means, exact full-covariance transformation, and version 7
learned OOD-objective metadata/application, including legacy support-only v7
metadata, plus version 8 gated and constrained effect-size
metadata/application. The post-scale final-multiplier-aware OOD refinement is
implemented and covered by the conditional calibration/public API/workflow test
set, but it is not locally qualified for LUMI submission because effect-size and
combined-shift OOD coverage remain below gate. Final-multiplier diagnostics are
covered by the same test set and verified in a benchmark metadata smoke run. The
experimental effect-shift head is also covered by metadata round-trip tests and
a benchmark metadata smoke run. An
end-to-end benchmark smoke test verifies that anchored coefficient artifacts
carry version 5 metadata while predictive artifacts remain on version 2 and
SBC rows expose support-trust, mean-magnitude, and effect-size signal
diagnostics.

The frozen five-seed in-domain/OOD comparison is recorded in
`docs/neural_hmsc_conditional_comparison_2026-07-02.md`. The implementation
fixed overall in-domain rank variance but failed rare-prevalence and OOD gates.
That result applies to the version 3 objective. The version 4 comparison is
recorded in `docs/neural_hmsc_rankaware_v4_comparison_2026-07-09.md`. Version 4
fixed rare coverage and recovered most OOD degradation, but prevalence
rank-mean and intercept rank-variance gates still failed. It is not qualified.
The IRLS/Laplace anchor, version 6 support-excess inflation, and version 7
learned OOD objective are implemented. Version 7 still needs a five-seed LUMI
comparison against scalar, version 4, version 5 IRLS, version 6 default, and the
conservative version 6 sweep candidate. Later local version 8 experiments are
recorded in `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`; the later
combined-shift path now includes a selective scale gate and an
effect-bin-specific combined-shift selector. The effect-bin selector is covered
by the conditional calibration/public API/workflow test set and by a benchmark
metadata smoke run. The production-like local sanity workflow selected zero
effect-bin amplitude and left held-out coverage unchanged from the selective
scalar combined-shift run, so this branch remains local-only. The follow-up
context-gated combined-shift selector is implemented and covered by the same
test set plus a metadata smoke run. It serializes a classifier-style
support/effect/low-design/low-community context gate and rejects nonzero
candidates with excessive in-domain context overlap. It still needs a
production-like local sanity workflow before any LUMI comparison. The
production-like context-gated run selected zero amplitude and left held-out
coverage unchanged from the selective scalar and effect-bin runs, so this
branch also remains local-only. The next revision should move combined-shift
coverage pressure into the learned OOD objective itself rather than adding
another post-scale selector. That revision is now implemented in the
final-multiplier-aware OOD fitting path: the learned combined branch receives
direct combined-shift coverage, effect-quantile coverage, and context-weighted
coverage losses, while the in-domain gate penalizes learned combined-context
overlap and context-weighted extra inflation. It is covered by the test suite
and a metadata smoke run. The production-like local sanity workflow reduced
diagnostic in-domain gate penalties but worsened held-out effect-size and
combined-shift coverage, so it remains local-only. The next revision should
rebalance or stage the combined-shift-aware objective so OOD coverage is
recovered before applying the in-domain overlap constraint. The staged schedule
is now implemented in the final-aware OOD refit: early epochs boost combined
coverage terms while down-weighting the in-domain gate, then later epochs ramp
the in-domain gate and combined-context overlap penalty back to full strength.
It is covered by the test suite and a metadata smoke run. The production-like
local sanity workflow still failed OOD coverage, so this objective family
remains local-only. The next revision should add a new representation or data
split for combined shift, such as separate pure-effect and combined-shift
experts with held-out expert selection, rather than tuning the same combined
objective again.

That replacement path is now implemented as a held-out domain-expert OOD
selector. After the shared final-multiplier-aware OOD refit, the calibrator
fits separate pure-effect and combined-shift expert candidates on
domain-specific OOD calibration splits, evaluates target coverage gains against
in-domain gate-delta thresholds, and serializes the audit under
`ood_objective.final_multiplier_diagnostics.domain_expert_selection`. The
implementation deliberately preserves the existing OOD parameter vector format;
if neither expert passes the held-out gate, calibration keeps the baseline OOD
parameters. The conditional calibration/public API/LUMI workflow tests pass,
and `/private/tmp/neural_hmsc_domain_expert_smoke` confirms the metadata
plumbing. The next validation step is a production-like local sanity run before
any LUMI comparison.

The domain-expert selector now also evaluates a baseline-to-expert shrinkage
grid and records per-shrinkage diagnostics. The production-like rare32 local
sanity run still selected the baseline: useful OOD gains appeared only after
the in-domain extra-inflation gate was exceeded. This indicates that post-fit
linear shrinkage is too blunt; the next implementation should put a
trust-region or overlap regularizer into expert fitting itself.

The first trust-region expert objective is implemented and locally tested. It
penalizes in-domain OOD log-inflation drift from the pre-expert baseline during
expert fitting, then still applies the shrinkage selector. The rare32 local
sanity run showed the penalty reduced expert aggressiveness but did not qualify
any shrinkage point; the next work should tune the trust-region strength or
localize the overlap penalty to in-domain contexts that resemble the OOD expert
domain.

The trust-region path now includes a small fitting-time sweep over moderate,
strong, and tight in-domain log-inflation constraints. A compact tuning run
accepted the combined-shift expert under all three settings, with the tightest
constraint producing the smallest in-domain extra-inflation delta. This is only
a selector-tuning result; the next validation step is the production-like
rare32 local sanity workflow with the sweep enabled.

The rare32 workflow with the trust-region sweep accepted a boundary pure-effect
candidate internally, but final OOD coverage did not improve. The global trust
region is therefore not sufficient. The next implementation should use a
domain-localized overlap penalty and a stricter expert-selection gate requiring
practical held-out gains without OOD-domain degradation.

The domain-localized overlap penalty and stricter gate are now implemented. A
compact tuning check rejected all candidates: localized overlap and
non-target-domain degradation were controlled, but target gains stayed below
the new practical-gain floor. The next work should strengthen the expert's
target-domain coverage objective before another production-like rare32 run.

Adding scalar target-domain and effect-quantile coverage pressure did not
increase compact tuning gains. The bottleneck now appears to be the expert
parameterization rather than objective weighting; the next work should add
effect-bin-specific or target-domain-specific expert parameters before another
rare32 validation.

The first representation change is implemented as an effect-bin-specific
domain-expert head. New OOD objective fits emit a 14-parameter effect-shift
head: the original eight pure-effect/combined-shift context-gate parameters
plus three localized effect-bin log-amplitudes for each expert. The support
metadata records `parameter_count`, `effect_bin_centers`, `effect_bin_width`,
`pure_effect_bin_log_amplitudes`, and
`combined_effect_bin_log_amplitudes`; legacy 8-parameter heads remain loadable.
A compact tuning run in
`/private/tmp/neural_hmsc_v8_domain_expert_bin_head_tuning` accepted the
combined-shift expert under the strengthened held-out gate, with target gain
`0.0222` and no non-target degradation. The pure-effect expert remained below
the practical-gain threshold. Next validation is a production-like rare32 local
sanity workflow before any LUMI comparison.

The first follow-up constrains the effect-bin head without changing its
serialized shape. Pure-effect bin amplitudes are now capped by support-excess
or low-design context, and combined-shift bin amplitudes are capped by
support-excess, low-design, and optional low-community context when `Y` is
available. Expert fitting also adds per-effect-bin in-domain penalties for
extra log inflation, rank-mean drift, and low coverage in each active bin
context. Compact tuning in
`/private/tmp/neural_hmsc_v8_domain_expert_context_capped_bin_tuning` showed
the caps controlled in-domain leakage but were too conservative: the best
combined-shift target gain fell to `0.0111`, below the practical-gain floor.
Do not run rare32 from this checkpoint; next work should tune the cap shape or
use a two-stage target-then-projection fit.

The first cap-shape tuning pass added fixed target-domain floors to the same
context caps (`0.20` for pure-effect contexts and `0.35` for combined-shift
contexts). Compact tuning in
`/private/tmp/neural_hmsc_v8_domain_expert_soft_floor_bin_tuning` increased the
learned combined-bin amplitudes but did not improve selected held-out coverage:
the selector still chose `baseline`, and the best combined-shift target gain
remained `0.0111`. Do not run rare32 yet; the next design step is a two-stage
target-then-projection expert fit.

The two-stage target-then-projection profile is now implemented in the
held-out domain-expert selector. Stage 1 fits a target-domain expert with
reduced in-domain/bin constraints and moderate target/effect-quantile pressure.
Stage 2 evaluates the learned expert through the existing selector gates after
crossing the standard shrinkage grid with branch-specific projection caps
`(0.25, 0.5, 0.75, 1.0)`. Projection caps shrink only the active branch's
effect-head scalar amplitude and three effect-bin amplitudes toward the
baseline vector. The next step is a compact local tuning run before any rare32
workflow.

The compact two-stage projection run in
`/private/tmp/neural_hmsc_v8_domain_expert_two_stage_projection_tuning` did not
qualify. The selector kept `baseline`; the best two-stage pure-effect target
gain was `0.0037`, and the best two-stage combined-shift target gain was
`0.0111`. Projection reduced some combined-shift overlap leakage, but did not
move enough held-out coefficients across the interval boundary. Do not run
rare32 from this checkpoint. The next revision should change the target signal
or compact calibration pool, likely via a larger/harder target-domain pool or a
margin-aware OOD loss for near-miss coefficients before projection.

The margin-aware target-signal variant is implemented as a positive
`margin_weight` only on the `two_stage_target_then_projection` selector
profile. It targets coefficients that are just outside the baseline nominal
interval, then still relies on the existing projection and held-out selector
gates. Compact tuning in
`/private/tmp/neural_hmsc_v8_domain_expert_margin_projection_tuning` was
effectively unchanged from the prior two-stage projection run: selected expert
remained `baseline`, best pure-effect target gain remained `0.0037`, and best
combined-shift target gain remained `0.0111`. Do not run rare32. The next
revision should change the compact calibration pool itself, enriching for
near-boundary target-domain misses and separating pure-effect from
combined-shift regimes before fitting the same projection selector.

The hard target-domain OOD calibration pool is now implemented in the benchmark
runner behind opt-in CLI flags. It keeps default behavior unchanged, but can
oversample `effect_size_shift` and `combined_shift` candidates, score them by
near-boundary coefficient misses under the current posterior, and retain the
hardest subset for the same projection selector. Compact tuning in
`/private/tmp/neural_hmsc_v8_domain_expert_hard_pool_projection_tuning` moved
target gains in the right direction: best pure-effect target gain increased to
`0.0185`, and best combined-shift target gain increased to `0.0160`. The
selector still chose `baseline` because both gains stayed below the `0.0200`
floor and max group extra-cap deltas stayed around `0.091`, above the `0.080`
gate. Do not run rare32 yet; next work should refine hard-pool/projection
interaction locally.

The first refinement emits hard target-domain datasets as separate calibration
batches, making the expert selector use `alternating_batches` train/evaluation
splits instead of one within-batch split. This controlled the extra-cap gate:
split x3 reduced max group extra-cap deltas to `0.0075` for pure-effect and
`0.0112` for combined-shift. It did not qualify because target gains fell to
`0.0111` and `0.0173`, respectively. Increasing hardness to x4 worsened the
held-out target gains. Do not run rare32; next work should make hard-pool
selection gate-aware or build matched train/validation hard pools.

Gate-aware hard-pool scoring is now implemented. It subtracts a regime-specific
overlap proxy from near-boundary target-domain candidate scores. Compact tuning
in `/private/tmp/neural_hmsc_v8_domain_expert_gate_aware_hard_pool_tuning`
improved held-out OOD diagnostics and kept extra-cap deltas controlled, but the
best target gains still stayed below the `0.0200` floor (`0.0148` for
pure-effect and `0.0173` for combined-shift). Do not run rare32. The next
revision should build matched train/validation hard pools with similar
near-boundary difficulty.

Matched hard-pool grouping is now implemented by balancing the two
target-domain calibration batches on the gate-aware near-boundary score.
Compact tuning in
`/private/tmp/neural_hmsc_v8_domain_expert_matched_hard_pool_tuning` did not
qualify and worsened held-out OOD coverage relative to gate-aware x3. The best
pure-effect target gain stayed `0.0148`, and the best combined-shift target
gain fell to `0.0148`. Do not run rare32. Next work should instrument the
hard-pool selection path with train/evaluation score and overlap summaries
before changing the heuristic again.

The hard-pool instrumentation is now in place. Target hard-pool benchmark
records include candidate and selected score distributions, raw near-boundary
scores before overlap penalty, overlap proxies, miss summaries, and matched
train/evaluation group summaries by regime. The compact instrumentation check
in `/private/tmp/neural_hmsc_v8_hard_pool_instrumentation_check` reproduced the
matched-pool baseline selection and showed that selected hard pools still carry
substantial overlap penalties: effect-size-shift selected score mean `-0.0485`
with overlap mean `0.2815`, and combined-shift selected score mean `-0.0293`
with overlap mean `0.2458`. The combined-shift matched train/evaluation score
delta was also larger (`0.1345`) than effect-size-shift (`0.0605`). Do not run
rare32. Next work should inspect the diagnostic arrays directly and redesign
hard-pool construction to require both enough raw near-boundary misses and low
target-domain overlap, matched separately for each target regime.

Constrained hard-pool construction is now implemented. It separates raw
near-boundary difficulty from overlap, relaxes overlap before raw difficulty if
the eligible set is too small, and matches train/evaluation hard-pool batches
by raw difficulty, overlap, and final score. The compact check in
`/private/tmp/neural_hmsc_v8_constrained_hard_pool_check` did not qualify:
baseline stayed selected, mean held-out OOD fell to `0.7525`, and worst
held-out OOD fell to `0.7457`. The matcher worked as intended, reducing target
group score deltas to `0.0132` for effect-size shift and `0.0107` for combined
shift, but both regimes had to relax the overlap threshold to the `0.95`
quantile to preserve enough raw near-boundary misses. Do not run rare32. The
next change should target candidate-pool generation: generate or oversample
low-overlap target-domain candidates before applying the constrained selector.

Low-overlap candidate-pool generation is now implemented in the benchmark
runner. Target regimes generate a wider seed window, prefilter that generated
pool by low overlap while keeping a raw near-boundary miss floor, and then apply
the constrained hard-pool selector. The compact check in
`/private/tmp/neural_hmsc_v8_low_overlap_candidate_pool_check` improved
combined-shift overlap and in-domain extra-inflation penalties, but still did
not qualify: baseline remained selected, mean held-out OOD was `0.7562`, and
worst held-out OOD was `0.7519`. Combined-shift candidate-pool overlap mean
fell from `0.2683` to `0.2377`, selected combined score mean improved to
`-0.0184`, extra-inflation loss improved to `0.1339`, and max extra-cap loss
improved to `0.1060`. Do not run rare32. The next implementation should change
the simulated target-domain candidate distribution itself, such as adding a
context-controlled low-overlap OOD simulator variant that creates more
low-overlap hard misses before selection.

The context-controlled low-overlap candidate simulator is now implemented for
hard target calibration candidate generation. Default OOD simulations are
unchanged. The compact check in
`/private/tmp/neural_hmsc_v8_context_controlled_low_overlap_check` shows that
the candidate distribution change solved the pool-quality issue but not expert
acceptance. Baseline remained selected, but mean held-out OOD improved to
`0.7969`, effect-size-shift coverage improved to `0.8358`, and combined-shift
coverage was `0.7580`. Effect-size candidate overlap fell sharply: generated
overlap mean `0.1308`, pool overlap mean `0.1050`, and selected score mean
`0.0347`. Combined-shift overlap also improved: generated overlap mean
`0.2311`, pool overlap mean `0.1970`, and selected score mean `-0.0012`. The
remaining blocker is not candidate quality; it is expert fitting and
acceptance. Best pure-effect target gain was `0.0173` with extra-inflation
delta about `0.3090`; best combined-shift target gain was `0.0136` with
extra-inflation delta about `0.2643`. Do not run rare32. Next work should keep
the context-controlled candidate distribution and redesign expert
fitting/acceptance to reduce in-domain extra inflation and improve
combined-shift target gain.

Gate-compatible expert projection is now implemented. The domain-expert
selector uses a finer shrinkage grid, stricter projection caps, a
combined-shift-specific target-loss profile, and selects the best
gate-compatible projection row when no candidate clears the full target-gain
floor. The compact check in
`/private/tmp/neural_hmsc_v8_gate_compatible_projection_check` kept baseline
selected, with held-out OOD unchanged from the context-controlled candidate
run. Candidate diagnostics improved on risk control but lost target gain: best
pure-effect target gain was `0.0136` with extra-inflation delta `0.2369`, and
best combined-shift target gain was `0.0099` with extra-inflation delta
`0.1837`. Do not run rare32. The next implementation should move the
extra-inflation constraint into expert fitting itself, likely via an
in-objective gate-compatible amplitude penalty or combined-shift curriculum, so
combined-shift gain can recover while staying below the extra-inflation gate.
