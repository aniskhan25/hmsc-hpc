# Outcome-Blind MCMC-Teacher Real-Context Routing

Date: 2026-07-21

## Decision

The frozen routing check failed closed. Whittaker routed to exact identity as
required, but Big Spatial did not route to an approved target/effect context
and was outside the selected support cap. Held-out ecological responses and
MCMC predictions were not opened, no proper scores were computed, and paired
real-data scoring remains blocked.

## Method

`examples/check_neural_hmsc_teacher_context_routing.py` loads the exact frozen
`neural_predictive_affine_v1` affine manifests, verifies the ordered local
member hashes and compatibility signatures, averages response probabilities,
and then evaluates the selected cross-fitted teacher gate from baseline
probabilities and held-out `X.csv` covariates only. The harness accepts no
response path and records its outcome-blind input contract in JSON.

The run used the three frozen members `20260721`, `20260722`, and `20260723`
for both datasets. Whittaker used 12 held-out sites and Big Spatial used 360
held-out sites, each with 75 species.

## Routing Result

| Dataset | Required | Selected | Active | Approved distance / cap | Movement | Pass |
|---|---|---|---:|---:|---:|---:|
| Whittaker | identity | rare validation | false | 5.0709 / 3.0000 | `1.11e-16` | true |
| Big Spatial | approved target/effect | in distribution | false | 11.9382 / 3.0000 | `1.11e-16` | false |

Whittaker's nearest fallback prototype was rare validation at distance
`1.5277`; its nearest approved prototype was effect-size shift at `5.0709`,
outside the `3.0` cap. The resulting teacher prediction was numerical identity.

Big Spatial's nearest fallback prototype was in distribution at distance
`10.7183`; its nearest approved prototype was effect-size shift at `11.9382`,
also outside the `3.0` cap. The gate therefore applied exact identity.

## Diagnosis

The current simulation corpus fitted and evaluated every context on 20
held-out sites. Big Spatial has 360 held-out sites. Its normalized
`mean_log_design_information` coordinate is `10.12`, compared with `0.02` for
the effect-size prototype and `1.46` for the in-distribution prototype. This
single coordinate contributes most of the support failure.

The mismatch is broader than the router. Big Spatial's response probabilities
are substantially lower and narrower than the approved prototypes:
probability mean is `0.1002`, standard deviation is `0.0801`, and the 90th
percentile is `0.2111`. The residual head itself also contains total design
information and site-count features. Overriding only the context decision
would therefore apply the residual network outside its fitted support and is
not permitted.

## Provenance

- Frozen baseline SHA-256:
  `858e6843a29c462eeb5dbc8299112293fe416278fc5a9e9f97eb65944f5bff36`
- Whittaker affine manifest SHA-256:
  `ec14e540496da16a8990c580022ebbfa2371fe3b27d3cf218c533a7dda733aa2`
- Big Spatial affine manifest SHA-256:
  `903f04b9ed66908f19c6dfd6c7f47c41bee2e7f75648373d0255fadb1dd9c51f`
- Teacher metadata SHA-256:
  `dd68ad26d46cb5667835744b080e800e19f956db41144c6fd8a8de3d291b60c4`
- Routing JSON SHA-256:
  `1c7ea836c00c89d9bd3e6ad4a29d56d2ab4361feb4a5769e90404485d8bc366a`

Retained local routing output:

`/private/tmp/neural_hmsc_teacher_context_routing_20260721`

## Verification

- `19` focused teacher, ensemble-artifact, and routing tests passed.
- The routing script passed Python compilation.
- The run verified all six local predictive-member hashes against the frozen
  manifests.
- Both result rows record `target_response_opened=false`,
  `proper_scores_computed=false`, and qualified manifest parity provenance.

## Next Step

The sample-size-stable v3 corpus and routing replay are complete, but both gates
failed closed. Sample-size extrapolation was removed; the remaining target
support mismatch is prevalence/context shape, and one independent effect-shift
seed also failed no degradation. Details are in
`docs/neural_hmsc_mcmc_teacher_sample_size_v3_2026-07-21.md`. The next step is
an outcome-blind simulation-support qualification before another MCMC/head fit,
followed by fresh compact evaluation only after target and identity contexts
are demonstrably covered.
