# Neural-HMSC v0.1 release-readiness audit

Date: 2026-07-21

Decision: **release-ready for the narrowed fixed-shape probit scope**.

This audit did not fit, tune, or select a model. It validated the frozen
`neural_predictive_affine_v1` bundle and retained three-seed evidence, then ran
the public `NeuralHmscInference` API against a newly compiled fixed-effect
artifact. The numerical probit approximation passes its frozen envelope, and
checkpoint version `0.4` now packages and applies the qualified coefficient
calibration without fitting or selecting another model.

## Gate summary

| Gate | Result |
| --- | --- |
| Frozen artifact and provenance validation | pass |
| Public checkpoint load and compiled-artifact compatibility | pass |
| Posterior emission, `HmscFit` summaries, and prediction | pass |
| Unsupported trait artifact rejected | pass |
| Qualified coefficient calibration packaged with checkpoint | pass |
| Probit production-shape SBC envelope | pass |
| Gaussian production-shape SBC evidence | **not qualified** |
| Poisson production-shape SBC evidence | **not qualified** |
| Whittaker and Big Spatial predictive envelope | pass |
| Ecological inference-only speed envelope | pass |

The narrowed v0.1 decision requires the probit, API, provenance, ecological,
and runtime gates. The Gaussian/Poisson rows record explicit release-scope
exclusions and do not block a probit-only release.

## Public API smoke

The audit loaded frozen seed `20260721`, compiled the 40-site, 75-species
Whittaker training boundary as fixed-effect probit `presence ~ TMG`, and
successfully produced:

- `Beta` draws with shape `(1, 8, 2, 75)`;
- `beta_mean()` and `beta_ci()` outputs with shape `(2, 75)`;
- response predictions with shape `(40, 75)`;
- an HDF5 posterior readable as an ordinary `HmscFit`.

A matching compiled artifact with traits was rejected with the public
compatibility error. The checkpoint manifest correctly advertises fixed shape,
fixed-effect `Beta` only and excludes traits, phylogeny, iid/spatial latent
effects, random effects, and detection submodels.

The version `0.4` manifest binds `coefficient_calibration.json` by SHA-256 and
records its method, parameter, distribution, dimensions, and coefficient names.
`NeuralHmscInference.load()` validates those fields, the canonical calibration
metadata hash, and independent-simulation provenance before inference. The
public API then applies the frozen calibration by default; `calibrated=False`
retains explicit access to the raw amortizer for diagnostics. In the audit
fixture the mean calibrated-to-raw scale ratio was `2.778770`, confirming that
the emitted posterior is not the raw checkpoint posterior.

## Simulation evidence

| Distribution | Public implementation | Release qualification |
| --- | --- | --- |
| Probit | fixed-shape public API | release-qualified |
| Gaussian/Normal | fixed-shape public API | experimental; no retained production-shape release SBC evidence |
| Poisson | fixed-shape public API | experimental; no retained production-shape release SBC evidence |

Production-shape calibrated probit SBC:

| Seed | Coverage 95 | Rank-mean error | Rank-variance error | Beta RMSE | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 20260721 | 0.955937 | 0.007999 | 0.013018 | 0.460860 | pass |
| 20260722 | 0.952865 | 0.007322 | 0.012158 | 0.459983 | pass |
| 20260723 | 0.956198 | 0.001824 | 0.014322 | 0.461182 | pass |

The frozen thresholds are coverage in `[0.925, 0.975]` and rank mean/variance
errors no greater than `0.025`. Beta RMSE is reported diagnostically; v0.1 does
not define an MCMC posterior-mean equivalence threshold.

## Ecological evidence

| Dataset | Neural Brier | MCMC Brier | Ratio | Neural log loss | MCMC log loss | Ratio | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Whittaker | 0.075542 | 0.073968 | 1.021279 | 0.270509 | 0.261936 | 1.032730 | pass |
| Big Spatial | 0.051218 | 0.047468 | 1.078996 | 0.205443 | 0.191443 | 1.073132 | pass |

Both ratios remain below the v0.1 `1.10` bounded-approximation threshold.
Qualified Python MCMC remains the statistical reference and is better on all
four proper scores. This evidence does not establish joint-posterior or full
HMSC equivalence.

## Runtime evidence

Three-seed means:

| Dataset | Checkpoint training | Neural inference | MCMC | Inference speedup | Break-even |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whittaker | 448.289 s | 0.145449 s | 37.173 s | 256.0x | 12.5 datasets |
| Big Spatial | 448.289 s | 0.152711 s | 32.241 s | 211.2x | 14.0 datasets |

The training cost is the shared Whittaker-shape checkpoint cost reused for the
transfer evaluation. The result supports repeated amortized inference, not a
claim that training plus one inference is faster than one compact MCMC fit.

## Hash ledger

Frozen bundle and evidence:

| Artifact | SHA-256 |
| --- | --- |
| `baseline.json` | `858e6843a29c462eeb5dbc8299112293fe416278fc5a9e9f97eb65944f5bff36` |
| Whittaker affine manifest | `ec14e540496da16a8990c580022ebbfa2371fe3b27d3cf218c533a7dda733aa2` |
| Whittaker scale-only manifest | `e0f08cd96f1727acf24cbc2009d132a56265ef2ceaf1905d6181be3711c770a0` |
| Big Spatial affine manifest | `903f04b9ed66908f19c6dfd6c7f47c41bee2e7f75648373d0255fadb1dd9c51f` |
| Big Spatial scale-only manifest | `af3a0a202b3ecb7585c35818e38d09b853329e521443857e38b6cdbc4ef3aa54` |
| API requalification report | `c7e1e6a121e615d6496631edd74eec5ffa855299522a9648cd983eb4a7c65071` |
| Default-wiring smoke report | `8c5f289375be3df9771a07bee5e4b9094adc8c703aef3b2bc56dac8fb7f0e8c6` |
| Whittaker R/Python parity report | `1329a9ae3da335a334cb09b58b792e2b3391cd73ae894d93a7353210b0607435` |
| Big Spatial R/Python parity report | `a52f2d37d05e8bcd4d7454b8e28d330e6a66b75b5f0363f790d4b3bedb0df88c` |

Affine predictive members:

| Dataset | Seed | SHA-256 |
| --- | ---: | --- |
| Whittaker | 20260721 | `e45c4907a68438e755488a830ef542089b645b85366f108102441e4920d6872a` |
| Whittaker | 20260722 | `20ba7981d7c7d047542de942ef87df8839bb2f5d9734a2fb09662854215d90a1` |
| Whittaker | 20260723 | `e48fee74a7f5ca4339ca712ce67bee10c9bdbdb73a39e38e466928a6dfff1e9a` |
| Big Spatial | 20260721 | `12a108f22e6128fdd6bda41c4cc480b07bd85ae0a5207e8181cdf8240474756e` |
| Big Spatial | 20260722 | `6c8456eafe7e690586c7710f4f3a04a5b52b071085f638670b3644036d188ce5` |
| Big Spatial | 20260723 | `9b7a1b82974cf9f5f427e05fd08c4ee5e68f0a014e34020eede1c0aad2e04fa5` |

Packaged checkpoint bundle:

| Artifact | Seed 20260721 | Seed 20260722 | Seed 20260723 |
| --- | --- | --- | --- |
| Checkpoint manifest | `f62cd2217df6cc71cbe9f915c0cfbd3a3327b6684b3c5452bd9399aa130133a8` | `49939deb832a4280a36d2149b445c128af9f5594d2997511cc25d97fb6b6cc08` | `9b695200d0c47906206b8b3f61e752658d68106d620a186275a3bfd58a23b136` |
| Calibration artifact | `595fc0796d36802002cee09b270d53162f1fce100b83aecd32476e0958a0fd94` | `172ad83e32b78eba71fb1e83ae7972b24d30a86d35450928d176524b3df7aceb` | `1c0739e82ff90182f5b38db9432920bf6dbacac311bd5963ef37175d33318a5a` |
| Unchanged weights | `bb6e76d3ec9bc5e294ceac3051c3b2d7e5273db5053cfa5ceac676913d6265d9` | `6b103d3bc78c7ef97702b391521fe5f18349dbe6b7889e7de562ae312e068bca` | `85b3612442f1041fcc099b0d35d271ec14fa0260efc1e686b46e86e4d79d2245` |

The complete package manifest SHA-256 is
`d2daa81ec841390df59324a208216ffa0032ac514e6c679649d98815490bdbc7`.
It records `model_fitting_performed=false`,
`calibration_selection_performed=false`, and byte-identical source/packaged
weight hashes for all three seeds. It also includes the byte-identical
`affine_branch` and `scale_only` predictive artifacts for both named datasets.

SBC reports for seeds `20260721`, `20260722`, and `20260723` have SHA-256
`6b92f5196dedbcc72cc0b1e0eea545073d0ad96bd2ff2785d23a9293587a61ef`,
`566dc175603a918d3077095a87113730c2eb098246a164195e200984fd033764`,
and `f15c50c0bd634a2f82c8ee931e7b4f989f2df26b3bbd666d254c19372f350fa5`.

The machine-readable re-audit was generated at
`/private/tmp/neural_hmsc_v0_1_release_audit_complete_20260721/` with JSON
SHA-256
`9b6392b4ca45f91c3bd4979656712a6ce4e0d1a1573eb753f8cc65eb6e4a188f`
and Markdown SHA-256
`5d9638cf823667d9e417ed9dee02251ae9f9b161fdcac4125c005eb6fdeb6f9f`.

## Support boundary

| Capability | v0.1 status |
| --- | --- |
| Fixed-shape fixed-effect probit `Beta` inference | release-qualified under the bounded v0.1 envelope |
| Whittaker/Big Spatial manifest ensemble | qualified predictive-only artifact for these named datasets |
| Gaussian and Poisson fixed-effect paths | implemented, experimental |
| Variable site/species shape | prototype, not public |
| Traits and phylogeny | prototype, rejected by public checkpoint API |
| iid/spatial latent effects and random slopes | prototype, rejected by public checkpoint API |
| Detection, GPP, and NNGP | unsupported |
| Full Bayesian or joint-posterior MCMC equivalence | not provided or claimed |

## Reproduction

```bash
python examples/package_neural_hmsc_v0_1_checkpoints.py \
  --source-members-root /private/tmp/neural_predictive_affine_v1_members \
  --sensitivity-root /private/tmp/neural_hmsc_source_transfer_realdata_sensitivity_20260720 \
  --output-root /private/tmp/neural_hmsc_v0_1_packaged_members_complete

python examples/audit_neural_hmsc_v0_1_release.py \
  --baseline-root /private/tmp/neural_predictive_affine_v1/neural_predictive_affine_v1 \
  --members-root /private/tmp/neural_hmsc_v0_1_packaged_members_complete \
  --sensitivity-root /private/tmp/neural_hmsc_source_transfer_realdata_sensitivity_20260720 \
  --requalification-json /private/tmp/neural_hmsc_probability_ensemble_api_requalification_20260720/neural_hmsc_probability_ensemble_api_requalification_20260720/probability_ensemble_comparison.json \
  --output /private/tmp/neural_hmsc_v0_1_release_audit_complete_20260721
```

## Decision and next step

The narrowed probit v0.1 release gates pass. Qualified Python MCMC remains the
statistical reference, and Gaussian/Poisson remain experimental. The complete
bundle is frozen under `neural_hmsc_v0_1`; see
`docs/neural_hmsc_v0_1_release.md`. The next roadmap step is variable-shape
fixed-effect inference under Milestone 52.
