{
  "artifact_audit": {
    "failed_m56_negative_reference": {
      "decision": "m56_terminal_failure_reserved_evaluation_sealed",
      "freeze_sha256": "c4fcb04cf1ebd7123be12144803de319ce1ff16a31e4fc5a1fb3e224f361a526",
      "local_root": "/private/tmp/neural_hmsc_m56_train_validation_20192218",
      "lumi_root": "/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_m56_train_validation_20192218",
      "overlay_manifest_sha256": "24f7eafa4a886afab94711bab77c56e76aef726fc93c0911c372b639bfa0121d",
      "overlay_weights_sha256": "66033d4f84cd443abf94053923e929180c0307fb08ac2a1bb9eaa75fe32ccde5",
      "promotion_role": "negative_comparator_only",
      "reserved_213m_215m_opened": false
    },
    "fixed_release": {
      "bound_member": {
        "calibration_artifact_sha256": "595fc0796d36802002cee09b270d53162f1fce100b83aecd32476e0958a0fd94",
        "calibration_internal_sha256": "81041eb9075b32c4c0f848927c1feea1d49e5cdcde7fb4e3aa7c4f566865a0a4",
        "calibration_method": "external_context_monotone_scale",
        "checkpoint_manifest_sha256": "f62cd2217df6cc71cbe9f915c0cfbd3a3327b6684b3c5452bd9399aa130133a8",
        "checkpoint_version": "0.4",
        "covariate_names": [
          "Intercept",
          "TMG"
        ],
        "dimensions": {
          "n_covariates": 2,
          "n_sites": 40,
          "n_species": 75
        },
        "distribution": "probit",
        "formula": "~ TMG",
        "posterior_family": "diagonal_normal",
        "seed": 20260721,
        "weights_sha256": "bb6e76d3ec9bc5e294ceac3051c3b2d7e5273db5053cfa5ceac676913d6265d9"
      },
      "inventory_files_validated": 36,
      "inventory_has_extra_files": false,
      "manifest_sha256": "31ee489898e3657b97919803c0e850dc20494ef9118e9b963fe4a20365822e98",
      "package_manifest_sha256": "d2daa81ec841390df59324a208216ffa0032ac514e6c679649d98815490bdbc7",
      "release_content_sha256": "affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8",
      "release_id": "neural_hmsc_v0_1",
      "release_registry_local": "/private/tmp/neural_hmsc_releases/neural_hmsc_v0_1",
      "release_registry_lumi": "/scratch/project_462000131/anisrahm/hmsc-hpc-deployments/neural_hmsc_v0_1",
      "runtime_validator_passed": true
    },
    "variable_shape_baseline": {
      "baseline_content_sha256": "badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9",
      "baseline_id": "neural_hmsc_variable_probit_v1",
      "calibration_sha256": "c3c8fd4ff50583ced5273c009e501ea0b6f400ff144a74f510513633edd7b771",
      "checkpoint_manifest_sha256": "cf46ebfdfc457e71a0da28f48f7709613f7e47b101b946553f711d5e1e4f47a5",
      "inventory_files_validated": 11,
      "inventory_has_extra_files": false,
      "manifest_sha256": "b9387efc147ecd9e3978c80cb2cc2a3ebdcddd5c68445c8b63ce8f37af61a2f1",
      "multiseed_qualification_sha256": "0ba58fd9bc4d49710068881ef41d3d86010aeef988351a9c79b3b40c287e02ce",
      "registry_local": "/private/tmp/neural_hmsc_variable_deployments/neural_hmsc_variable_probit_v1",
      "registry_lumi": "/scratch/project_462000131/anisrahm/hmsc-hpc-deployments/neural_hmsc_variable_probit_v1",
      "runtime_validator_passed": true,
      "weights_sha256": "70ef4548eeb1dc3a0d9367cb8edaedb5a2030370179241f35b372aecd8d5c4cd"
    }
  },
  "audit_date": "2026-07-24",
  "decision_document_sha256": "a1a7bc4a54eca4c78f6b32537f1afff662a524557accbd99d7267a28bc2cb2ba",
  "kind": "neural_hmsc_m57_artifact_seed_audit",
  "proposed_seed_roles": {
    "disposable_evaluation_contexts": {
      "count": 27,
      "end": 392000027,
      "start": 392000001
    },
    "disposable_training_contexts": {
      "count": 27,
      "derived_response_realizations_per_context": 2,
      "end": 391000027,
      "start": 391000001
    },
    "fixed_validation_contexts": {
      "count": 324,
      "end": 322000324,
      "start": 322000001
    },
    "model_seed": 321900001,
    "reserved_evaluation_a_contexts": {
      "count": 324,
      "end": 323000324,
      "start": 323000001
    },
    "reserved_evaluation_b_contexts": {
      "count": 324,
      "end": 324000324,
      "start": 324000001
    },
    "reserved_evaluation_c_contexts": {
      "count": 324,
      "end": 325000324,
      "start": 325000001
    },
    "training_contexts": {
      "count": 324,
      "derived_response_realizations_per_context": 2,
      "end": 321000324,
      "start": 321000001,
      "training_realization_count": 648
    }
  },
  "rng_contract": {
    "free_additional_seed_ranges_permitted": false,
    "mcmc_chain_seeds": "SeedSequence children of the owning evaluation context seed and fixed protocol tags",
    "paired_training_responses": "two SeedSequence children of each owning 321M context seed",
    "posterior_draw_seeds": "SeedSequence children of the owning context seed and fixed protocol tags",
    "realdata_replay_seeds": "derived from frozen artifact hashes and fixed protocol tags",
    "validation_heldout_response_seeds": "SeedSequence children of the owning context seed and fixed protocol tags"
  },
  "schema_version": 1,
  "seed_token_audit": {
    "candidate_integer_ranges": [
      [
        321000001,
        321000324
      ],
      [
        321900001,
        321900001
      ],
      [
        322000001,
        322000324
      ],
      [
        323000001,
        323000324
      ],
      [
        324000001,
        324000324
      ],
      [
        325000001,
        325000324
      ],
      [
        391000001,
        391000027
      ],
      [
        392000001,
        392000027
      ]
    ],
    "integer_token_rule": "(?<![0-9.])[0-9]{9}(?![0-9.])",
    "local_repository": {
      "files_scanned": 6215,
      "matching_seed_tokens": 0,
      "root": "/Users/anisr/Documents/hmsc-hpc"
    },
    "local_retained_evidence": {
      "files_scanned": 1673,
      "matching_seed_tokens": 0,
      "root": "/private/tmp"
    },
    "lumi_repository": {
      "files_scanned": 977,
      "matching_seed_tokens": 0,
      "root": "/scratch/project_462000131/anisrahm/hmsc-hpc"
    },
    "lumi_retained_runs": {
      "files_scanned": 5016,
      "matching_seed_tokens": 0,
      "root": "/scratch/project_462000131/anisrahm/hmsc-hpc-runs"
    },
    "result": "all_proposed_seed_roles_unused"
  }
}
