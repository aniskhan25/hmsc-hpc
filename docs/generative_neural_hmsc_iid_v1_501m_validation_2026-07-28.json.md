{
  "schema_version": 1,
  "kind": "generative_neural_hmsc_iid_v1_501m_independent_validation",
  "protocol": "generative_neural_hmsc_iid_probit_v1",
  "date": "2026-07-28",
  "job": {
    "id": "20301852",
    "partition": "standard-g",
    "slurm_state": "FAILED",
    "elapsed": "11:03:29",
    "exit_code": "1:0",
    "max_rss_kib": 9782180,
    "failure_phase": "post_training_read_only_validation",
    "training_artifacts_written_before_failure": true
  },
  "source_commit": "fc2ac5aff84f2fbed2c3604f3001f3647618fdc0",
  "failure_classification": {
    "kind": "validator_schema_alias_mismatch",
    "generated_key": "fixed_validation_opened",
    "validator_expected_key": "fixed_validation_seed_ranges_opened",
    "model_or_optimizer_failure": false,
    "artifact_corruption": false,
    "later_seed_access": false,
    "retraining_required": false
  },
  "freeze": {
    "sha256": "93f11221c9bbbd3b8ced541888397541ab61f0b88ae23eebc3431e969512ae39",
    "sidecar_value_matches": true,
    "source_commit_matches": true,
    "frozen_document_hashes_match": true
  },
  "candidate_checkpoint": {
    "content_sha256": "d36dd3b23ccdba36041792716b9fb2cb21a437265870e686cdef1f01b9d05e30",
    "manifest_sha256": "48a6bfb95cc9c93dbf4770aca013a8d552a1d18a1ed5f087667113141aabb45d",
    "weights_sha256": "43b4eded085b0213f53ffa795e5bf91f367a2dc86cd17a2915da7e404f8043c7",
    "weights_bytes": 1192352,
    "load_roundtrip_passed": true,
    "all_weights_finite": true,
    "ordinary_fixture_posterior_finite": true
  },
  "no_latent_ablation_checkpoint": {
    "content_sha256": "691f8c992ec709ac241af32ea0fd7e94e43c3ed9d79c768e01a23a4a1e8193bc",
    "manifest_sha256": "ba7c809117798559ba5a74cc30881f23fb1feda5856fadef2ae5b56332193a16",
    "weights_sha256": "1ab01e332b7b23609fb0bdb7a41e978a29c3f237c94eb02c0ab0276bb541232d",
    "weights_bytes": 1192352,
    "load_roundtrip_passed": true,
    "all_weights_finite": true,
    "ordinary_fixture_posterior_finite": true
  },
  "training_corpus": {
    "manifest_sha256": "aeba904e8c047cf2952f1f2a3e61482f2d358f4644ecaa2385282e2b34ae8697",
    "owning_seed_range": [
      501000001,
      501000324
    ],
    "owning_context_count": 324,
    "responses_per_context": 2,
    "training_realization_count": 648,
    "factorial_contract_matches": true,
    "metadata_digests_well_formed": true
  },
  "training_report": {
    "sha256": "07ac63f9295d82c9e1aed5d43a0af89faa580b4ee2bbc8b352e7a52e7646524c",
    "status": "candidate_501m_training_complete",
    "wall_time_seconds": 39690.8498818872,
    "candidate_final_loss": 1207.7735360227985,
    "candidate_final_iwelbo": -1207.7735360227985,
    "candidate_final_gradient_norm": 480.08390516116293,
    "ablation_final_loss": 1194.669885046688,
    "ablation_final_iwelbo": -1194.669885046688,
    "ablation_final_gradient_norm": 311.4044099030671,
    "all_metrics_finite": true,
    "all_artifact_bindings_match": true,
    "candidate_and_ablation_source_inventories_match": true
  },
  "independent_validation": {
    "postfreeze_validation_sha256": "0f6ac100df4497d7df8636962cf5c67a76dbefcb4915dcd77a4df6446c3c87c6",
    "focused_tests": {
      "passed": 37,
      "skipped": 1,
      "failed": 0
    },
    "ordinary_smoke_seed": 881501001,
    "candidate_posterior_state_dimension": 494,
    "ablation_posterior_state_dimension": 494,
    "candidate_diagonal_scale_positive": true,
    "ablation_diagonal_scale_positive": true
  },
  "seed_seal": {
    "fixed_validation_502m_opened": false,
    "reserved_503m_505m_opened": false,
    "redesign_511m_515m_opened": false
  },
  "decision": {
    "accept_501m_training_artifacts": true,
    "rerun_501m": false,
    "eligible_to_consider_separate_502m_authorization": true,
    "this_report_authorizes_502m": false
  }
}
