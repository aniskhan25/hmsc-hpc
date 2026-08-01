{
  "audit_date": "2026-08-01",
  "base_commit": "253e7802642192b0d72427b461bf9fc9cc30fa99",
  "branch": "feature/neural-transport-hmsc",
  "decision": "reserve_fresh_blocks_without_opening",
  "ledger_search_roots": [
    "docs",
    "examples",
    "tests",
    "pyhmsc",
    "hmsc"
  ],
  "prior_ledger_prefixes_audited_through": "594M structured milestone blocks plus all nine-digit literals in the search roots",
  "protocol": "neural_transport_hmsc_iid_probit_v0_1",
  "simulation_generation_called": false,
  "mcmc_generation_called": false,
  "scheduler_submission_performed": false,
  "artifact_output_created": false,
  "all_reserved_blocks_opened": false,
  "blocks": {
    "disposable_training_contexts": {
      "range": [791000001, 791000018],
      "count": 18,
      "opened": false
    },
    "disposable_reference_chain_seeds": {
      "range": [791100001, 791100072],
      "count": 72,
      "opened": false
    },
    "disposable_validation_contexts": {
      "range": [792000001, 792000018],
      "count": 18,
      "opened": false
    },
    "disposable_validation_chain_seeds": {
      "range": [792100001, 792100072],
      "count": 72,
      "opened": false
    },
    "production_training_contexts": {
      "range": [711000001, 711000108],
      "count": 108,
      "opened": false
    },
    "production_training_reference_chain_seeds": {
      "range": [712000001, 712000432],
      "count": 432,
      "opened": false
    },
    "fixed_validation_contexts": {
      "range": [713000001, 713000036],
      "count": 36,
      "opened": false
    },
    "fixed_validation_paired_chain_seeds": {
      "range": [714000001, 714000144],
      "count": 144,
      "opened": false
    },
    "reserved_evaluation_contexts": {
      "range": [715000001, 715000036],
      "count": 36,
      "opened": false
    },
    "reserved_evaluation_paired_chain_seeds": {
      "range": [716000001, 716000144],
      "count": 144,
      "opened": false
    },
    "whittaker_realdata_chain_seeds": {
      "range": [717000001, 717000008],
      "count": 8,
      "opened": false
    },
    "network_and_shuffle_seeds": {
      "values": [719900001, 719900002],
      "count": 2,
      "opened": false
    }
  },
  "role_rules": {
    "disposable": "plumbing, finite optimization, exact-target, correction, fallback, and seal verification only; cannot select architecture or gates",
    "production_training": "fit the frozen context encoder and affine transport only",
    "fixed_validation": "first statistical exactness and efficiency decision; cannot modify the candidate",
    "reserved_evaluation": "one-shot confirmation opened only after every fixed-validation gate passes",
    "realdata": "frozen Whittaker confirmation opened only after reserved evaluation passes; outcomes cannot train or select",
    "network_and_shuffle": "candidate initialization and deterministic corpus order only"
  },
  "retired_and_forbidden": {
    "generative_iid_v2": [511000001, 515000324],
    "reason": "retired with the closed v2 representation and never reusable"
  },
  "freshness_check": {
    "prefixes": [711, 712, 713, 714, 715, 716, 717, 719, 791, 792],
    "all_absent_before_this_audit": true,
    "replacement_after_opening_forbidden": true,
    "cross_role_reuse_forbidden": true
  }
}
