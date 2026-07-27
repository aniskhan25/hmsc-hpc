{
  "audit_date": "2026-07-27",
  "kind": "generative_neural_hmsc_iid_v1_seed_audit",
  "protocol": "generative_neural_hmsc_iid_probit_v1",
  "proposed_seed_roles": {
    "candidate": {
      "training_contexts": {
        "start": 501000001,
        "end": 501000324,
        "count": 324,
        "derived_response_realizations_per_context": 2
      },
      "fixed_validation_contexts": {
        "start": 502000001,
        "end": 502000324,
        "count": 324
      },
      "reserved_evaluation_a_contexts": {
        "start": 503000001,
        "end": 503000324,
        "count": 324
      },
      "reserved_evaluation_b_contexts": {
        "start": 504000001,
        "end": 504000324,
        "count": 324
      },
      "reserved_evaluation_c_contexts": {
        "start": 505000001,
        "end": 505000324,
        "count": 324
      },
      "model_seed": 501900001,
      "disposable_training_contexts": {
        "start": 591000001,
        "end": 591000018,
        "count": 18
      },
      "disposable_validation_contexts": {
        "start": 592000001,
        "end": 592000018,
        "count": 18
      }
    },
    "single_permitted_representation_redesign": {
      "status": "sealed_unless_candidate_fixed_validation_fails",
      "training_contexts": {
        "start": 511000001,
        "end": 511000324,
        "count": 324,
        "derived_response_realizations_per_context": 2
      },
      "fixed_validation_contexts": {
        "start": 512000001,
        "end": 512000324,
        "count": 324
      },
      "reserved_evaluation_a_contexts": {
        "start": 513000001,
        "end": 513000324,
        "count": 324
      },
      "reserved_evaluation_b_contexts": {
        "start": 514000001,
        "end": 514000324,
        "count": 324
      },
      "reserved_evaluation_c_contexts": {
        "start": 515000001,
        "end": 515000324,
        "count": 324
      },
      "model_seed": 511900001,
      "disposable_training_contexts": {
        "start": 593000001,
        "end": 593000018,
        "count": 18
      },
      "disposable_validation_contexts": {
        "start": 594000001,
        "end": 594000018,
        "count": 18
      }
    }
  },
  "rng_contract": {
    "owning_seed": "one nine-digit context seed owns parameters, covariates, response realizations, masks, posterior draws, and comparator chains through named SeedSequence children",
    "training_responses": "two independent response children per production training context",
    "validation_response": "one response child plus one independently derived masked-cell holdout child",
    "posterior_draws": "named SeedSequence child of the owning context seed",
    "exact_model_mcmc": "four named chain children of the owning validation or evaluation context seed",
    "python_hmsc_hpc": "four named chain children distinct from exact-model MCMC",
    "real_data": "no simulation owning seed; all stochastic operations derive from the frozen candidate content hash and protocol tags",
    "additional_seed_ranges_permitted": false
  },
  "seed_token_rule": "(?<![0-9.])[0-9]{9}(?![0-9.])",
  "audit_results": {
    "local_repository": {
      "root": "/Users/anisr/Documents/hmsc-hpc",
      "scope": "docs, pyhmsc, hmsc, examples, tests; md/json/yaml/yml/py/sh",
      "files_scanned": 401,
      "matching_seed_tokens": 0
    },
    "local_retained_evidence": {
      "root": "/private/tmp",
      "scope": "depth <= 6; manifest/freeze/seed/summary JSON and Markdown under 5 MiB",
      "files_scanned": 11,
      "matching_seed_tokens": 0
    },
    "lumi_repository": {
      "root": "/scratch/project_462000131/anisrahm/hmsc-hpc",
      "scope": "docs, pyhmsc, hmsc, examples, tests; md/json/yaml/yml/py/sh",
      "matching_seed_tokens": 0
    },
    "lumi_retained_runs": {
      "root": "/scratch/project_462000131/anisrahm/hmsc-hpc-runs",
      "scope": "depth <= 6; manifest/freeze/seed/summary JSON, Markdown, and run paths",
      "matching_seed_tokens": 0
    },
    "result": "all_candidate_and_redesign_seed_roles_unused_before_preregistration"
  },
  "barriers": {
    "candidate_production": "501M-505M remain sealed until implementation and disposable 591M-592M smoke pass",
    "candidate_reserved": "503M-505M remain sealed until every 502M fixed-validation gate passes",
    "redesign": "511M-515M and 593M-594M remain sealed until a documented candidate failure authorizes the single representation redesign",
    "real_data": "Whittaker outcomes remain sealed until all simulation and comparator gates pass"
  },
  "schema_version": 1
}
