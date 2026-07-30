{
  "schema_version": 1,
  "kind": "generative_neural_hmsc_iid_v2_seed_reaudit",
  "date": "2026-07-31",
  "candidate_protocol": "generative_neural_hmsc_iid_probit_v1",
  "redesign_protocol": "generative_neural_hmsc_iid_probit_v2_orbit",
  "original_seed_audit": {
    "path": "docs/generative_neural_hmsc_iid_v1_seed_audit_2026-07-27.json.md",
    "sha256": "39e8763bf8a4fd525dc624570cd2f2b3392dbd1f62d7fa2e3c326f9340194cd6"
  },
  "candidate_failure": {
    "path": "docs/generative_neural_hmsc_iid_v1_502m_failure_2026-07-30.md",
    "sha256": "36f04ee135974f549e5544c33dc911f213fa0536c9ec902a2c71e0046c09bb91",
    "decision": "stop_before_reserved_evaluation"
  },
  "redesign_seed_roles": {
    "disposable_training": [593000001, 593000018],
    "disposable_validation": [594000001, 594000018],
    "production_training": [511000001, 511000324],
    "fixed_validation": [512000001, 512000324],
    "reserved_evaluation_a": [513000001, 513000324],
    "reserved_evaluation_b": [514000001, 514000324],
    "reserved_evaluation_c": [515000001, 515000324],
    "model_seed": 511900001
  },
  "retained_evidence_scan": {
    "local_root": "/private/tmp",
    "local_scope": "depth <= 6; JSON, Markdown, text, and log files under 5 MiB",
    "local_exact_or_prefix_matches": 0,
    "lumi_root": "/scratch/project_462000131/anisrahm/hmsc-hpc-runs",
    "lumi_scope": "depth <= 6; JSON, Markdown, text, and log files under 5 MiB plus redesign-labelled run paths",
    "lumi_redesign_labelled_paths": 0,
    "lumi_broad_prefix_matches": 1,
    "lumi_false_positive": {
      "path": "neural_hmsc_benchmark_19829666/gpu_utilization.log",
      "value": 515747354,
      "classification": "repeated GPU telemetry counter outside the assigned 515000001-515000324 range"
    },
    "actual_redesign_seed_evidence_matches": 0
  },
  "result": "all_511m_515m_and_593m_594m_roles_remain_unused",
  "authorization": {
    "implementation": false,
    "disposable_593m_594m": false,
    "production_511m_515m": false,
    "statement": "This audit permits preregistration review only and opens no seed."
  }
}
