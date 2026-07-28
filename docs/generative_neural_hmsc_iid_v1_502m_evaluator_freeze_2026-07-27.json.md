{
  "schema_version": 1,
  "kind": "generative_neural_hmsc_iid_v1_evaluator_source_freeze",
  "protocol": "generative_neural_hmsc_iid_probit_v1",
  "evaluator_version": "generative_iid_v1_502_evaluator_v1",
  "date": "2026-07-27",
  "branch": "feature/generative-neural-hmsc",
  "source_commit": "recorded_after_commit_in_roadmap",
  "frozen_documents": {
    "preregistration": {
      "path": "docs/generative_neural_hmsc_iid_v1_preregistration_2026-07-27.md",
      "sha256": "09c6a195ca139bdf168816b4f50db321c789bfdd061628e4f99a28cca81cea3f"
    },
    "seed_audit": {
      "path": "docs/generative_neural_hmsc_iid_v1_seed_audit_2026-07-27.json.md",
      "sha256": "39e8763bf8a4fd525dc624570cd2f2b3392dbd1f62d7fa2e3c326f9340194cd6"
    },
    "design_review": {
      "path": "docs/generative_neural_hmsc_iid_v1_design_review_2026-07-27.md",
      "sha256": "d271caed64dc1346b1f8d9e192534949adedd3122c1e311638e912ca868990cc"
    },
    "evaluator_review": {
      "path": "docs/generative_neural_hmsc_iid_v1_502m_evaluator_review_2026-07-27.md",
      "sha256": "cff63ee4ae41a2dc56c1aa426df188b197ff9661ea6167dae8404b78dcb7eb5f"
    }
  },
  "source_files": [
    {
      "path": "pyhmsc/neural/generative_iid.py",
      "sha256": "a7885c9123ac4e52beb1ed366fd5c09857f132789e21cac540be6c96663b8d52"
    },
    {
      "path": "pyhmsc/neural/generative_iid_mcmc.py",
      "sha256": "558e40a6e98639899588f56c42f7595b81c9e05c34467a87ec5502eca794ee7c"
    },
    {
      "path": "pyhmsc/neural/generative_iid_artifact.py",
      "sha256": "fb6429a5a58eee2caffcd1f33118847db269b53cfdcd4fc3556d9ae1ed523cac"
    },
    {
      "path": "pyhmsc/neural/generative_iid_evaluation.py",
      "sha256": "cac0cc621a1f1aa74637a6f000008a4ddaa4627af7b818f3a36bc38ae5f219ee"
    },
    {
      "path": "pyhmsc/neural/generative_iid_comparators.py",
      "sha256": "512d6b2bcf19e50b8491f91219ccd9f7a167c42875adae0c583d1bb420c6dd16"
    },
    {
      "path": "pyhmsc/neural/__init__.py",
      "sha256": "9792eb5781b7d6bf6ce0b5c9a5a7162d84b72c0588b077c8931f979a9123b3db"
    },
    {
      "path": "examples/run_generative_neural_hmsc_iid_v1.py",
      "sha256": "0caa930b220ace7f036d03c924fdabf3a038be74d21d7e97e3ad99bccf6ef199"
    },
    {
      "path": "examples/run_generative_neural_hmsc_iid_v1_production.py",
      "sha256": "bf2658addd6184c4da9e012c9b5b89d7f285caac178a71ed7e914f4a65dbfd19"
    },
    {
      "path": "docs/lumi_generative_neural_hmsc_iid_v1_training_sbatch.sh",
      "sha256": "f7d77a354c8b5c60a3f02628ce93f9e990572ddf2d2c80e30c172c6e39703757"
    },
    {
      "path": "docs/lumi_generative_neural_hmsc_iid_v1_fixed_validation_sbatch.sh",
      "sha256": "8ac1dca460f2b466d1762238b5e058c20ee1bbbcf453283acb9b7d87bffca072"
    },
    {
      "path": "tests/test_neural_hmsc_generative_iid_v1.py",
      "sha256": "cee4190bc61a5073a08f430c8b0cd3e0b07e4fafef0a08e3a686a3071182d9f2"
    },
    {
      "path": "tests/test_neural_hmsc_generative_iid_v1_evaluation.py",
      "sha256": "f1e903303697bd35f4c4465002966d45398a34606a526780b6cfd0a1c97c71ff"
    }
  ],
  "verification": {
    "focused_tests": {
      "passed": 36,
      "skipped": 1,
      "failed": 0
    },
    "python_bytecode_compilation": true,
    "training_scheduler_bash_syntax": true,
    "fixed_validation_scheduler_bash_syntax": true,
    "git_diff_check": true,
    "ordinary_seed_exact_mcmc_adapter": true,
    "ordinary_seed_python_hmsc_adapter": true,
    "ordinary_seed_neural_hmsc_v0_1_adapter": true
  },
  "seed_seal": {
    "candidate_training_501m_opened": false,
    "fixed_validation_502m_opened": false,
    "reserved_503m_505m_opened": false,
    "redesign_511m_515m_opened": false
  },
  "authorization": {
    "this_artifact_authorizes_seed_access": false,
    "candidate_training_requires_explicit_confirmation": true,
    "fixed_validation_requires_later_separate_confirmation": true
  }
}
