{
  "artifact_audit": {
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
      "source_weights_sha256": "bb6e76d3ec9bc5e294ceac3051c3b2d7e5273db5053cfa5ceac676913d6265d9",
      "weights_sha256": "bb6e76d3ec9bc5e294ceac3051c3b2d7e5273db5053cfa5ceac676913d6265d9",
      "weights_unchanged": true
    },
    "inventory_files_validated": 36,
    "other_immutable_members": [
      {
        "calibration_artifact_sha256": "172ad83e32b78eba71fb1e83ae7972b24d30a86d35450928d176524b3df7aceb",
        "checkpoint_manifest_sha256": "49939deb832a4280a36d2149b445c128af9f5594d2997511cc25d97fb6b6cc08",
        "seed": 20260722,
        "weights_sha256": "6b103d3bc78c7ef97702b391521fe5f18349dbe6b7889e7de562ae312e068bca"
      },
      {
        "calibration_artifact_sha256": "1c0739e82ff90182f5b38db9432920bf6dbacac311bd5963ef37175d33318a5a",
        "checkpoint_manifest_sha256": "9b695200d0c47906206b8b3f61e752658d68106d620a186275a3bfd58a23b136",
        "seed": 20260723,
        "weights_sha256": "85b3612442f1041fcc099b0d35d271ec14fa0260efc1e686b46e86e4d79d2245"
      }
    ],
    "package_manifest_sha256": "d2daa81ec841390df59324a208216ffa0032ac514e6c679649d98815490bdbc7",
    "release_content_sha256": "affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8",
    "release_id": "neural_hmsc_v0_1",
    "release_registry_local": "/private/tmp/neural_hmsc_releases/neural_hmsc_v0_1",
    "release_registry_lumi": "/scratch/project_462000131/anisrahm/hmsc-hpc-deployments/neural_hmsc_v0_1"
  },
  "audit_date": "2026-07-23",
  "kind": "neural_hmsc_m56_artifact_seed_audit",
  "proposed_seed_roles": {
    "disposable_evaluation": {
      "count": 27,
      "end": 292000027,
      "start": 292000001
    },
    "disposable_training": {
      "count": 27,
      "end": 291000027,
      "start": 291000001
    },
    "fixed_validation": {
      "count": 324,
      "end": 212000324,
      "start": 212000001
    },
    "model_seed": 211900001,
    "reserved_evaluation_a": {
      "count": 324,
      "end": 213000324,
      "start": 213000001
    },
    "reserved_evaluation_b": {
      "count": 324,
      "end": 214000324,
      "start": 214000001
    },
    "reserved_evaluation_c": {
      "count": 324,
      "end": 215000324,
      "start": 215000001
    },
    "training": {
      "count": 324,
      "end": 211000324,
      "start": 211000001
    }
  },
  "schema_version": 1,
  "seed_token_audit": {
    "integer_token_rule": "(?<![0-9.])[0-9]{9}(?![0-9.])",
    "local_repository": {
      "files_scanned": 487,
      "matching_seed_tokens": 0,
      "root": "/Users/anisr/Documents/hmsc-hpc"
    },
    "local_retained_evidence": {
      "files_scanned": 1513,
      "matching_seed_tokens": 0,
      "root": "/private/tmp"
    },
    "lumi_retained_evidence": {
      "matching_seed_tokens": 0,
      "repository_files_present": 961,
      "roots": [
        "/scratch/project_462000131/anisrahm/hmsc-hpc",
        "/scratch/project_462000131/anisrahm/hmsc-hpc-runs"
      ],
      "run_files_present": 5009,
      "scanner": "GNU grep 3.1 PCRE integer-token search"
    },
    "result": "all_proposed_seed_roles_unused"
  }
}
