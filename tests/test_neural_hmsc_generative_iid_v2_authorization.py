from pathlib import Path


def test_v2_disposable_scheduler_is_exactly_scoped():
    script = Path(
        "docs/lumi_generative_neural_hmsc_iid_v2_disposable_sbatch.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=dev-g" in script
    assert "#SBATCH --time=03:00:00" in script
    assert "#SBATCH --gpus-per-node=1" in script
    assert "940d73d6de6e032797e4d695bd9799a74ef0b943" in script
    assert "GENERATE_593M_594M_DISPOSABLE_ONLY" in script
    assert "593000001-593000018" in script
    assert "594000001-594000018" in script
    assert "511000001-515000324" in script
    assert 'env -u "${OPENING_ENV}"' in script
    assert "--mode preflight" in script
    assert "--mode disposable-smoke" in script
    assert "--mode validate-disposable" in script
    assert "OPEN_GENERATIVE_IID_V2_511M" not in script


def test_v2_disposable_authorization_keeps_later_blocks_sealed():
    record = Path(
        "docs/generative_neural_hmsc_iid_v2_disposable_"
        "authorization_2026-08-01.md"
    ).read_text(encoding="utf-8")
    assert "76911182c1d34bcd4c979f70b1340af126ddd89baafdc821c4024cc6f846a43a" in record
    assert "7a0bf9ecf89a5e1896ba254d24e916978cfe8e76caf0898a7dffef1df679cf07" in record
    assert "does not open 511M-515M" in record
    assert "Any retry requires a new" in record


def test_v2_disposable_retry_scheduler_uses_container_safe_pythonpath():
    script = Path(
        "docs/lumi_generative_neural_hmsc_iid_v2_disposable_retry_sbatch.sh"
    ).read_text(encoding="utf-8")
    assert 'export PYTHONPATH=".:${PYTHONPATH:-}"' in script
    assert 'export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"' not in script
    assert script.count("-m examples.run_generative_neural_hmsc_iid_v2") == 3
    assert '"${PYTHON}" examples/run_generative_neural_hmsc_iid_v2.py' not in script
    assert "940d73d6de6e032797e4d695bd9799a74ef0b943" in script
    assert "GENERATE_593M_594M_DISPOSABLE_ONLY" in script
    assert 'env -u "${OPENING_ENV}"' in script
    assert "OPEN_GENERATIVE_IID_V2_511M" not in script

    attempt = Path(
        "docs/generative_neural_hmsc_iid_v2_disposable_attempt_20518403.md"
    ).read_text(encoding="utf-8")
    assert "ff618d92c4d4f616507aaa31e2f434cb2cdaa9b2d985bcc0e7e567bc6735cdb7" in attempt
    assert "No 593M or 594M seed was opened" in attempt


def test_v2_disposable_retry_authorization_is_bounded():
    record = Path(
        "docs/generative_neural_hmsc_iid_v2_disposable_retry_"
        "authorization_2026-08-01.md"
    ).read_text(encoding="utf-8")
    assert "ff618d92c4d4f616507aaa31e2f434cb2cdaa9b2d985bcc0e7e567bc6735cdb7" in record
    assert "940d73d6de6e032797e4d695bd9799a74ef0b943" in record
    assert "593000001-593000018" in record
    assert "594000001-594000018" in record
    assert "does not open 511M-515M" in record
    assert "does not authorize another retry" in record
    assert "generative_iid_v2_disposable_retry1_940d73d_20260801" in record


def test_v2_disposable_retry_failure_stops_before_production():
    result = Path(
        "docs/generative_neural_hmsc_iid_v2_disposable_retry_"
        "failure_2026-08-01.md"
    ).read_text(encoding="utf-8")
    assert "stop_before_511m_numerical_failure" in result
    assert "matched all 36" in result
    assert "non-finite v2 gradient" in result
    assert "checkpoint manifest or weights" in result
    assert "Do not open 511M-515M" in result
    assert "retry authorization is consumed" in result


def test_v2_repaired_preflight_scheduler_is_token_free_and_hash_pinned():
    script = Path(
        "docs/lumi_generative_neural_hmsc_iid_v2_repaired_preflight_sbatch.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=dev-g" in script
    assert "87828857ee1718a8825a1a15e7af99abe49a86ee4d179f6cbce6591162aa71bc" in script
    assert "e3b708a09b0c920676e592759f44f7457cc75decff66138b6b29c509254a6192" in script
    assert "--mode preflight" in script
    assert "simulation_generation_called" in script
    assert "disposable_seed_ranges_opened" in script
    assert "GENERATE_593M" not in script
    assert "OPEN_GENERATIVE_IID_V2_511M" not in script


def test_v2_repaired_preflight_record_keeps_all_seed_roles_sealed():
    record = Path(
        "docs/generative_neural_hmsc_iid_v2_repaired_preflight_2026-08-01.md"
    ).read_text(encoding="utf-8")
    assert "repaired_boundary_preflight_passed_no_seed_opened" in record
    assert "cca9e97518e77c5ca958dfdc3bee753997ed7ac5" in record
    assert "bb343bcef927455b5ffedb0483015f75f3da053176d58c1f032b3fece7790eb1" in record
    assert "a9f3c3f0f535f31217da279f9907f8c1d0fcf11001a7337ffbb4a4fdade9fe6f" in record
    assert "Blocks 593M-594M and 511M-515M remain sealed" in record
    assert "not authorize disposable simulation" in record


def test_v2_final_disposable_scheduler_is_terminal_and_exactly_scoped():
    script = Path(
        "docs/lumi_generative_neural_hmsc_iid_v2_disposable_final_sbatch.sh"
    ).read_text(encoding="utf-8")
    assert "cca9e97518e77c5ca958dfdc3bee753997ed7ac5" in script
    assert "GENERATE_593M_594M_DISPOSABLE_ONLY" in script
    assert "593000001-593000018" in script
    assert "594000001-594000018" in script
    assert "511000001-515000324" in script
    assert "87828857ee1718a8825a1a15e7af99abe49a86ee4d179f6cbce6591162aa71bc" in script
    assert "b09e3e1eb743fe62a509876a284323e5bb151ce043cac7b60dba8a9a35f9300e" in script
    assert "--mode preflight" in script
    assert "--mode disposable-smoke" in script
    assert "--mode validate-disposable" in script
    assert "OPEN_GENERATIVE_IID_V2_511M" not in script


def test_v2_final_disposable_authorization_is_terminal_and_bounded():
    record = Path(
        "docs/generative_neural_hmsc_iid_v2_final_disposable_"
        "authorization_2026-08-01.md"
    ).read_text(encoding="utf-8")
    assert "authorize_one_final_repaired_disposable_verification" in record
    assert "cca9e97518e77c5ca958dfdc3bee753997ed7ac5" in record
    assert "bb343bcef927455b5ffedb0483015f75f3da053176d58c1f032b3fece7790eb1" in record
    assert "9ca1238d7e88560e58b0e92727c821933e90ec704d1ea69e61a86c4aef31066c" in record
    assert "593000001-593000018" in record
    assert "594000001-594000018" in record
    assert "zero failed-Cholesky warnings" in record
    assert "final disposable execution" in record
    assert "does not authorize an" in record
    assert "every 511M-515M seed remain" in record
