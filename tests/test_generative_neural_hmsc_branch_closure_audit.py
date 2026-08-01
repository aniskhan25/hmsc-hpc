from pathlib import Path


AUDIT = Path(
    "docs/generative_neural_hmsc_branch_closure_audit_2026-08-01.md"
)


def test_closure_audit_has_explicit_capability_classes():
    audit = AUDIT.read_text(encoding="utf-8")
    assert "close_current_branch_without_structural_neural_qualification" in audit
    assert "## Capability Matrix" in audit
    assert "**Qualified, marginal-only**" in audit
    assert "**Predictive-only**" in audit
    assert "**Infrastructure-only:**" in audit
    assert "**Failed:**" in audit
    assert "**Unsupported:**" in audit


def test_closure_audit_preserves_only_defensible_claims():
    audit = AUDIT.read_text(encoding="utf-8")
    assert "neural_hmsc_v0_1" in audit
    assert "neural_hmsc_variable_probit_v1" in audit
    assert "Python MCMC remains the statistical" in audit
    assert "not be described as generative Neural HMSC" in audit
    assert "near-equivalent to MCMC" in audit
    assert (
        "Users requiring those capabilities must use qualified Python MCMC"
        in audit
    )


def test_closure_audit_records_both_generative_failures():
    audit = AUDIT.read_text(encoding="utf-8")
    assert "26/65 gates passed" in audit
    assert "close to no-latent ablation" in audit
    assert "36/36 corpus fingerprints" in audit
    assert "non-finite v2 gradient" in audit
    assert "zero Cholesky warnings" in audit
    assert "do not retry v1 or v2" in audit


def test_closure_audit_retires_later_seeds_and_separates_next_decision():
    audit = AUDIT.read_text(encoding="utf-8")
    assert "511M-515M" in audit
    assert "retired with this representation" in audit
    assert "Next Decision, Not Yet Authorized" in audit
    assert "must not begin implementation or allocate" in audit
    assert "fresh train, validation, and reserved seed ledgers" in audit
