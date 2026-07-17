"""
Test suite for underwriting_engine.py

Run locally with: pytest tests/
This is exactly what a CI pipeline runs automatically on every push.
"""

import pytest
from underwriting_engine import Applicant, underwrite, required_income_proof, medical_risk_score, Decision


# ---------------------------------------------------------------------------
# Financial rule tests
# ---------------------------------------------------------------------------

def test_no_income_proof_needed_under_1cr():
    assert required_income_proof(50_00_000) == []  # 50 lakh, under 1 Cr


def test_income_proof_required_between_1_and_2cr():
    docs = required_income_proof(1.5 * 1e7)
    assert len(docs) == 1
    assert "surrogate" in docs[0]


def test_standard_proof_mandatory_above_5cr():
    docs = required_income_proof(6 * 1e7)
    assert "mandatory" in docs[0]


# ---------------------------------------------------------------------------
# Medical risk scoring tests
# ---------------------------------------------------------------------------

def make_applicant(**overrides):
    defaults = dict(
        name="Test Applicant",
        age=30,
        annual_income_inr=1_000_000,
        sum_assured_inr=50_00_000,  # under 1 Cr, no docs needed
        height_cm=170,
        weight_kg=65,
        smoker=False,
        pre_existing_conditions=[],
        occupation_risk_class=1,
        income_proof_provided=[],
    )
    defaults.update(overrides)
    return Applicant(**defaults)


def test_healthy_applicant_low_risk_score():
    applicant = make_applicant()
    assert medical_risk_score(applicant) == 0


def test_smoker_increases_risk_score():
    non_smoker = make_applicant(smoker=False)
    smoker = make_applicant(smoker=True)
    assert medical_risk_score(smoker) > medical_risk_score(non_smoker)


def test_high_risk_condition_increases_score_significantly():
    healthy = make_applicant()
    cardiac = make_applicant(pre_existing_conditions=["cardiac"])
    assert medical_risk_score(cardiac) - medical_risk_score(healthy) >= 5


# ---------------------------------------------------------------------------
# End-to-end decision tests — this is what catches "did my logic change break
# a real business outcome" during CI, not just a unit-level function
# ---------------------------------------------------------------------------

def test_healthy_low_sum_assured_applicant_approved():
    applicant = make_applicant()
    result = underwrite(applicant)
    assert result["decision"] == Decision.APPROVE


def test_high_risk_applicant_declined():
    applicant = make_applicant(
        age=60,
        smoker=True,
        pre_existing_conditions=["cardiac"],
        weight_kg=110,
        height_cm=165,
    )
    result = underwrite(applicant)
    assert result["decision"] == Decision.DECLINE


def test_missing_income_docs_triggers_refer():
    applicant = make_applicant(sum_assured_inr=3 * 1e7, income_proof_provided=[])
    result = underwrite(applicant)
    assert result["decision"] == Decision.REFER


def test_invalid_age_rejected_by_validation():
    with pytest.raises(Exception):
        make_applicant(age=15)  # below pydantic's ge=18 constraint
