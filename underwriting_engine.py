"""
Insurance Underwriting Engine
------------------------------
Evaluates whether a policy application should be:
  - APPROVED (standard terms)
  - APPROVED WITH CONDITIONS (loading / exclusion / extra docs)
  - REFERRED (needs manual underwriter review)
  - DECLINED

Factors considered: sum assured, income proof, age, BMI/medical history,
lifestyle risk (smoking), and occupation risk.

Libraries used here: pydantic (input validation), pandas (batch scoring),
dataclasses/enum for clean rule modeling. In production you'd swap the
hardcoded rule tables for a config file (YAML/JSON) or a DB table so
underwriting rules can be updated without a code deploy.
"""

from enum import Enum
from pydantic import BaseModel, Field, field_validator
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Input schema (pydantic enforces types & basic sanity checks up front)
# ---------------------------------------------------------------------------

class Applicant(BaseModel):
    name: str
    age: int = Field(ge=18, le=75)
    annual_income_inr: float = Field(gt=0)
    sum_assured_inr: float = Field(gt=0)
    height_cm: float
    weight_kg: float
    smoker: bool
    pre_existing_conditions: list[str] = []
    occupation_risk_class: int = Field(ge=1, le=4)  # 1 = low risk (desk job), 4 = high risk (mining, etc.)
    income_proof_provided: list[str] = []           # e.g. ["bank_statement", "itr", "form_26as"]

    @field_validator("annual_income_inr", "sum_assured_inr")
    @classmethod
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @property
    def bmi(self) -> float:
        h_m = self.height_cm / 100
        return round(self.weight_kg / (h_m ** 2), 1)


class Decision(str, Enum):
    APPROVE = "APPROVE"
    APPROVE_WITH_CONDITIONS = "APPROVE_WITH_CONDITIONS"
    REFER = "REFER"
    DECLINE = "DECLINE"


# ---------------------------------------------------------------------------
# 2. Financial underwriting rules (mirrors your sheet's SUC → doc requirement)
# ---------------------------------------------------------------------------

def required_income_proof(sum_assured: float) -> list[str]:
    """Sum-assured-based document requirement, same logic as the underwriting sheet."""
    cr = 1e7  # 1 crore
    if sum_assured <= 1 * cr:
        return []  # Nil
    elif sum_assured <= 2 * cr:
        return ["surrogate_or_cibil_estimator_or_payu_or_standard_proof"]
    elif sum_assured <= 5 * cr:
        return ["cibil_income_estimator_or_standard_proof"]
    else:
        return ["standard_income_proof_mandatory"]  # >5 Cr


def income_multiple_ok(applicant: Applicant) -> bool:
    """Common life-insurance rule of thumb: sum assured shouldn't wildly
    exceed an income multiple (varies by insurer/age band — placeholder logic)."""
    max_multiple = 25 if applicant.age < 45 else 15
    return applicant.sum_assured_inr <= applicant.annual_income_inr * max_multiple


# ---------------------------------------------------------------------------
# 3. Medical / lifestyle risk scoring
# ---------------------------------------------------------------------------

HIGH_RISK_CONDITIONS = {"diabetes_uncontrolled", "cardiac", "cancer_history", "kidney_disease"}
MODERATE_RISK_CONDITIONS = {"hypertension", "thyroid", "diabetes_controlled"}

def medical_risk_score(applicant: Applicant) -> int:
    """Simple additive point system — production systems often use a
    logistic regression or gradient boosted model trained on claims data
    instead of hardcoded points, but the point-based table is still common
    for transparent/auditable underwriting."""
    score = 0

    # BMI bands
    bmi = applicant.bmi
    if bmi < 18.5 or bmi >= 35:
        score += 3
    elif 30 <= bmi < 35:
        score += 2
    elif 25 <= bmi < 30:
        score += 1

    # Smoking
    if applicant.smoker:
        score += 3

    # Pre-existing conditions
    conditions = set(applicant.pre_existing_conditions)
    if conditions & HIGH_RISK_CONDITIONS:
        score += 5
    elif conditions & MODERATE_RISK_CONDITIONS:
        score += 2

    # Age bands
    if applicant.age >= 60:
        score += 3
    elif applicant.age >= 45:
        score += 1

    # Occupation risk
    score += (applicant.occupation_risk_class - 1) * 2

    return score


# ---------------------------------------------------------------------------
# 4. Decision engine — combines financial + medical rules
# ---------------------------------------------------------------------------

def underwrite(applicant: Applicant) -> dict:
    reasons = []

    # --- Financial check ---
    needed_docs = required_income_proof(applicant.sum_assured_inr)
    docs_missing = [d for d in needed_docs if not applicant.income_proof_provided]

    if docs_missing:
        return {
            "decision": Decision.REFER,
            "reasons": [f"Missing required income proof: {docs_missing}"],
            "medical_risk_score": None,
        }

    if not income_multiple_ok(applicant):
        return {
            "decision": Decision.DECLINE,
            "reasons": ["Sum assured exceeds permissible multiple of annual income"],
            "medical_risk_score": None,
        }

    # --- Medical/lifestyle check ---
    risk_score = medical_risk_score(applicant)

    if risk_score >= 10:
        decision = Decision.DECLINE
        reasons.append(f"Medical risk score {risk_score} exceeds decline threshold")
    elif risk_score >= 6:
        decision = Decision.REFER
        reasons.append(f"Medical risk score {risk_score} requires manual underwriter review")
    elif risk_score >= 3:
        decision = Decision.APPROVE_WITH_CONDITIONS
        reasons.append(f"Medical risk score {risk_score} — approve with premium loading / exclusion")
    else:
        decision = Decision.APPROVE
        reasons.append("Standard risk — approved at standard terms")

    return {
        "decision": decision,
        "reasons": reasons,
        "medical_risk_score": risk_score,
        "bmi": applicant.bmi,
    }


# ---------------------------------------------------------------------------
# 5. Batch scoring example (pandas) — e.g. scoring 10,000 applications at once
# ---------------------------------------------------------------------------

def batch_underwrite(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in df.iterrows():
        applicant = Applicant(**row.to_dict())
        result = underwrite(applicant)
        results.append({"name": applicant.name, **result})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        Applicant(
            name="Applicant A",
            age=32,
            annual_income_inr=1_200_000,
            sum_assured_inr=15_000_000,   # 1.5 Cr -> needs income proof
            height_cm=170,
            weight_kg=68,
            smoker=False,
            pre_existing_conditions=[],
            occupation_risk_class=1,
            income_proof_provided=["standard_income_proof"],
        ),
        Applicant(
            name="Applicant B",
            age=55,
            annual_income_inr=800_000,
            sum_assured_inr=6_000_000,
            height_cm=165,
            weight_kg=95,
            smoker=True,
            pre_existing_conditions=["cardiac"],
            occupation_risk_class=2,
            income_proof_provided=["standard_income_proof"],
        ),
    ]

    for applicant in test_cases:
        result = underwrite(applicant)
        print(f"\n{applicant.name}: {result}")
