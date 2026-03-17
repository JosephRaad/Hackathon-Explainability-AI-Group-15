# =============================================================================
# TrustedAI — merge_datasets.py
# Step 1: Merge all 3 HR datasets into a comprehensive unified schema.
# Generates synthetic enrichment fields (departure_cause, exit_feedback, etc.)
# Output: data/processed/hr_merged.csv
# =============================================================================

import pandas as pd
import numpy as np
import os
import hashlib
from datetime import datetime

RAW_DIR = "data/raw/"
OUTPUT_PATH = "data/processed/hr_merged.csv"

RICH_PATH = f"{RAW_DIR}HRDataset_v14.csv"
IBM_PATH = f"{RAW_DIR}IBM_HR_Attrition.csv"
KAGGLE_PATH = f"{RAW_DIR}HR_comma_sep.csv"

np.random.seed(42)


# ── EXIT FEEDBACK TEMPLATES ──────────────────────────────────────────────────
_EXIT_TEMPLATES = {
    "compensation": [
        "After {tenure} years, my salary hasn't kept pace with market rates. I received an offer that was significantly higher and felt I had no choice but to accept it.",
        "The compensation structure here doesn't reward experience. I've consistently exceeded targets but my pay reviews have been minimal.",
        "I enjoyed the team but the total compensation package wasn't competitive. Benefits and base salary both fell below industry standard.",
    ],
    "career_growth": [
        "I've been in the same role for {tenure} years with no clear path forward. I want to grow into a leadership position but those opportunities don't exist here.",
        "Despite strong performance reviews, there's been no discussion about advancement. I need a role where I can develop new skills and take on more responsibility.",
        "The lack of internal mobility is frustrating. I asked about lateral moves twice and was told to wait. I found a company that invests in career development.",
    ],
    "management": [
        "My relationship with my manager deteriorated over the past year. Feedback was inconsistent and I felt my contributions weren't recognized.",
        "There were ongoing communication issues with leadership in my department. Decisions were made without consulting the team and morale suffered.",
        "I raised concerns about workload distribution multiple times but nothing changed. The management style in our department needs improvement.",
    ],
    "work_life_balance": [
        "The overtime expectations became unsustainable. I was regularly working evenings and weekends with no flexibility offered in return.",
        "I need better work-life balance. The constant pressure and lack of remote work options made it difficult to manage personal responsibilities.",
        "After {tenure} years of intense hours, I'm burned out. The company culture prioritizes presence over productivity.",
    ],
    "relocation": [
        "I'm relocating for family reasons and unfortunately remote work isn't available for my role. I'd consider returning if circumstances change.",
        "My spouse accepted a position in another city. I explored transfer options but none were available in my department.",
    ],
    "layoff": [
        "My position was eliminated as part of the restructuring. I understand the business reasons but it was still difficult after {tenure} years.",
        "The department reduction affected my role. I appreciate the severance package and would consider returning if a suitable position opens up.",
    ],
    "performance": [
        "I acknowledge that my recent performance wasn't meeting expectations. I was dealing with personal issues that affected my work.",
        "The role evolved beyond my current skill set and the training support wasn't sufficient to bridge the gap.",
    ],
}

_CAUSE_WEIGHTS = {
    "compensation": 0.25,
    "career_growth": 0.22,
    "management": 0.18,
    "work_life_balance": 0.15,
    "relocation": 0.08,
    "layoff": 0.07,
    "performance": 0.05,
}


def _generate_exit_feedback(cause, tenure_years):
    """Generate realistic exit interview text based on departure cause."""
    templates = _EXIT_TEMPLATES.get(cause, _EXIT_TEMPLATES["compensation"])
    text = np.random.choice(templates)
    return text.format(tenure=max(1, int(tenure_years)))


def _map_term_reason_to_cause(reason):
    """Map TermReason from Rich dataset to standardized departure_cause."""
    if pd.isna(reason) or reason == "N/A-StillEmployed":
        return None
    reason_lower = str(reason).lower()
    if any(k in reason_lower for k in ["another position", "more money"]):
        return np.random.choice(["compensation", "career_growth"], p=[0.6, 0.4])
    if any(k in reason_lower for k in ["unhappy", "manager"]):
        return np.random.choice(["management", "work_life_balance"], p=[0.6, 0.4])
    if any(k in reason_lower for k in ["retire", "relocation", "return to school", "medical"]):
        return "relocation"
    if any(k in reason_lower for k in ["performance", "attendance"]):
        return "performance"
    if any(k in reason_lower for k in ["layoff", "restructur"]):
        return "layoff"
    return np.random.choice(list(_CAUSE_WEIGHTS.keys()), p=list(_CAUSE_WEIGHTS.values()))


# ── 1. LOAD DR. RICH DATASET ────────────────────────────────────────────────
def load_rich(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame()

    # ID & target
    out["employee_id"] = df["EmpID"].astype(str)
    out["Termd"] = df["Termd"].astype(int)

    # Demographics (sensitive — kept for fairness audit)
    out["Sex"] = df["Sex"].str.strip().map({"M": "Male", "F": "Female"}).fillna("Unknown")
    out["RaceDesc"] = df["RaceDesc"].str.strip().fillna("Unknown")
    out["MaritalStatus"] = df["MaritalDesc"].str.strip().fillna("Unknown")

    # Age
    if "DOB" in df.columns:
        dob = pd.to_datetime(df["DOB"], errors="coerce", dayfirst=False)
        out["Age"] = (datetime.now().year - dob.dt.year).fillna(35).astype(int)
    else:
        out["Age"] = 35

    # Job attributes
    out["Department"] = df["Department"].str.strip()
    out["Position"] = df["Position"].str.strip()
    out["Salary"] = df["Salary"].fillna(df["Salary"].median())
    out["PerformanceScore"] = df["PerformanceScore"].str.strip()

    # Engagement & satisfaction
    out["EngagementSurvey"] = df["EngagementSurvey"].fillna(3.0)
    out["EmpSatisfaction"] = df["EmpSatisfaction"].fillna(3)
    out["SpecialProjectsCount"] = df.get("SpecialProjectsCount", pd.Series([0]*len(df))).fillna(0)
    out["Absences"] = df["Absences"].fillna(0)
    out["DaysLateLast30"] = df.get("DaysLateLast30", pd.Series([0]*len(df))).fillna(0)

    # Tenure
    hire = pd.to_datetime(df["DateofHire"], errors="coerce")
    term = pd.to_datetime(df["DateofTermination"], errors="coerce")
    end_date = term.fillna(pd.Timestamp.now())
    out["YearsAtCompany"] = ((end_date - hire).dt.days / 365.25).clip(0, 40).fillna(3).round(1)

    # Overtime (simulated based on absences and late days)
    out["OverTime"] = ((df["DaysLateLast30"].fillna(0) > 2) |
                       (df["Absences"].fillna(0) > 15)).astype(int)

    # Work-life balance (simulated: inverse of overtime + absences)
    out["WorkLifeBalance"] = np.clip(
        4 - out["OverTime"] - (out["Absences"] > 10).astype(int) +
        np.random.choice([0, 1], size=len(out), p=[0.6, 0.4]), 1, 4
    ).astype(int)

    out["RecruitmentSource"] = df.get("RecruitmentSource", pd.Series(["Unknown"]*len(df))).fillna("Unknown")

    # TermReason and enrichment
    out["TermReason"] = df.get("TermReason", pd.Series([None]*len(df)))
    out["departure_cause"] = out["TermReason"].apply(_map_term_reason_to_cause)
    # For active employees, no departure cause
    out.loc[out["Termd"] == 0, "departure_cause"] = None

    out["source_dataset"] = "dr_rich"

    print(f"  [Rich]    {len(out)} rows | attrition: {out['Termd'].mean():.1%}")
    return out


# ── 2. LOAD IBM DATASET ──────────────────────────────────────────────────────
def load_ibm(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame()

    out["employee_id"] = (df.get("EmployeeNumber", pd.Series(range(len(df))))).astype(str)
    out["Termd"] = (df["Attrition"] == "Yes").astype(int)

    # Demographics
    out["Sex"] = df["Gender"].str.strip().fillna("Unknown")
    out["RaceDesc"] = "Unknown"  # IBM dataset doesn't have race
    out["MaritalStatus"] = df["MaritalStatus"].str.strip().fillna("Unknown")
    out["Age"] = df["Age"].fillna(35).astype(int)

    # Job attributes
    out["Department"] = df["Department"].str.strip().str.title()
    out["Position"] = df.get("JobRole", pd.Series(["Unknown"]*len(df))).str.strip()
    out["Salary"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median()) * 12
    perf = df.get("PerformanceRating", pd.Series([3]*len(df)))
    out["PerformanceScore"] = perf.map(
        {1: "Needs Improvement", 2: "Needs Improvement", 3: "Fully Meets", 4: "Exceeds"}
    ).fillna("Fully Meets")

    # Engagement & satisfaction (rescale 1-4 to 1-5)
    inv = df.get("JobInvolvement", pd.Series([3]*len(df))).fillna(3)
    env = df.get("EnvironmentSatisfaction", pd.Series([3]*len(df))).fillna(3)
    out["EngagementSurvey"] = ((inv + env) / 2 * (5/4)).clip(1, 5).round(2)

    sat = df.get("JobSatisfaction", pd.Series([3]*len(df))).fillna(3)
    out["EmpSatisfaction"] = (sat * (5/4)).clip(1, 5).round(0).astype(int)

    out["SpecialProjectsCount"] = df.get("NumCompaniesWorked", pd.Series([0]*len(df))).fillna(0)
    out["Absences"] = np.random.poisson(7, len(df))  # Simulated
    out["DaysLateLast30"] = np.random.choice([0, 0, 0, 1, 2, 3], size=len(df))

    out["YearsAtCompany"] = df.get("YearsAtCompany", pd.Series([3]*len(df))).fillna(3).round(1)
    out["OverTime"] = (df.get("OverTime", pd.Series(["No"]*len(df))) == "Yes").astype(int)
    out["WorkLifeBalance"] = df.get("WorkLifeBalance", pd.Series([3]*len(df))).fillna(3).astype(int)
    out["RecruitmentSource"] = "Unknown"

    out["TermReason"] = None
    # Generate departure causes for those who left
    causes = list(_CAUSE_WEIGHTS.keys())
    probs = list(_CAUSE_WEIGHTS.values())
    left_mask = out["Termd"] == 1
    out.loc[left_mask, "departure_cause"] = np.random.choice(causes, size=left_mask.sum(), p=probs)
    out.loc[~left_mask, "departure_cause"] = None

    out["source_dataset"] = "ibm"

    print(f"  [IBM]     {len(out)} rows | attrition: {out['Termd'].mean():.1%}")
    return out


# ── 3. LOAD KAGGLE DATASET ───────────────────────────────────────────────────
def load_kaggle(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame()

    out["employee_id"] = df.get("EmpID", pd.Series(range(len(df)))).astype(str)
    out["Termd"] = (df["Attrition"] == "Yes").astype(int)

    out["Sex"] = df["Gender"].str.strip().fillna("Unknown")
    out["RaceDesc"] = "Unknown"
    out["MaritalStatus"] = df.get("MaritalStatus", pd.Series(["Unknown"]*len(df))).str.strip()
    out["Age"] = df.get("Age", pd.Series([35]*len(df))).fillna(35).astype(int)

    out["Department"] = df["Department"].str.strip().str.title()
    out["Position"] = df.get("JobRole", pd.Series(["Unknown"]*len(df))).str.strip()
    out["Salary"] = df.get("MonthlyIncome", pd.Series([5000]*len(df))).fillna(5000) * 12
    perf = df.get("PerformanceRating", pd.Series([3]*len(df)))
    out["PerformanceScore"] = perf.map(
        {1: "Needs Improvement", 2: "Needs Improvement", 3: "Fully Meets", 4: "Exceeds"}
    ).fillna("Fully Meets")

    inv = df.get("JobInvolvement", pd.Series([3]*len(df))).fillna(3)
    env = df.get("EnvironmentSatisfaction", pd.Series([3]*len(df))).fillna(3)
    out["EngagementSurvey"] = ((inv + env) / 2 * (5/4)).clip(1, 5).round(2)

    sat = df.get("JobSatisfaction", pd.Series([3]*len(df))).fillna(3)
    out["EmpSatisfaction"] = (sat * (5/4)).clip(1, 5).round(0).astype(int)

    out["SpecialProjectsCount"] = df.get("NumCompaniesWorked", pd.Series([0]*len(df))).fillna(0)
    out["Absences"] = np.random.poisson(7, len(df))
    out["DaysLateLast30"] = np.random.choice([0, 0, 0, 1, 2, 3], size=len(df))

    out["YearsAtCompany"] = df.get("YearsAtCompany", pd.Series([3]*len(df))).fillna(3).round(1)
    out["OverTime"] = (df.get("OverTime", pd.Series(["No"]*len(df))) == "Yes").astype(int)
    out["WorkLifeBalance"] = df.get("WorkLifeBalance", pd.Series([3]*len(df))).fillna(3).astype(int)
    out["RecruitmentSource"] = "Unknown"

    out["TermReason"] = None
    causes = list(_CAUSE_WEIGHTS.keys())
    probs = list(_CAUSE_WEIGHTS.values())
    left_mask = out["Termd"] == 1
    out.loc[left_mask, "departure_cause"] = np.random.choice(causes, size=left_mask.sum(), p=probs)
    out.loc[~left_mask, "departure_cause"] = None

    out["source_dataset"] = "kaggle"

    print(f"  [Kaggle]  {len(out)} rows | attrition: {out['Termd'].mean():.1%}")
    return out


# ── GENERATE EXIT FEEDBACK TEXT ──────────────────────────────────────────────
def enrich_with_exit_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """Generate realistic exit interview text for employees who left."""
    df["exit_feedback"] = None
    left = df["Termd"] == 1
    for idx in df[left].index:
        cause = df.loc[idx, "departure_cause"]
        tenure = df.loc[idx, "YearsAtCompany"]
        if cause and not pd.isna(cause):
            df.loc[idx, "exit_feedback"] = _generate_exit_feedback(cause, tenure)
    n_feedback = df["exit_feedback"].notna().sum()
    print(f"  [Enrich]  Generated {n_feedback} exit interview texts")
    return df


# ── GENERATE SATISFACTION TREND ──────────────────────────────────────────────
def enrich_with_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Add satisfaction_trend based on satisfaction + engagement + departure."""
    conditions = [
        (df["EmpSatisfaction"] <= 2) | (df["EngagementSurvey"] < 2.5),
        (df["EmpSatisfaction"] >= 4) & (df["EngagementSurvey"] >= 4.0),
    ]
    choices = ["declining", "improving"]
    df["satisfaction_trend"] = np.select(conditions, choices, default="stable")
    # Override: employees who left likely had declining trend
    left_mask = df["Termd"] == 1
    decline_prob = np.random.random(left_mask.sum()) < 0.7
    df.loc[left_mask, "satisfaction_trend"] = np.where(
        decline_prob, "declining", df.loc[left_mask, "satisfaction_trend"]
    )
    return df


# ── MAIN MERGE PIPELINE ─────────────────────────────────────────────────────
def merge_datasets() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  TrustedAI — Multi-Dataset Merge Pipeline")
    print("=" * 60 + "\n")

    frames = []

    # 1. Dr. Rich (base — required)
    if os.path.exists(RICH_PATH):
        frames.append(load_rich(RICH_PATH))
    else:
        print(f"  [ERROR] Required: {RICH_PATH}")
        return None

    # 2. IBM
    if os.path.exists(IBM_PATH):
        frames.append(load_ibm(IBM_PATH))
    else:
        print(f"  [SKIP]  IBM not found: {IBM_PATH}")

    # 3. Kaggle
    if os.path.exists(KAGGLE_PATH):
        frames.append(load_kaggle(KAGGLE_PATH))
    else:
        print(f"  [SKIP]  Kaggle not found: {KAGGLE_PATH}")

    # Combine
    df = pd.concat(frames, ignore_index=True)

    # Clip & clean
    df["EngagementSurvey"] = df["EngagementSurvey"].clip(1, 5)
    df["EmpSatisfaction"] = df["EmpSatisfaction"].clip(1, 5)
    df["Absences"] = df["Absences"].clip(0, 50)
    df["Age"] = df["Age"].clip(18, 70)
    df["YearsAtCompany"] = df["YearsAtCompany"].clip(0, 40)

    # Enrich
    df = enrich_with_exit_feedback(df)
    df = enrich_with_trends(df)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n  Combined: {len(df)} rows × {len(df.columns)} cols")
    print(f"  Attrition rate: {df['Termd'].mean():.1%}")
    print(f"  Departments: {df['Department'].nunique()}")
    print(f"  Sources: {df['source_dataset'].value_counts().to_dict()}")
    print(f"\n  ✅ Saved: {OUTPUT_PATH}")
    print("=" * 60 + "\n")
    return df


if __name__ == "__main__":
    df = merge_datasets()
