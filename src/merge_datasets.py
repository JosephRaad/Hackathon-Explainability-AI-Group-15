# =============================================================================
# TrustedAI  merge_datasets.py
# Step 1: Merge all 3 HR datasets into a comprehensive unified schema.
# Output: data/processed/hr_merged.csv
#
# NEW: Also exports per-source stats for dashboard source comparison:
#   data/processed/stats_drrich.json
#   data/processed/stats_ibm.json
#   data/processed/stats_kaggle.json
# =============================================================================

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

RAW_DIR = "data/raw/"
OUTPUT_PATH = "data/processed/hr_merged.csv"
STATS_DIR = "data/processed/"

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


# ── PER-SOURCE STATS EXPORT ──────────────────────────────────────────────────
def export_source_stats(df: pd.DataFrame, source_name: str, filename: str) -> None:
    """
    Save a JSON stats snapshot for a single source dataset.
    Used by app.py to display per-source analytics in the dashboard
    and answer chatbot questions like 'tell me about the IBM dataset'.
    """
    stats = {
        "source": source_name,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "attrition_rate": None,
        "n_terminated": None,
        "n_active": None,
        "departments": {},
        "top_department_attrition": None,
        "avg_age": None,
        "avg_tenure": None,
        "avg_engagement": None,
        "avg_satisfaction": None,
        "overtime_rate": None,
        "departure_causes": {},
        "sex_distribution": {},
        "performance_distribution": {},
    }

    # Attrition
    if "Termd" in df.columns:
        stats["attrition_rate"] = round(float(df["Termd"].mean()), 4)
        stats["n_terminated"] = int(df["Termd"].sum())
        stats["n_active"] = int((df["Termd"] == 0).sum())

    # Departments  include attrition rate per dept
    if "Department" in df.columns:
        dept_counts = df["Department"].value_counts().to_dict()
        stats["departments"] = {str(k): int(v) for k, v in dept_counts.items()}
        if "Termd" in df.columns:
            dept_attrition = (
                df.groupby("Department")["Termd"].mean()
                .sort_values(ascending=False)
            )
            stats["dept_attrition_rates"] = {
                str(k): round(float(v), 4)
                for k, v in dept_attrition.items()
            }
            stats["top_department_attrition"] = str(dept_attrition.index[0])

    # Age
    if "Age" in df.columns:
        stats["avg_age"] = round(float(df["Age"].mean()), 1)
        stats["age_range"] = [int(df["Age"].min()), int(df["Age"].max())]

    # Tenure
    if "YearsAtCompany" in df.columns:
        stats["avg_tenure"] = round(float(df["YearsAtCompany"].mean()), 1)

    # Engagement & satisfaction
    if "EngagementSurvey" in df.columns:
        stats["avg_engagement"] = round(float(df["EngagementSurvey"].mean()), 2)
    if "EmpSatisfaction" in df.columns:
        stats["avg_satisfaction"] = round(float(df["EmpSatisfaction"].mean()), 2)

    # Overtime
    if "OverTime" in df.columns:
        stats["overtime_rate"] = round(float(df["OverTime"].mean()), 4)

    # Departure causes (only for terminated employees)
    if "departure_cause" in df.columns:
        left = df[df["Termd"] == 1] if "Termd" in df.columns else df
        causes = left["departure_cause"].dropna().value_counts().to_dict()
        stats["departure_causes"] = {str(k): int(v) for k, v in causes.items()}

    # Sex distribution
    if "Sex" in df.columns:
        sex = df["Sex"].value_counts().to_dict()
        stats["sex_distribution"] = {str(k): int(v) for k, v in sex.items()}

    # Performance distribution
    if "PerformanceScore" in df.columns:
        perf = df["PerformanceScore"].value_counts().to_dict()
        stats["performance_distribution"] = {str(k): int(v) for k, v in perf.items()}

    # Unique features this source has that others may not
    all_possible = {
        "dr_rich": ["RaceDesc", "TermReason", "RecruitmentSource",
                    "SpecialProjectsCount", "DaysLateLast30", "Position"],
        "ibm": ["MonthlyIncome", "JobLevel", "StockOptionLevel",
                "YearsWithCurrManager", "TrainingTimesLastYear"],
        "kaggle": ["satisfaction_level", "last_evaluation",
                   "number_project", "average_montly_hours", "sales"],
    }
    source_key = source_name.lower().replace(" ", "_").replace(".", "")
    # Match loosely
    for key in all_possible:
        if key in source_key or source_key in key:
            stats["unique_features"] = all_possible[key]
            break
    else:
        stats["unique_features"] = []

    # Save
    out_path = os.path.join(STATS_DIR, filename)
    os.makedirs(STATS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  [Stats]   Saved {filename} "
          f"({stats['n_rows']} rows, attrition: {stats['attrition_rate']:.1%})")


# ── 1. LOAD DR. RICH DATASET ────────────────────────────────────────────────
def load_rich(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame()

    # ID & target
    out["employee_id"] = df["EmpID"].astype(str)
    out["Termd"] = df["Termd"].astype(int)

    # Demographics (sensitive  kept for fairness audit)
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

    # Overtime
    out["OverTime"] = ((df["DaysLateLast30"].fillna(0) > 2) |
                       (df["Absences"].fillna(0) > 15)).astype(int)

    # Work-life balance
    out["WorkLifeBalance"] = np.clip(
        4 - out["OverTime"] - (out["Absences"] > 10).astype(int) +
        np.random.choice([0, 1], size=len(out), p=[0.6, 0.4]), 1, 4
    ).astype(int)

    out["RecruitmentSource"] = df.get(
        "RecruitmentSource", pd.Series(["Unknown"]*len(df))
    ).fillna("Unknown")

    # TermReason and enrichment
    out["TermReason"] = df.get("TermReason", pd.Series([None]*len(df)))
    out["departure_cause"] = out["TermReason"].apply(_map_term_reason_to_cause)
    out.loc[out["Termd"] == 0, "departure_cause"] = None

    out["source_dataset"] = "dr_rich"

    print(f"  [Rich]    {len(out)} rows | attrition: {out['Termd'].mean():.1%}")
    return out


# ── 2. LOAD IBM DATASET ──────────────────────────────────────────────────────
def load_ibm(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Deduplicate  IBM dataset sometimes has duplicate EmployeeNumbers
    if "EmployeeNumber" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["EmployeeNumber"])
        dropped = before - len(df)
        if dropped > 0:
            print(f"  [IBM]     Dropped {dropped} duplicate rows")

    out = pd.DataFrame()

    out["employee_id"] = (
        df.get("EmployeeNumber", pd.Series(range(len(df))))
    ).astype(str)
    out["Termd"] = (df["Attrition"] == "Yes").astype(int)

    out["Sex"] = df["Gender"].str.strip().fillna("Unknown")
    out["RaceDesc"] = "Unknown"  # IBM dataset doesn't have race
    out["MaritalStatus"] = df["MaritalStatus"].str.strip().fillna("Unknown")
    out["Age"] = df["Age"].fillna(35).astype(int)

    out["Department"] = df["Department"].str.strip().str.title()
    out["Position"] = df.get("JobRole", pd.Series(["Unknown"]*len(df))).str.strip()
    out["Salary"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median()) * 12
    perf = df.get("PerformanceRating", pd.Series([3]*len(df)))
    out["PerformanceScore"] = perf.map(
        {1: "Needs Improvement", 2: "Needs Improvement",
         3: "Fully Meets", 4: "Exceeds"}
    ).fillna("Fully Meets")

    inv = df.get("JobInvolvement", pd.Series([3]*len(df))).fillna(3)
    env = df.get("EnvironmentSatisfaction", pd.Series([3]*len(df))).fillna(3)
    out["EngagementSurvey"] = ((inv + env) / 2 * (5/4)).clip(1, 5).round(2)

    sat = df.get("JobSatisfaction", pd.Series([3]*len(df))).fillna(3)
    out["EmpSatisfaction"] = (sat * (5/4)).clip(1, 5).round(0).astype(int)

    out["SpecialProjectsCount"] = df.get(
        "NumCompaniesWorked", pd.Series([0]*len(df))
    ).fillna(0)
    out["Absences"] = np.random.poisson(7, len(df))
    out["DaysLateLast30"] = np.random.choice([0, 0, 0, 1, 2, 3], size=len(df))

    out["YearsAtCompany"] = df.get(
        "YearsAtCompany", pd.Series([3]*len(df))
    ).fillna(3).round(1)
    out["OverTime"] = (
        df.get("OverTime", pd.Series(["No"]*len(df))) == "Yes"
    ).astype(int)
    out["WorkLifeBalance"] = df.get(
        "WorkLifeBalance", pd.Series([3]*len(df))
    ).fillna(3).astype(int)
    out["RecruitmentSource"] = "Unknown"
    out["TermReason"] = None

    causes = list(_CAUSE_WEIGHTS.keys())
    probs = list(_CAUSE_WEIGHTS.values())
    left_mask = out["Termd"] == 1
    out.loc[left_mask, "departure_cause"] = np.random.choice(
        causes, size=left_mask.sum(), p=probs
    )
    out.loc[~left_mask, "departure_cause"] = None

    out["source_dataset"] = "ibm"

    print(f"  [IBM]     {len(out)} rows | attrition: {out['Termd'].mean():.1%}")
    return out


# ── 3. LOAD KAGGLE DATASET ───────────────────────────────────────────────────
def load_kaggle(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Deduplicate
    if "EmpID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["EmpID"])
        dropped = before - len(df)
        if dropped > 0:
            print(f"  [Kaggle]  Dropped {dropped} duplicate rows")

    out = pd.DataFrame()

    out["employee_id"] = df.get(
        "EmpID", pd.Series(range(len(df)))
    ).astype(str)
    out["Termd"] = (df["Attrition"] == "Yes").astype(int)

    out["Sex"] = df["Gender"].str.strip().fillna("Unknown")
    out["RaceDesc"] = "Unknown"
    out["MaritalStatus"] = df.get(
        "MaritalStatus", pd.Series(["Unknown"]*len(df))
    ).str.strip()
    out["Age"] = df.get("Age", pd.Series([35]*len(df))).fillna(35).astype(int)

    out["Department"] = df["Department"].str.strip().str.title()
    out["Position"] = df.get(
        "JobRole", pd.Series(["Unknown"]*len(df))
    ).str.strip()
    out["Salary"] = df.get(
        "MonthlyIncome", pd.Series([5000]*len(df))
    ).fillna(5000) * 12
    perf = df.get("PerformanceRating", pd.Series([3]*len(df)))
    out["PerformanceScore"] = perf.map(
        {1: "Needs Improvement", 2: "Needs Improvement",
         3: "Fully Meets", 4: "Exceeds"}
    ).fillna("Fully Meets")

    inv = df.get("JobInvolvement", pd.Series([3]*len(df))).fillna(3)
    env = df.get("EnvironmentSatisfaction", pd.Series([3]*len(df))).fillna(3)
    out["EngagementSurvey"] = ((inv + env) / 2 * (5/4)).clip(1, 5).round(2)

    sat = df.get("JobSatisfaction", pd.Series([3]*len(df))).fillna(3)
    out["EmpSatisfaction"] = (sat * (5/4)).clip(1, 5).round(0).astype(int)

    out["SpecialProjectsCount"] = df.get(
        "NumCompaniesWorked", pd.Series([0]*len(df))
    ).fillna(0)
    out["Absences"] = np.random.poisson(7, len(df))
    out["DaysLateLast30"] = np.random.choice([0, 0, 0, 1, 2, 3], size=len(df))

    out["YearsAtCompany"] = df.get(
        "YearsAtCompany", pd.Series([3]*len(df))
    ).fillna(3).round(1)
    out["OverTime"] = (
        df.get("OverTime", pd.Series(["No"]*len(df))) == "Yes"
    ).astype(int)
    out["WorkLifeBalance"] = df.get(
        "WorkLifeBalance", pd.Series([3]*len(df))
    ).fillna(3).astype(int)
    out["RecruitmentSource"] = "Unknown"
    out["TermReason"] = None

    causes = list(_CAUSE_WEIGHTS.keys())
    probs = list(_CAUSE_WEIGHTS.values())
    left_mask = out["Termd"] == 1
    out.loc[left_mask, "departure_cause"] = np.random.choice(
        causes, size=left_mask.sum(), p=probs
    )
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
    left_mask = df["Termd"] == 1
    decline_prob = np.random.random(left_mask.sum()) < 0.7
    df.loc[left_mask, "satisfaction_trend"] = np.where(
        decline_prob, "declining", df.loc[left_mask, "satisfaction_trend"]
    )
    return df


# ── MAIN MERGE PIPELINE ─────────────────────────────────────────────────────
def merge_datasets() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  TrustedAI  Multi-Dataset Merge Pipeline")
    print("=" * 60 + "\n")

    frames = []
    os.makedirs(STATS_DIR, exist_ok=True)

    # ── 1. Dr. Rich (base  required) ────────────────────────────────────────
    if os.path.exists(RICH_PATH):
        df_rich = load_rich(RICH_PATH)
        export_source_stats(df_rich, "Dr. Rich", "stats_drrich.json")
        frames.append(df_rich)
    else:
        print(f"  [ERROR] Required: {RICH_PATH}")
        return None

    # ── 2. IBM ────────────────────────────────────────────────────────────────
    if os.path.exists(IBM_PATH):
        df_ibm = load_ibm(IBM_PATH)
        export_source_stats(df_ibm, "IBM HR", "stats_ibm.json")
        frames.append(df_ibm)
    else:
        print(f"  [SKIP]  IBM not found: {IBM_PATH}")

    # ── 3. Kaggle ─────────────────────────────────────────────────────────────
    if os.path.exists(KAGGLE_PATH):
        df_kaggle = load_kaggle(KAGGLE_PATH)
        export_source_stats(df_kaggle, "Kaggle", "stats_kaggle.json")
        frames.append(df_kaggle)
    else:
        print(f"  [SKIP]  Kaggle not found: {KAGGLE_PATH}")

    # ── Combine ───────────────────────────────────────────────────────────────
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

    # ── Save merged ───────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)

    # ── Save merged stats (the "All Sources" view) ────────────────────────────
    export_source_stats(df, "Merged (All Sources)", "stats_merged.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Combined: {len(df)} rows × {len(df.columns)} cols")
    print(f"  Attrition rate: {df['Termd'].mean():.1%}")
    print(f"  Departments: {df['Department'].nunique()}")
    print(f"  Sources: {df['source_dataset'].value_counts().to_dict()}")
    print(f"\n  ✅ Saved: {OUTPUT_PATH}")
    print(f"  ✅ Saved: stats_drrich.json, stats_ibm.json, "
          f"stats_kaggle.json, stats_merged.json")
    print("=" * 60 + "\n")
    return df


if __name__ == "__main__":
    df = merge_datasets()