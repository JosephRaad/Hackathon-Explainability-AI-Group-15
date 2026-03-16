# =============================================================================
# TrustedAI — Multi-Dataset Merger (UPDATED FOR CORRECT SCHEMAS)
# =============================================================================

import pandas as pd
import numpy as np
import os

RAW_DIR      = "data/raw/"
OUTPUT_PATH  = "data/processed/hr_combined.csv"

RICH_PATH    = f"{RAW_DIR}HRDataset_v14.csv"
IBM_PATH     = f"{RAW_DIR}IBM_HR_Attrition.csv"
KAGGLE_PATH  = f"{RAW_DIR}HR_comma_sep.csv"

# ── COMMON SCHEMA ─────────────────────────────────────────────────────────────
SCHEMA = [
    "Termd",               # Target: 1=left, 0=active
    "EngagementSurvey",    # 1-5 scale (normalized)
    "EmpSatisfaction",     # 1-5 scale (normalized)
    "SpecialProjectsCount",
    "Absences",
    "DaysLateLast30",
    "Department",
    "PerformanceScore",
    "RecruitmentSource",
    "source_dataset"       # Provenance tracking
]

# ── 1. LOAD DR. RICH ──────────────────────────────────────────────────────────
def load_rich(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame()
    
    out["Termd"]                = df["Termd"].astype(int)
    out["EngagementSurvey"]     = df["EngagementSurvey"]
    out["EmpSatisfaction"]      = df["EmpSatisfaction"]
    out["SpecialProjectsCount"] = df.get("SpecialProjectsCount", pd.Series([0]*len(df))).fillna(0)
    out["Absences"]             = df["Absences"]
    out["DaysLateLast30"]       = df.get("DaysLateLast30", pd.Series([0]*len(df))).fillna(0)
    out["Department"]           = df["Department"].str.strip()
    out["PerformanceScore"]     = df["PerformanceScore"]
    out["RecruitmentSource"]    = df.get("RecruitmentSource", pd.Series(["Unknown"]*len(df)))
    out["source_dataset"]       = "dr_rich"
    
    print(f"  [Rich]    Loaded {len(out)} rows")
    return out

# ── 2. LOAD IBM AND KAGGLE (SHARED SCHEMA) ────────────────────────────────────
# Since both your IBM and Kaggle files share the exact same column structure
# (Attrition, Gender, JobSatisfaction, etc.), we can use one robust function for both!
def load_ibm_format(path: str, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    
    # Target
    out["Termd"] = (df["Attrition"] == "Yes").astype(int)

    # JobInvolvement + EnvironmentSatisfaction → proxy for EngagementSurvey
    inv = df.get("JobInvolvement", pd.Series([3]*len(df))).fillna(3)
    env = df.get("EnvironmentSatisfaction", pd.Series([3]*len(df))).fillna(3)
    out["EngagementSurvey"] = ((inv + env) / 2) * (5/4) # Rescale 1-4 → 1-5

    # Satisfaction
    sat = df.get("JobSatisfaction", pd.Series([3]*len(df))).fillna(3)
    out["EmpSatisfaction"] = sat * (5/4)
    
    out["SpecialProjectsCount"] = df.get("NumCompaniesWorked", pd.Series([0]*len(df))).fillna(0)
    
    # Proxy for Absences
    hike = df.get("PercentSalaryHike", pd.Series([10]*len(df))).fillna(10)
    out["Absences"] = (hike - 11).clip(0) * 2 
    
    out["DaysLateLast30"] = 0 
    
    # Department string formatting
    out["Department"] = df.get("Department", pd.Series(["Unknown"]*len(df))).astype(str).str.title()
    
    # Map Performance Ratings
    perf = df.get("PerformanceRating", pd.Series([3]*len(df)))
    out["PerformanceScore"] = perf.map({1: "Needs Improvement", 2: "Needs Improvement", 3: "Fully Meets", 4: "Exceeds"}).fillna("Fully Meets")
    
    out["RecruitmentSource"] = "Unknown"
    out["source_dataset"] = source_name
    
    print(f"  [{source_name}] Loaded {len(out)} rows (attrition rate: {out['Termd'].mean():.1%})")
    return out


# ── MERGE PIPELINE ────────────────────────────────────────────────────────────
def merge_datasets() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  TrustedAI — Multi-Dataset Merge Pipeline")
    print("=" * 60 + "\n")

    frames = []

    # 1. Dr. Rich Data (Base)
    if os.path.exists(RICH_PATH):
        frames.append(load_rich(RICH_PATH))
    else:
        print(f"  [ERROR] Required file missing: {RICH_PATH}")
        return None

    # 2. IBM Data
    if os.path.exists(IBM_PATH):
        frames.append(load_ibm_format(IBM_PATH, "ibm"))
    else:
        print(f"  [IBM]    Not found at {IBM_PATH} — skipping")

    # 3. Kaggle Data
    if os.path.exists(KAGGLE_PATH):
        frames.append(load_ibm_format(KAGGLE_PATH, "kaggle_hr"))
    else:
        print(f"  [Kaggle] Not found at {KAGGLE_PATH} — skipping")

    # Combine everything
    df_combined = pd.concat(frames, ignore_index=True)

    # Clean and clip the data
    df_combined["EngagementSurvey"] = df_combined["EngagementSurvey"].clip(1, 5)
    df_combined["EmpSatisfaction"]  = df_combined["EmpSatisfaction"].clip(1, 5)
    df_combined["Absences"]         = df_combined["Absences"].clip(0, 50)

    # Save to Processed Folder
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\n  Combined dataset: {len(df_combined)} rows × {len(df_combined.columns)} cols")
    print(f"  Overall termination rate: {df_combined['Termd'].mean():.1%}")
    print("\n  ✅ Saved successfully to:", OUTPUT_PATH)
    print("=" * 60 + "\n")

    return df_combined

if __name__ == "__main__":
    df = merge_datasets()