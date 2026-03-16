# =============================================================================
# TrustedAI — preprocess.py
# Step 3: Feature engineering on anonymized data.
# Protected attributes (Sex, RaceDesc) kept separately for bias audit.
# Output: data/processed/hr_features.csv + hr_features_meta.json
# =============================================================================

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = "data/processed/hr_anonymized.csv"
OUTPUT_PATH = "data/processed/hr_features.csv"
META_PATH = "data/processed/hr_features_meta.json"

TARGET = "Termd"

# Protected attributes — used for fairness audit, NOT as model features
PROTECTED = ["Sex", "RaceDesc"]

# Model features (numeric)
NUMERIC = [
    "EngagementSurvey", "EmpSatisfaction", "SpecialProjectsCount",
    "Absences", "DaysLateLast30", "YearsAtCompany", "OverTime",
    "WorkLifeBalance",
]

# Model features (categorical — will be encoded)
CATEGORICAL = [
    "Department", "PerformanceScore", "RecruitmentSource",
    "MaritalStatus", "AgeBracket", "SalaryBand", "satisfaction_trend",
]


def run():
    print("\n" + "=" * 60)
    print("  TrustedAI — Feature Engineering")
    print("=" * 60 + "\n")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Anonymized dataset not found: {INPUT_PATH}\n"
            "Run: merge_datasets.py → anonymize.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    print(f"  Input: {len(df)} rows × {len(df.columns)} columns")

    # Validate target
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found!")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    dist = df[TARGET].value_counts().to_dict()
    print(f"  Target '{TARGET}': {dist} (attrition: {df[TARGET].mean():.1%})")

    # ── Impute numeric features ──────────────────────────────────────────
    print("\n  Numeric features:")
    for col in NUMERIC:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss > 0:
                med = df[col].median()
                df[col] = df[col].fillna(med)
                print(f"    {col}: {n_miss} missing → median ({med:.2f})")
            else:
                print(f"    {col}: OK ✓")
        else:
            print(f"    {col}: NOT FOUND — creating default")
            df[col] = 0

    # ── Encode categorical features ──────────────────────────────────────
    print("\n  Categorical features:")
    label_mappings = {}
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(
                ["nan", "None", "", "NaN"], "Unknown"
            ).fillna("Unknown")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_mappings[col] = {int(i): str(c) for i, c in enumerate(le.classes_)}
            print(f"    {col}: {len(le.classes_)} classes → encoded")
        else:
            print(f"    {col}: NOT FOUND — skipping")

    # ── Encode protected attributes (for audit only) ─────────────────────
    print("\n  Protected attributes (audit only, NOT model features):")
    protected_mappings = {}
    for col in PROTECTED:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(
                ["nan", "None", "", "NaN"], "Unknown"
            ).fillna("Unknown")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            protected_mappings[col] = {int(i): str(c) for i, c in enumerate(le.classes_)}
            label_mappings[col] = protected_mappings[col]
            print(f"    {col}: {len(le.classes_)} classes → encoded (kept for AIF360)")
        else:
            print(f"    {col}: NOT FOUND")

    # ── Select final columns ─────────────────────────────────────────────
    feature_cols = [c for c in NUMERIC + CATEGORICAL if c in df.columns]
    protected_cols = [c for c in PROTECTED if c in df.columns]
    keep = [TARGET] + feature_cols + protected_cols
    if "employee_id" in df.columns:
        keep.append("employee_id")
    if "Department" in df.columns and "Department" not in keep:
        keep.append("Department")
    # Keep departure info for analysis
    for extra in ["departure_cause", "exit_feedback", "source_dataset", "Position"]:
        if extra in df.columns and extra not in keep:
            keep.append(extra)

    keep = list(dict.fromkeys(keep))  # Deduplicate preserving order
    df_out = df[keep].copy()

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    meta = {
        "target": TARGET,
        "feature_cols": feature_cols,
        "numeric": [c for c in NUMERIC if c in df_out.columns],
        "categorical": [c for c in CATEGORICAL if c in df_out.columns],
        "protected": protected_cols,
        "label_mappings": label_mappings,
        "protected_mappings": protected_mappings,
        "n_rows": len(df_out),
        "attrition_rate": float(df_out[TARGET].mean()),
        "source": INPUT_PATH,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Output: {df_out.shape}")
    print(f"  Model features:     {len(feature_cols)}")
    print(f"  Protected columns:  {protected_cols}")
    print(f"\n  ✅ Features: {OUTPUT_PATH}")
    print(f"  ✅ Metadata: {META_PATH}")
    print("=" * 60 + "\n")
    return df_out


if __name__ == "__main__":
    df = run()
    print(df.dtypes)
    print(df.head(3))
