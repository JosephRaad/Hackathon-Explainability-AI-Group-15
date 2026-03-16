# =============================================================================
# TrustedAI — preprocess.py
# Feature engineering. Auto-uses hr_combined.csv if available,
# otherwise falls back to hr_anonymized.csv.
# Output: data/processed/hr_features.csv + hr_features_meta.json
# =============================================================================

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder

COMBINED_PATH = "data/processed/hr_combined.csv"
ANON_PATH     = "data/processed/hr_anonymized.csv"
OUTPUT_PATH   = "data/processed/hr_features.csv"
META_PATH     = "data/processed/hr_features_meta.json"

TARGET   = "Termd"
SENSITIVE = ["Sex", "RaceDesc"]
NUMERIC   = ["EngagementSurvey", "EmpSatisfaction", "SpecialProjectsCount",
             "Absences", "DaysLateLast30"]
CATEGORIC = ["Department", "PerformanceScore", "RecruitmentSource", "MaritalDesc"]


def run():
    print("\n========================================")
    print("  TrustedAI — Feature Engineering")
    print("========================================\n")

    # Auto-select input
    if os.path.exists(COMBINED_PATH):
        input_path = COMBINED_PATH
        print(f"  Using: hr_combined.csv (multi-dataset)")
    elif os.path.exists(ANON_PATH):
        input_path = ANON_PATH
        print(f"  Using: hr_anonymized.csv (single dataset)")
    else:
        raise FileNotFoundError("Run anonymize.py (and optionally merge_datasets.py) first.")

    df = pd.read_csv(input_path)
    print(f"  Loaded: {len(df)} rows × {len(df.columns)} columns")

    # Validate target
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found!")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    dist = df[TARGET].value_counts().to_dict()
    print(f"\n  Target '{TARGET}': {dist}  (attrition rate: {df[TARGET].mean():.1%})")

    # Impute numerics
    print("\n  Imputing missing values:")
    for col in NUMERIC:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss:
                med = df[col].median()
                df[col] = df[col].fillna(med)
                print(f"    {col}: {n_miss} missing → median ({med:.2f})")
            else:
                print(f"    {col}: no missing values ✓")

    # Encode categoricals + sensitive
    print("\n  Encoding categorical features:")
    mappings = {}
    for col in CATEGORIC + SENSITIVE:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(["nan", "None", ""], "Unknown").fillna("Unknown")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            mappings[col] = {int(i): str(c) for i, c in enumerate(le.classes_)}
            print(f"    {col}: {len(le.classes_)} classes")

    # Select final columns
    keep = [TARGET] + [c for c in NUMERIC + CATEGORIC + SENSITIVE if c in df.columns]
    if "EmpID" in df.columns:
        keep.append("EmpID")
    df_out = df[keep].copy()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    meta = {
        "target":          TARGET,
        "sensitive":       [c for c in SENSITIVE  if c in df_out.columns],
        "numeric":         [c for c in NUMERIC    if c in df_out.columns],
        "categorical":     [c for c in CATEGORIC  if c in df_out.columns],
        "label_mappings":  mappings,
        "n_rows":          len(df_out),
        "attrition_rate":  float(df_out[TARGET].mean()),
        "source":          input_path,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ Features saved : {OUTPUT_PATH}  ({df_out.shape})")
    print(f"  ✅ Metadata saved : {META_PATH}")
    print("========================================\n")
    return df_out


if __name__ == "__main__":
    df = run()
    print(df.dtypes)
    print(df.head(3))
