# =============================================================================
# TrustedAI — anonymize.py
# GDPR pipeline: Suppression · Pseudonymization · Generalization · Masking
# Input : data/raw/HRDataset_v14.csv
# Output: data/processed/hr_anonymized.csv
# =============================================================================

import pandas as pd
import hashlib
import os
from datetime import datetime

INPUT_PATH  = "data/raw/HRDataset_v14.csv"
OUTPUT_PATH = "data/processed/hr_anonymized.csv"


def run(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    print("\n========================================")
    print("  TrustedAI — GDPR Anonymization")
    print("========================================\n")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"  Loaded: {len(df)} rows × {len(df.columns)} columns")
    original_cols = len(df.columns)

    # 1 — SUPPRESSION: remove direct identifiers and sensible data
    drop_cols = ["Employee_Name", "ManagerName", "ManagerID",
                 "MarriedID", "GenderID", "EmpStatusID", "DeptID",
                 "PerfScoreID", "PositionID", "FromDiversityJobFairID", 
                 "MaritalStatusID", "Sex", "MaritalDesc", "HispanicLatino", 
                 "RaceDesc"]
    existing  = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)
    print(f"  [1] Suppression     — removed {len(existing)} columns: {existing}")

    # 2 — PSEUDONYMIZATION: hash EmpID
    if "EmpID" in df.columns:
        df["EmpID"] = (df["EmpID"].astype(str)
                       .apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:12]))
        print(f"  [2] Pseudonymization — EmpID → SHA-256 (12 chars)")

    # 3 — GENERALIZATION: DOB → AgeBracket
    if "DOB" in df.columns:
        dob = pd.to_datetime(df["DOB"], errors="coerce", infer_datetime_format=True)
        age = datetime.now().year - dob.dt.year
        df["AgeBracket"] = pd.cut(
            age,
            bins=[0, 25, 35, 45, 55, 65, 120],
            labels=["Under 25", "25-34", "35-44", "45-54", "55-64", "65+"],
            right=False
        ).astype(str)
        df["AgeBracket"] = df["AgeBracket"].replace("nan", "Unknown")
        df = df.drop(columns=["DOB"])
        print(f"  [3] Generalization  — DOB → AgeBracket (6 bands)")

    # 4 — MASKING: ZIP → 2-digit prefix
    if "Zip" in df.columns:
        df["Zip"] = df["Zip"].astype(str).str.strip().str[:2] + "***"
        print(f"  [4] Masking         — Zip → regional prefix (XX***)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n  Columns: {original_cols} → {len(df.columns)} ({original_cols-len(df.columns)} removed)")
    print(f"  Rows preserved: {len(df)}")
    print(f"\n  ✅ Saved: {output_path}")

    # Validation
    checks = {
        "Employee_Name removed": "Employee_Name" not in df.columns,
        "DOB removed":           "DOB"           not in df.columns,
        "AgeBracket present":    "AgeBracket"    in df.columns,
        "EmpID hashed (12 ch)":  df["EmpID"].str.len().eq(12).all(),
        "Zip masked":            df["Zip"].str.endswith("***").all(),
        "Row count intact":      len(df) == pd.read_csv(input_path, encoding="utf-8-sig").shape[0],
    }
    print("\n  Validation:")
    all_ok = True
    for check, ok in checks.items():
        print(f"    {'✅' if ok else '❌'}  {check}")
        if not ok:
            all_ok = False
    print(f"\n  {'✅ ALL CHECKS PASSED' if all_ok else '❌ SOME CHECKS FAILED'}")
    print("========================================\n")
    return df


if __name__ == "__main__":
    run()
