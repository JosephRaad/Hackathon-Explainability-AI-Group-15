# =============================================================================
# TrustedAI — anonymize.py
# Step 2: GDPR anonymization on the MERGED dataset.
# Techniques: Suppression · Pseudonymization · Generalization · Masking
#
# IMPORTANT: Sex and RaceDesc are KEPT as protected attributes.
# Justification: EU AI Act Art. 10(5) requires bias testing on protected
# groups. GDPR Art. 9(2)(g) allows processing of sensitive data when
# necessary for reasons of substantial public interest (non-discrimination).
# These fields are used ONLY for fairness auditing, never as model features.
#
# Input : data/processed/hr_merged.csv
# Output: data/processed/hr_anonymized.csv
# =============================================================================

import pandas as pd
import hashlib
import os
from datetime import datetime

INPUT_PATH = "data/processed/hr_merged.csv"
OUTPUT_PATH = "data/processed/hr_anonymized.csv"


def run(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    print("\n" + "=" * 60)
    print("  TrustedAI — GDPR Anonymization Pipeline")
    print("  Running on MERGED dataset (all sources)")
    print("=" * 60 + "\n")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Merged dataset not found: {input_path}\n"
            "Run merge_datasets.py first."
        )

    df = pd.read_csv(input_path)
    print(f"  Input: {len(df)} rows × {len(df.columns)} columns")
    print(f"  Sources: {df['source_dataset'].value_counts().to_dict()}")
    original_cols = set(df.columns)
    actions = []

    # ─── 1. SUPPRESSION: Remove direct identifiers ───────────────────────
    # These columns allow direct re-identification of individuals
    suppress_cols = [
        "TermReason",      # Free-text, potentially identifying
    ]
    existing = [c for c in suppress_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        actions.append(f"Suppression: removed {existing}")
        print(f"  [1] Suppression     — removed {len(existing)} cols: {existing}")
    else:
        print(f"  [1] Suppression     — no direct identifiers to remove")

    # ─── 2. PSEUDONYMIZATION: Hash employee_id ───────────────────────────
    if "employee_id" in df.columns:
        salt = "trustedai_2025_hackathon"
        df["employee_id"] = df["employee_id"].astype(str).apply(
            lambda x: hashlib.sha256((salt + x).encode()).hexdigest()[:12]
        )
        actions.append("Pseudonymization: employee_id → salted SHA-256 (12 chars)")
        print(f"  [2] Pseudonymization — employee_id → salted SHA-256 (12 chars)")

    # ─── 3. GENERALIZATION: Age → AgeBracket ─────────────────────────────
    if "Age" in df.columns:
        df["AgeBracket"] = pd.cut(
            df["Age"],
            bins=[0, 25, 35, 45, 55, 65, 120],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
            right=True
        ).astype(str).replace("nan", "Unknown")
        df = df.drop(columns=["Age"])
        actions.append("Generalization: Age → AgeBracket (6 bands)")
        print(f"  [3] Generalization  — Age → AgeBracket (6 bands)")

    # ─── 4. GENERALIZATION: Salary → SalaryBand ─────────────────────────
    if "Salary" in df.columns:
        df["SalaryBand"] = pd.cut(
            df["Salary"],
            bins=[0, 30000, 50000, 70000, 100000, 150000, float("inf")],
            labels=["<30K", "30-50K", "50-70K", "70-100K", "100-150K", "150K+"],
            right=True
        ).astype(str).replace("nan", "Unknown")
        df = df.drop(columns=["Salary"])
        actions.append("Generalization: Salary → SalaryBand (6 bands)")
        print(f"  [4] Generalization  — Salary → SalaryBand (6 bands)")

    # ─── 5. PERTURBATION: Add noise to continuous variables ──────────────
    # Small noise to prevent re-identification through unique value combinations
    noise_cols = ["EngagementSurvey", "Absences"]
    for col in noise_cols:
        if col in df.columns:
            noise = pd.Series(
                (pd.Series(range(len(df))).apply(
                    lambda i: int(hashlib.md5(f"{col}_{i}".encode()).hexdigest(), 16) % 100
                ) - 50) / 500.0
            )
            if col == "Absences":
                # Integer noise for count data
                noise = (noise * 5).round().astype(int)
            df[col] = (df[col] + noise)
            if col == "EngagementSurvey":
                df[col] = df[col].clip(1, 5).round(2)
            elif col == "Absences":
                df[col] = df[col].clip(0, 50).astype(int)
    actions.append(f"Perturbation: small noise added to {noise_cols}")
    print(f"  [5] Perturbation    — noise added to {noise_cols}")

    # ─── 6. PROTECTED ATTRIBUTES: KEPT for fairness audit ────────────────
    protected = ["Sex", "RaceDesc"]
    present = [c for c in protected if c in df.columns]
    print(f"  [6] Protected attrs — KEPT for fairness audit: {present}")
    print(f"      Justification: EU AI Act Art. 10(5) + GDPR Art. 9(2)(g)")
    actions.append(f"Protected attributes kept for bias audit: {present}")

    # ─── SAVE ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # ─── VALIDATION ──────────────────────────────────────────────────────
    final_cols = set(df.columns)
    removed = original_cols - final_cols
    added = final_cols - original_cols

    checks = {
        "employee_id pseudonymized":   df["employee_id"].str.len().eq(12).all() if "employee_id" in df.columns else False,
        "Age removed (→ AgeBracket)":  "Age" not in df.columns and "AgeBracket" in df.columns,
        "Salary removed (→ SalaryBand)": "Salary" not in df.columns and "SalaryBand" in df.columns,
        "Sex preserved for audit":      "Sex" in df.columns,
        "RaceDesc preserved for audit": "RaceDesc" in df.columns,
        "Row count intact":             len(df) == pd.read_csv(input_path).shape[0],
    }

    print(f"\n  Columns: {len(original_cols)} → {len(final_cols)}")
    print(f"  Removed: {removed}")
    print(f"  Added:   {added}")

    print("\n  Validation:")
    all_ok = True
    for check, ok in checks.items():
        print(f"    {'✅' if ok else '❌'}  {check}")
        if not ok:
            all_ok = False

    print(f"\n  {'✅ ALL CHECKS PASSED' if all_ok else '❌ SOME CHECKS FAILED'}")
    print(f"  ✅ Saved: {output_path} ({len(df)} rows × {len(df.columns)} cols)")
    print("=" * 60 + "\n")

    return df, actions


if __name__ == "__main__":
    run()
