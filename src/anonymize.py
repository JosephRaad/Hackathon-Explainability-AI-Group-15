"""
Data Anonymization Module — GDPR Compliance

Removes personally identifiable information (PII) from the HR dataset.
Produces a clean, anonymized dataset for all downstream analysis.

PII removal strategy:
- Employee_Name: REMOVED (direct identifier)
- DOB: REMOVED (quasi-identifier, can re-identify with other fields)
- Zip: REMOVED (quasi-identifier, location can narrow down identity)
- ManagerName: REMOVED (use ManagerID instead, already present)

Retained fields:
- EmpID: numeric, not a real-world identifier
- ManagerID: numeric reference, no PII
- Sex, RaceDesc: kept for fairness audit (AI Ethics), but are sensitive attributes
"""

import pandas as pd
from pathlib import Path

# Columns that constitute PII or quasi-identifiers
PII_COLUMNS = ["Employee_Name", "DOB", "Zip", "ManagerName"]

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "HRDataset_v14.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "hr_anonymized.csv"


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, encoding="utf-8-sig")


def anonymize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=PII_COLUMNS)

    # Clean known dirty values in TermReason
    dirty_reasons = {"Learned that he is a gangster", "Fatal attraction"}
    df["TermReason"] = df["TermReason"].replace(dirty_reasons, "Other")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    return df


def main():
    df = load_raw_data()
    print(f"Raw data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Removing PII columns: {PII_COLUMNS}")

    df_clean = anonymize(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)

    print(f"Anonymized data: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
