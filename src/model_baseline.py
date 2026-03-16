# =============================================================================
# TrustedAI — model_baseline.py
# Orchestrator: runs the full pipeline in the correct order.
# Usage: python src/model_baseline.py
# =============================================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_pipeline():
    print("\n" + "🛡️" * 20)
    print("  TrustedAI — Full Pipeline Execution")
    print("  Themes: AI & Cybersecurity · Ethical AI")
    print("🛡️" * 20 + "\n")

    # Step 1: Merge datasets
    print("=" * 60)
    print("  STEP 1/4 — Merge Datasets")
    print("=" * 60)
    from merge_datasets import merge_datasets
    df_merged = merge_datasets()
    if df_merged is None:
        print("  ❌ Merge failed. Aborting.")
        return

    # Step 2: Anonymize (GDPR)
    print("=" * 60)
    print("  STEP 2/4 — GDPR Anonymization")
    print("=" * 60)
    from anonymize import run as anonymize
    df_anon, actions = anonymize()

    # Step 3: Feature engineering
    print("=" * 60)
    print("  STEP 3/4 — Feature Engineering")
    print("=" * 60)
    from preprocess import run as preprocess
    df_features = preprocess()

    # Step 4: Bias audit + model training + SHAP
    print("=" * 60)
    print("  STEP 4/4 — Model Training + Bias Audit + SHAP")
    print("=" * 60)
    from bias_audit import run as bias_audit
    metrics = bias_audit()

    # Summary
    print("\n" + "🛡️" * 20)
    print("  PIPELINE COMPLETE")
    print("🛡️" * 20)
    print(f"""
  Outputs:
    data/processed/hr_merged.csv        — Unified dataset
    data/processed/hr_anonymized.csv    — GDPR-compliant data
    data/processed/hr_features.csv      — Model-ready features
    data/processed/predictions.csv      — All employee predictions
    data/processed/fairness_metrics.json — Audit results
    data/processed/model_fair.pkl       — Trained fair model
    data/processed/shap_values.pkl      — SHAP explainability data

  To launch the dashboard:
    python -m streamlit run src/app.py
    """)

    return metrics


if __name__ == "__main__":
    run_pipeline()
