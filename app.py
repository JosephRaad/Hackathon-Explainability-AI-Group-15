"""
Streamlit Demo — Trusted AI x HR
Combines all modules: anonymization, prediction, fairness audit, chatbot security.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Paths
BASE = Path(__file__).parent
RAW_PATH = BASE / "data" / "raw" / "HRDataset_v14.csv"
CLEAN_PATH = BASE / "data" / "processed" / "hr_anonymized.csv"
MODEL_PATH = BASE / "models" / "xgb_turnover.json"
PLOTS_DIR = BASE / "outputs" / "plots"

# Import project modules
from src.anonymize import PII_COLUMNS
from src.model import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, prepare_features
from src.fairness import compute_fairness_metrics, print_fairness_report
from src.chatbot import chat, detect_injection, sanitize_input, DEMO_ATTACKS

st.set_page_config(page_title="Trusted AI x HR", page_icon="🛡️", layout="wide")
st.title("🛡️ Trusted AI x HR — Employee Turnover Analysis")
st.markdown("**Topics: AI Cybersecurity & AI Ethics** | Group 15")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data & Anonymization",
    "🔮 Turnover Prediction & SHAP",
    "⚖️ Fairness Audit",
    "🤖 Secure HR Chatbot",
])

# ============================================================
# Tab 1: Data Anonymization (Cybersecurity)
# ============================================================
with tab1:
    st.header("Data Anonymization — GDPR Compliance")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("❌ Before: Raw Data (with PII)")
        raw_df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
        st.dataframe(raw_df[["Employee_Name", "DOB", "Zip", "ManagerName", "Salary", "Termd"]].head(10))
        st.warning(f"PII columns present: {PII_COLUMNS}")

    with col2:
        st.subheader("✅ After: Anonymized Data")
        clean_df = pd.read_csv(CLEAN_PATH)
        st.dataframe(clean_df[["EmpID", "Salary", "Department", "Sex", "RaceDesc", "Termd"]].head(10))
        st.success(f"PII columns removed. {len(clean_df.columns)} columns retained.")

    st.markdown("### AI Act Risk Classification")
    st.info(
        "**Risk Level: HIGH** — This AI system falls under AI Act Article 6, Annex III, "
        "Category 4: 'AI systems intended to be used for recruitment or selection of natural persons, "
        "or to make decisions affecting terms of work-related relationships.' "
        "High-risk systems require: conformity assessment, transparency obligations, "
        "human oversight, and technical documentation."
    )

# ============================================================
# Tab 2: Prediction + SHAP (Explainability bonus)
# ============================================================
with tab2:
    st.header("Turnover Prediction with XGBoost")

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    df = pd.read_csv(CLEAN_PATH)
    X, y = prepare_features(df)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.1%}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.1%}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.1%}")

    # SHAP
    st.subheader("SHAP — Why does the model predict turnover?")
    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:
        if (PLOTS_DIR / "shap_summary.png").exists():
            st.image(str(PLOTS_DIR / "shap_summary.png"), caption="Global Feature Importance")

    with shap_col2:
        if (PLOTS_DIR / "shap_waterfall_example.png").exists():
            st.image(str(PLOTS_DIR / "shap_waterfall_example.png"), caption="Single Employee Explanation")

    # Individual employee prediction
    st.subheader("🔍 Predict for a specific employee")
    emp_idx = st.selectbox("Select employee (by test set index)", X_test.index.tolist())
    emp_data = X_test.loc[emp_idx]
    emp_prob = model.predict_proba(emp_data.values.reshape(1, -1))[0][1]
    risk = "🔴 HIGH RISK" if emp_prob > 0.5 else "🟡 MEDIUM RISK" if emp_prob > 0.3 else "🟢 LOW RISK"
    st.markdown(f"**Turnover probability: {emp_prob:.1%}** — {risk}")

# ============================================================
# Tab 3: Fairness Audit (AI Ethics)
# ============================================================
with tab3:
    st.header("Fairness Audit — AI Ethics")

    df = pd.read_csv(CLEAN_PATH)
    X, y = prepare_features(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    y_pred = model.predict(X_test)
    test_indices = X_test.index

    # Sex audit
    st.subheader("Gender Fairness")
    sex_test = df.loc[test_indices, "Sex"]
    sex_metrics = compute_fairness_metrics(y_test.values, y_pred, sex_test.values, "M")

    col1, col2, col3 = st.columns(3)
    col1.metric("Demographic Parity Diff", f"{sex_metrics['demographic_parity_diff']:+.3f}", delta="ideal: 0")
    col2.metric("Disparate Impact Ratio", f"{sex_metrics['disparate_impact_ratio']:.3f}", delta="ideal: 1.0")
    col3.metric("Equalized Odds Diff", f"{sex_metrics['equalized_odds_diff']:+.3f}", delta="ideal: 0")

    if abs(sex_metrics["demographic_parity_diff"]) > 0.1:
        st.error(
            f"⚠️ Gender bias detected: The model predicts termination for women at a rate of "
            f"{sex_metrics['unprivileged_pred_rate']:.1%} vs {sex_metrics['privileged_pred_rate']:.1%} for men. "
            f"This {sex_metrics['demographic_parity_diff']:+.1%} gap exceeds the ±10% fairness threshold."
        )

    # Race audit
    st.subheader("Racial Fairness")
    race_test = df.loc[test_indices, "RaceDesc"]
    race_binary = (race_test == "White").map({True: "White", False: "Non-White"})
    race_metrics = compute_fairness_metrics(y_test.values, y_pred, race_binary.values, "White")

    col1, col2, col3 = st.columns(3)
    col1.metric("Demographic Parity Diff", f"{race_metrics['demographic_parity_diff']:+.3f}", delta="ideal: 0")
    col2.metric("Disparate Impact Ratio", f"{race_metrics['disparate_impact_ratio']:.3f}", delta="ideal: 1.0")
    col3.metric("Equalized Odds Diff", f"{race_metrics['equalized_odds_diff']:+.3f}", delta="ideal: 0")

    if abs(race_metrics["demographic_parity_diff"]) > 0.1:
        st.error(
            f"⚠️ Racial bias detected: The model predicts termination for non-White employees at "
            f"{race_metrics['unprivileged_pred_rate']:.1%} vs {race_metrics['privileged_pred_rate']:.1%} for White employees."
        )

    # Fairness plot
    if (PLOTS_DIR / "fairness_comparison.png").exists():
        st.image(str(PLOTS_DIR / "fairness_comparison.png"), caption="Prediction Rate by Group")

    st.markdown("### Recommendations")
    st.markdown("""
    1. **Re-examine training data** for historical bias in termination decisions
    2. **Apply bias mitigation** (e.g., reweighting, adversarial debiasing via AIF360)
    3. **Monitor fairness metrics** continuously in production
    4. **Human oversight required** — model predictions should inform, not replace, HR decisions
    """)

# ============================================================
# Tab 4: Secure HR Chatbot (Cybersecurity)
# ============================================================
with tab4:
    st.header("Secure HR Chatbot — Prompt Injection Defense")

    st.markdown("""
    This chatbot uses Claude to answer HR analytics questions.
    It includes **multi-layer security**:
    1. **Input sanitization** — cleans and limits user input
    2. **Injection detection** — regex patterns catch common attacks
    3. **System prompt hardening** — strict role boundaries in the LLM prompt
    """)

    # Demo attacks section
    st.subheader("🎯 Attack Demo")
    st.markdown("Try these pre-built attack scenarios to see the defense in action:")

    for attack in DEMO_ATTACKS:
        with st.expander(f"Attack: {attack['name']}"):
            st.code(attack["input"])
            is_injection, pattern = detect_injection(attack["input"])
            if is_injection:
                st.error(f"⚠️ BLOCKED — Matched pattern: `{pattern}`")
            else:
                st.success("✅ Passed — Safe to send to LLM")

    # Interactive chat
    st.subheader("💬 Try it yourself")
    user_input = st.text_input("Ask an HR analytics question (or try an injection attack):")

    if user_input:
        is_injection, pattern = detect_injection(user_input)
        if is_injection:
            st.error(f"⚠️ **Prompt injection detected!** Matched pattern: `{pattern}`")
            st.info("Your message was blocked before reaching the LLM.")
        else:
            with st.spinner("Querying Claude..."):
                try:
                    response = chat(user_input)
                    st.markdown(response)
                except Exception as e:
                    st.warning(f"LLM call failed (API key may not be set): {e}")
                    st.info("Set the ANTHROPIC_API_KEY environment variable to enable the chatbot.")
