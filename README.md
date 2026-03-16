# 🛡️ TrustedAI — HR Talent Retention Platform

> **Hackathon Trusted AI × HR** — Capgemini × ESILV — Group 15  
> **Themes**: AI & Cybersecurity · Ethical AI

## 🎯 Objective

An imaginary company faces high resignation rates and wants to use AI to understand the causes of turnover and retain its talent. TrustedAI is an AI-powered HR analytics platform that:

- **Predicts** which employees are at risk of leaving (with risk levels)
- **Explains** why using SHAP feature importance per prediction
- **Audits** the model for gender and racial bias using IBM AIF360
- **Protects** employee data through GDPR-compliant anonymization
- **Provides** an interactive chatbot for HR to query insights naturally

## 👤 Personae

| Persona | Description |
|---|---|
| **HR Director** (Client) | "I need to understand why my employees are leaving and what I can do about it." |
| **TrustedAI** (Provider) | "We build responsible AI solutions for HR: predictive, explainable, fair, and secure." |

## 🏗️ Architecture

```
data/
├── raw/                          # Original datasets (.gitignored)
│   ├── HRDataset_v14.csv        # Dr. Rich Huebner (~311 employees)
│   ├── IBM_HR_Attrition.csv     # IBM HR dataset (~1,470 employees)
│   └── HR_comma_sep.csv         # Kaggle HR dataset (~1,470 employees)
├── processed/                    # Pipeline outputs
│   ├── hr_merged.csv            # Step 1: Unified schema
│   ├── hr_anonymized.csv        # Step 2: GDPR-compliant
│   ├── hr_features.csv          # Step 3: Model-ready features
│   ├── hr_features_meta.json    # Feature metadata & mappings
│   ├── predictions.csv          # Step 4: All predictions + risk scores
│   ├── fairness_metrics.json    # Bias audit results
│   ├── model_fair.pkl           # Trained fair model
│   └── shap_values.pkl          # SHAP explainability data
docs/
├── data_card.md
├── model_card.md
└── *.png                        # Charts and diagrams
notebooks/
└── 00_exploration_and_results.ipynb
src/
├── merge_datasets.py            # Step 1: Merge 3 datasets + enrich
├── anonymize.py                 # Step 2: GDPR anonymization
├── preprocess.py                # Step 3: Feature engineering
├── bias_audit.py                # Step 4: AIF360 + SHAP + model
├── model_baseline.py            # Pipeline orchestrator
├── app.py                       # Streamlit dashboard
└── genai_analysis.py            # Exit interview NLP + security
```

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python src/model_baseline.py
```
This executes: `merge → anonymize → preprocess → bias_audit` in order.

### 3. Launch the dashboard
```bash
python -m streamlit run src/app.py
```

### 4. (Optional) Enable Claude API chatbot
```bash
export ANTHROPIC_API_KEY=your_key_here
python -m streamlit run src/app.py
```
Without an API key, the chatbot uses a local data-driven engine.

## 🔒 AI & Cybersecurity (Theme 1)

### GDPR Compliance
The anonymization pipeline applies **4 techniques** on the merged dataset:

| Technique | Applied To | Result |
|---|---|---|
| **Suppression** | TermReason (free text) | Removed entirely |
| **Pseudonymization** | employee_id | Salted SHA-256 hash (12 chars) |
| **Generalization** | Age, Salary | Brackets (18-25, 26-35...) / Bands |
| **Perturbation** | EngagementSurvey, Absences | Deterministic noise (±ε) |

**Protected attributes** (Sex, RaceDesc) are kept per EU AI Act Art. 10(5) for mandatory bias testing.

### Prompt Injection Protection
The exit interview analyzer uses a **5-layer security pipeline**:
1. Input sanitization (control character removal)
2. Regex-based injection pattern detection (17 patterns)
3. Length cap at 3,000 characters
4. Role-locked system prompt (JSON-only output)
5. Output validation and parsing

### EU AI Act Classification
This system is classified as **Annex III, Category 4 — HIGH RISK** (AI in employment decisions).

| Requirement | Implementation |
|---|---|
| Risk Management (Art. 9) | AIF360 bias audit pipeline |
| Data Governance (Art. 10) | GDPR anonymization pipeline |
| Transparency (Art. 13) | Model Card, Data Card, SHAP |
| Human Oversight (Art. 14) | Advisory only — no autonomous decisions |
| Robustness (Art. 15) | Cross-validated, injection-protected |

## ⚖️ Ethical AI (Theme 2)

### Bias Audit with IBM AIF360
- **Protected attributes**: Sex (Gender) and RaceDesc (Race)
- **Method**: AIF360 Reweighing — adjusts training weights to equalize outcomes
- **Metrics**: Statistical Parity Difference (SPD), Disparate Impact (DI), Equal Opportunity Difference (EOD)
- **Threshold**: SPD < ±0.10 (industry standard)
- Protected attributes are **excluded from model features** and used only for auditing

### Explainability with SHAP
- **SHAP** (SHapley Additive exPlanations) computes feature importance
- Per-prediction explanations: "Employee X is high risk because their engagement is low and absences are high"
- Global feature importance ranking displayed on the dashboard

## 📊 Model Details

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Estimators | 150 |
| Max Depth | 3 |
| Learning Rate | 0.08 |
| Features | 15 (numeric + encoded categorical) |
| Protected Attributes | Excluded from features, audited separately |
| Fairness Method | AIF360 Reweighing (pre-processing) |
| Explainability | SHAP TreeExplainer |

## 🤖 Chatbot Capabilities

The AI chatbot can answer data-driven questions such as:
- "How many high-risk employees per department?"
- "What are the main departure causes?"
- "What measures to reduce attrition?"
- "Explain the bias audit results"
- "What is the EU AI Act risk level?"
- "How does the model work?"

It queries live prediction data and returns specific numbers and recommendations.

## 📋 Deliverables Checklist

- [x] Clear README with objectives, scope, personae, instructions
- [x] Technical documentation (Data Card, Model Card)
- [x] Architecture scheme
- [x] Data Card & Model Card
- [x] Demo-ready Streamlit dashboard
- [x] Executive summary
- [x] Reproducible pipeline (`python src/model_baseline.py`)

## 👥 Team — Group 15

Capgemini × ESILV TrustedAI Hackathon 2025
