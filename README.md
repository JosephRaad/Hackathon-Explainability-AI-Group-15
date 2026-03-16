# 🛡️ TrustedAI — HR Attrition Analytics System

> *A fair, secure, and explainable HR attrition prediction system built for the Capgemini × ESILV TrustedAI Hackathon.*

---

## 📌 Objectives

This system addresses a real enterprise problem: **predicting employee attrition risk while guaranteeing fairness, protecting privacy, and securing AI inputs.**

Specifically, the system:
- Predicts which employees are at risk of leaving (flight risk score)
- Audits and mitigates gender and racial bias in the predictions
- Enforces GDPR compliance on all HR data before processing
- Secures GenAI text analysis against prompt injection attacks
- Delivers everything through an interactive HR manager dashboard

---

## 🎯 Scope

| In Scope | Out of Scope |
|---|---|
| Binary attrition prediction (active / terminated) | Real-time streaming data |
| Gender + Race bias audit (AIF360) | Autonomous HR decision-making |
| GDPR anonymization pipeline | Employee surveillance |
| GenAI exit interview analysis | Re-identification from anonymized data |
| Streamlit HR dashboard | Production deployment |

---

## 👤 Persona

**Sarah — HR Analytics Manager**

Sarah manages a team of 300+ employees across 5 departments. She needs to:
- Identify employees at risk of leaving *before* they resign
- Ensure the prediction tool is not discriminating by gender or race
- Have qualitative context (exit interview summaries) alongside the numbers
- Demonstrate to legal/compliance that all data handling is GDPR-compliant

TrustedAI is built for Sarah.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TrustedAI Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HRDataset_v14.csv (local only, .gitignored)               │
│          │                                                  │
│          ▼                                                  │
│  ┌──────────────────┐   GDPR Pipeline (anonymize.py)       │
│  │  Suppression     │ ← Remove Employee_Name, ManagerName  │
│  │  Pseudonymization│ ← SHA-256 hash EmpID                 │
│  │  Generalization  │ ← DOB → Age Bracket                  │
│  │  Masking         │ ← ZIP → Regional prefix              │
│  └────────┬─────────┘                                       │
│           │  hr_anonymized.csv                              │
│           ▼                                                  │
│  ┌──────────────────┐   Feature Engineering (preprocess.py) │
│  │  Label Encoding  │                                        │
│  │  Median Impute   │                                        │
│  │  Feature Select  │                                        │
│  └────────┬─────────┘                                       │
│           │  hr_features.csv                                │
│           ▼                                                  │
│  ┌──────────────────────────────────┐                       │
│  │     bias_audit.py (AIF360)       │                       │
│  │  ┌───────────┐  ┌─────────────┐ │                       │
│  │  │ Baseline  │  │ Reweighing  │ │                       │
│  │  │  Model    │─▶│  Fair Model │ │                       │
│  │  └───────────┘  └─────────────┘ │                       │
│  │  Audit: Sex + RaceDesc           │                       │
│  └────────┬─────────────────────────┘                       │
│           │  predictions.csv + fairness_metrics.json        │
│           ▼                                                  │
│  ┌──────────────────────────────────┐                       │
│  │       Streamlit Dashboard        │                       │
│  │  Page 1: Flight Risk Table       │                       │
│  │  Page 2: Fairness Audit Report   │                       │
│  │  Page 3: Exit Interview Analyzer │ ← Claude API         │
│  │  Page 4: GDPR Compliance Report  │                       │
│  └──────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘

GenAI Security Layer (genai_analysis.py):
  Input → Sanitize → Injection Detect → Truncate → Claude API → JSON Parse
```

---

## 🚀 Instructions — How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1 — Place the dataset
Download `HRDataset_v14.csv` and place it in `data/raw/`.
This folder is `.gitignored` — the raw data never leaves your machine.

### Step 2 — Run the full pipeline
```bash
# 1. Anonymize (GDPR compliance)
python src/anonymize.py

# 2. Feature engineering
python src/preprocess.py

# 3. Train models + bias audit
python src/bias_audit.py

# 4. Launch dashboard
python -m streamlit run src/app.py
```

### Step 3 — Optional: Claude API for GenAI analysis
```bash
# Set your API key (Windows)
set ANTHROPIC_API_KEY=your_key_here

# Or create a .env file (also .gitignored)
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Step 4 — Run the notebook
```bash
cd notebooks
jupyter notebook 00_exploration_and_results.ipynb
```

---

## 🗂️ Repository Structure

```
trustedai-hr-hackathon/
├── data/
│   ├── raw/                    # .gitignored — never committed
│   └── processed/              # Anonymized outputs
├── src/
│   ├── anonymize.py            # GDPR pipeline
│   ├── preprocess.py           # Feature engineering
│   ├── bias_audit.py           # AIF360 audit + mitigation
│   ├── genai_analysis.py       # Claude API + injection guard
│   └── app.py                  # Streamlit dashboard
├── notebooks/
│   └── 00_exploration_and_results.ipynb
├── docs/
│   ├── architecture.png
│   ├── data_card.md
│   └── model_card.md
├── .gitignore
├── README.md
├── requirements.txt
└── executive_summary.md
```

---

## 📊 Dataset

- **Name:** Dr. Rich HRDataset_v14
- **Source:** [Kaggle — HR Analytics](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set)
- **Size:** 311 employees, 36 columns (original) → 26 columns (anonymized)
- **Target:** `Termd` (0 = active, 1 = terminated)
- **Termination rate:** 33.4%

See `docs/data_card.md` for full PII handling documentation.

---

## 🔬 Technical Stack

| Component | Technology |
|---|---|
| ML Model | Gradient Boosting Classifier (scikit-learn) |
| Fairness Auditing | IBM AIF360 (Reweighing) |
| GDPR Pipeline | Custom Python (hashlib, pandas) |
| GenAI Analysis | Anthropic Claude API |
| Dashboard | Streamlit |
| Visualization | Matplotlib, Seaborn |

---

## ⚖️ Responsible AI

- **Bias tested:** Gender (Sex) and Race (RaceDesc)
- **Metrics used:** Disparate Impact, Statistical Parity Difference, Equal Opportunity Difference
- **Mitigation:** AIF360 Reweighing (pre-processing)
- **Result:** Gender SPD reduced from 0.112 (❌ BIASED) to 0.053 (✅ FAIR)
- **Human oversight:** Tool is advisory only — no autonomous HR decisions
- **EU AI Act:** System designed for Annex III High-Risk AI compliance

---

## 👥 Team

Built during the Capgemini × ESILV TrustedAI Hackathon — March 2025.

---

*"Code that works. Models that are fair. Data that is safe."*
