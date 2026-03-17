# 🛡️ TrustedAI   HR Attrition Analytics System

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

**Sarah   HR Analytics Manager**

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
│  3 Raw Datasets (data/raw/   .gitignored)                   │
│  Dr. Rich (311) + IBM (1,470) + Kaggle (1,480)              │
│          │                                                  │
│          ▼  merge_datasets.py                               │
│  hr_merged.csv (3,261 rows × 24 cols)                       │
│  + stats_drrich.json / stats_ibm.json / stats_kaggle.json   │
│  + stats_merged.json  (per-source provenance snapshots)     │
│          │                                                  │
│          ▼  anonymize.py                                    │
│  ┌──────────────────┐   GDPR Pipeline                       │
│  │  Suppression     │ ← Remove TermReason                   │
│  │  Pseudonymization│ ← SHA-256 hash employee_id            │
│  │  Generalization  │ ← Age → Bracket, Salary → Band       │
│  │  Perturbation    │ ← Noise on continuous vars            │
│  └────────┬─────────┘                                       │
│           │  hr_anonymized.csv (3,261 rows × 23 cols)       │
│           ▼  preprocess.py                                  │
│  ┌──────────────────┐   Feature Engineering                 │
│  │  Label Encoding  │   15 model features                   │
│  │  Median Impute   │   2 protected (audit only)            │
│  │  Feature Select  │                                       │
│  └────────┬─────────┘                                       │
│           │  hr_features.csv                                │
│           ▼  bias_audit.py                                  │
│  ┌──────────────────────────────────┐                       │
│  │     AIF360 Fairness Pipeline     │                       │
│  │  ┌───────────┐  ┌─────────────┐ │                       │
│  │  │ Baseline  │  │ Reweighing  │ │                       │
│  │  │  Model    │─▶│  Fair Model │ │                       │
│  │  └───────────┘  └─────────────┘ │                       │
│  │  Audit: Sex + RaceDesc           │                       │
│  │  + SHAP Explainability           │                       │
│  └────────┬─────────────────────────┘                       │
│           │  predictions.csv + fairness_metrics.json        │
│           ▼  app.py                                         │
│  ┌──────────────────────────────────┐                       │
│  │       Streamlit Dashboard        │                       │
│  │  Page 1: Flight Risk Table       │                       │
│  │  Page 2: Fairness Audit Report   │                       │
│  │  Page 3: AI Chatbot              │ ← 14-intent hybrid   │
│  │  Page 4: Exit Interview Analyzer │ ← 5-layer security   │
│  │  Page 5: GDPR Compliance Report  │                       │
│  └──────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘

GenAI Security Layer (genai_analysis.py):
  Input → Sanitize → Injection Detect (17 patterns) → Truncate
       → Role-locked Prompt → JSON Parse

Chatbot Hybrid Engine (app.py):
  Question → 14-intent local matcher (synonym-aware)
          → Live data answer (no API needed)
          → Claude API fallback for complex freeform questions
          → Out-of-scope guard (stock prices, coding tasks, etc.)
```

---

## 🚀 Instructions   How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1   Place the datasets
Place the following files in `data/raw/`:
- `HRDataset_v14.csv` (required)
- `IBM_HR_Attrition.csv` (optional)
- `HR_comma_sep.csv` (optional)

This folder is `.gitignored`   raw data never leaves your machine.

### Step 2   Run the full pipeline
```bash
python src/model_baseline.py
```

This runs all 4 steps in order:
1. **Merge**   Combines datasets into unified schema (3,261 rows) + exports per-source stats JSONs
2. **Anonymize**   GDPR pipeline (4 techniques)
3. **Preprocess**   Feature engineering (15 features + 2 protected)
4. **Bias Audit**   AIF360 Reweighing + SHAP explainability

### Step 3   Launch the dashboard
```bash
python -m streamlit run src/app.py
```

The dashboard runs **fully locally**   no API key required.

### Step 4   Optional: Claude API for GenAI analysis
```bash
export ANTHROPIC_API_KEY=your_key_here
# Or create a .env file (also .gitignored)
```

When the API key is set, the chatbot escalates complex freeform HR questions
to Claude automatically. Without it, the 14-intent local engine handles all
standard queries.

### Step 5   Run the notebook
```bash
jupyter notebook notebooks/00_exploration_and_results.ipynb
```

---

## 🗂️ Repository Structure

```
trustedai-hr-analytics/
├── data/
│   ├── raw/                          # .gitignored   never committed
│   └── processed/                    # Pipeline outputs
│       ├── hr_merged.csv             #   3,261 rows × 24 cols
│       ├── hr_anonymized.csv         #   3,261 rows × 23 cols (0 PII)
│       ├── hr_features.csv           #   Model-ready features
│       ├── hr_features_meta.json     #   Feature metadata + label mappings
│       ├── predictions.csv           #   All employees with risk scores
│       ├── fairness_metrics.json     #   AIF360 audit results
│       ├── model_fair.pkl            #   Trained GradientBoosting model
│       ├── shap_values.pkl           #   SHAP explainability data
│       ├── stats_drrich.json         #   Per-source stats (Dr. Rich)
│       ├── stats_ibm.json            #   Per-source stats (IBM)
│       ├── stats_kaggle.json         #   Per-source stats (Kaggle)
│       └── stats_merged.json         #   Combined stats snapshot
├── src/
│   ├── merge_datasets.py             # Step 1: Multi-dataset merge + stats export
│   ├── anonymize.py                  # Step 2: GDPR pipeline
│   ├── preprocess.py                 # Step 3: Feature engineering
│   ├── model_baseline.py             # Pipeline orchestrator
│   ├── bias_audit.py                 # Step 4: AIF360 audit + SHAP
│   ├── genai_analysis.py             # GenAI security + NLP
│   └── app.py                        # Streamlit dashboard (5 pages)
├── notebooks/
│   └── 00_exploration_and_results.ipynb
├── docs/
│   ├── data_card.md
│   ├── model_card.md
│   └── consolidation_report.md
├── .streamlit/
│   └── config.toml                   # Theme configuration
├── .gitignore
├── README.md
├── requirements.txt
└── executive_summary.md
```

---

## 📊 Dataset

- **Primary:** Dr. Rich HRDataset_v14   311 employees, 36 columns
- **Secondary:** IBM HR Attrition   1,470 employees
- **Secondary:** Kaggle HR Analytics   1,480 employees
- **Combined:** 3,261 rows after schema-normalized merge
- **Target:** `Termd` (0 = active, 1 = terminated)
- **Combined attrition rate:** 17.8%
- **Data provenance:** Each source queryable independently via chatbot
  ("Tell me about each dataset") and stats JSON snapshots

See `docs/data_card.md` for full PII handling documentation.

---

## 🔬 Technical Stack

| Component | Technology |
|---|---|
| ML Model | Gradient Boosting Classifier (scikit-learn) |
| Fairness Auditing | IBM AIF360 (Reweighing + Threshold Equalization) |
| Explainability | SHAP TreeExplainer |
| GDPR Pipeline | Custom Python (hashlib, pandas) |
| GenAI Analysis | Anthropic Claude API + Local NLP Fallback |
| Dashboard | Streamlit (5 pages, Zoho CRM-inspired design) |
| Chatbot Engine | 14-intent hybrid matcher + Claude API escalation |

---

## ⚖️ Responsible AI

- **Bias tested:** Gender (Sex) and Race (RaceDesc)
- **Metrics used:** Disparate Impact, Statistical Parity Difference, Equal Opportunity Difference
- **Mitigation:** AIF360 Reweighing (pre-processing) + group-threshold equalization (post-processing)
- **Result:** Race SPD reduced from -0.238 (❌ BIASED) to 0.028 (✅ FAIR)
- **Gender:** Already within threshold (SPD = -0.018)
- **Human oversight:** Tool is advisory only   no autonomous HR decisions
- **EU AI Act:** System designed for Annex III High-Risk AI compliance

---

## 🤖 Chatbot Capabilities

The AI chatbot answers 14 categories of HR analytics questions locally
with no API dependency:

| Intent | Example Question |
|---|---|
| Department attrition | "Which department has the highest attrition rate?" |
| Department risk | "How many high-risk employees per department?" |
| Fairness score | "What is the gender fairness score after correction?" |
| Top risk employees | "Show me the top 5 employees most at risk" |
| Salary comparison | "Average income of employees who left vs stayed?" |
| Overtime correlation | "Does overtime correlate with higher attrition?" |
| Dataset provenance | "Tell me about each dataset" |
| Retention measures | "What measures reduce attrition?" |
| Departure causes | "What are the top departure causes?" |
| Risk overview | "Give me a risk summary" |
| GDPR | "How is GDPR compliance handled?" |
| EU AI Act | "What is the EU AI Act classification?" |
| Model info | "How does the model work?" |
| Exit security | "How is injection detection implemented?" |

Complex freeform questions outside these categories are escalated to
Claude API when a key is available, with graceful local fallback otherwise.

---

## 👥 Team

Built during the **Capgemini × ESILV TrustedAI Hackathon**   March 2025.

---

*"Code that works. Models that are fair. Data that is safe."*
