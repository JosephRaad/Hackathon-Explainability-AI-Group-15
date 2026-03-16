# Trusted AI x HR — Employee Turnover Analysis

**Hackathon Explainability AI — Group 15**

An end-to-end AI system that predicts employee turnover while demonstrating **AI cybersecurity** (data anonymization, prompt injection defense) and **AI ethics** (SHAP explainability, fairness auditing).

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key (for the chatbot)

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

> The chatbot tab requires a valid Anthropic API key. All other tabs work without it.

### 3. Run the app

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                          # Streamlit main application (4 tabs)
├── requirements.txt                # Python dependencies
├── src/
│   ├── anonymize.py                # GDPR-compliant data anonymization
│   ├── model.py                    # XGBoost turnover prediction + SHAP
│   ├── fairness.py                 # Bias detection (gender, race)
│   └── chatbot.py                  # LLM chatbot with prompt injection defense
├── data/
│   ├── raw/HRDataset_v14.csv       # Original dataset (Kaggle HR Dataset)
│   └── processed/hr_anonymized.csv # Anonymized dataset (PII removed)
├── models/
│   └── xgb_turnover.json           # Pre-trained XGBoost model
└── outputs/plots/                  # SHAP & fairness visualizations
```

## Features

### Tab 1: Data Anonymization (Cybersecurity)
- Removes PII columns (`Employee_Name`, `DOB`, `Zip`, `ManagerName`)
- Before/after comparison of raw vs anonymized data
- EU AI Act risk classification

### Tab 2: Turnover Prediction & SHAP (Explainability)
- XGBoost model with engineered features (83% accuracy)
- SHAP summary and waterfall plots for global/local explanations
- Individual employee risk scoring

### Tab 3: Fairness Audit (AI Ethics)
- Demographic parity, disparate impact, equalized odds metrics
- Gender and racial bias detection
- Actionable recommendations for bias mitigation

### Tab 4: Secure HR Chatbot (Cybersecurity)
- Claude-powered chatbot with **real data-driven answers** (dynamically computed from dataset)
- Multi-layer prompt injection defense:
  - Input sanitization
  - Regex-based injection pattern detection
  - System prompt hardening with strict role boundaries
- Pre-built attack demo scenarios

## Retrain the Model (optional)

The pre-trained model is included. To retrain from scratch:

```bash
python -m src.model
```

## Regenerate Anonymized Data (optional)

```bash
python -m src.anonymize
```

## Dataset

[HRDataset_v14](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set) from Kaggle — a synthetic HR dataset with 311 employees and 35 features.

## Team

Group 15 — ESILV Hackathon Explainability AI
