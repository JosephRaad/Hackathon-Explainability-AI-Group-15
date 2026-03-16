# 📦 Data Card — HRDataset_v14 (Anonymized)

**Version:** 1.0 | **Date:** March 2025 | **Team:** TrustedAI

---

## Dataset Identification

| Field | Value |
|---|---|
| **Name** | HRDataset_v14 (TrustedAI anonymized edition) |
| **Original Source** | Dr. Rich Huebner — [Kaggle](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set) |
| **Original Size** | 311 rows × 36 columns |
| **Processed Size** | 311 rows × 26 columns |
| **Format** | CSV |
| **Language** | English |

---

## Data Description

The dataset represents HR records for a fictional company, including employee demographics, performance metrics, engagement scores, and termination status.

**Target variable:** `Termd` (binary: 0 = active, 1 = terminated)
**Termination rate:** 33.4% (104 terminated / 207 active)

---

## Sensitive Attributes

| Column | Sensitivity | Handling |
|---|---|---|
| `Employee_Name` | Direct identifier | **Suppressed** (removed entirely) |
| `ManagerName` | Direct identifier | **Suppressed** (removed entirely) |
| `EmpID` | Quasi-identifier | **Pseudonymized** (SHA-256, 12 chars) |
| `DOB` | Quasi-identifier | **Generalized** (→ AgeBracket) |
| `Zip` | Quasi-identifier | **Masked** (→ 2-digit regional prefix) |
| `Sex` | Protected attribute | Retained for fairness audit only |
| `RaceDesc` | Protected attribute | Retained for fairness audit only |
| `HispanicLatino` | Sensitive | Retained — not used in modeling |

---

## GDPR Anonymization Applied

### Technique 1 — Suppression
Columns removed: `Employee_Name`, `ManagerName`, `ManagerID`, `MarriedID`, `GenderID`, `EmpStatusID`, `DeptID`, `PerfScoreID`, `PositionID`, `FromDiversityJobFairID`

**Legal basis:** GDPR Article 17 (Right to erasure) + Article 5(1)(c) (Data minimisation)

### Technique 2 — Pseudonymization
`EmpID` replaced with `hashlib.sha256(str(id).encode()).hexdigest()[:12]`

Referential integrity preserved. Re-identification requires the original EmpID values, which are not stored.

**Legal basis:** GDPR Recital 26 (Pseudonymized data)

### Technique 3 — Generalization
`DOB` (exact date) → `AgeBracket` (bands: Under 25, 25–34, 35–44, 45–54, 55–64, 65+)

**Legal basis:** GDPR Article 5(1)(c) (Data minimisation)

### Technique 4 — Masking
`Zip` (5-digit) → first 2 digits + `***` (e.g., `01601` → `01***`)

**Legal basis:** GDPR Article 5(1)(c) (Data minimisation)

---

## GDPR Principles Compliance

| Principle | Status | Evidence |
|---|---|---|
| Lawfulness, fairness, transparency | ✅ | Pipeline documented, open-source |
| Purpose limitation | ✅ | HR analytics only — no secondary use |
| Data minimisation | ✅ | 10 columns removed/transformed |
| Accuracy | ✅ | Source dataset unchanged |
| Storage limitation | ✅ | Raw data `.gitignored`, not committed |
| Integrity & confidentiality | ✅ | No PII in processed outputs |
| Accountability | ✅ | Reproducible pipeline, audit trail |

---

## Intended & Prohibited Uses

**Intended use:**
- Internal HR attrition risk prediction (advisory)
- Fairness auditing of ML models
- HR analytics dashboard demonstration

**Prohibited use:**
- Re-identification of individual employees
- Autonomous hiring or firing decisions
- Sharing with third parties
- Any purpose outside HR analytics in the original organizational context

---

## Data Lineage

```
HRDataset_v14.csv  (local, .gitignored)
    → anonymize.py
hr_anonymized.csv  (no PII)
    → preprocess.py
hr_features.csv    (model-ready, encoded)
```

---
---

# 🤖 Model Card — TrustedAI Fair Attrition Classifier

**Version:** 1.0 | **Date:** March 2025 | **Team:** TrustedAI

---

## Model Identification

| Field | Value |
|---|---|
| **Model type** | Gradient Boosting Classifier |
| **Framework** | scikit-learn 1.7.2 |
| **Task** | Binary classification (attrition prediction) |
| **Input** | Anonymized HR features (9 columns) |
| **Output** | Probability score + risk level (Low/Medium/High) |

---

## Model Architecture

**Algorithm:** `sklearn.ensemble.GradientBoostingClassifier`

**Hyperparameters:**
```
n_estimators  = 100
learning_rate = 0.1
max_depth     = 3
random_state  = 42
```

**Why Gradient Boosting?**
- Native support for `sample_weight` (required for AIF360 Reweighing)
- Built-in feature importance (explainability for jury/stakeholders)
- Strong performance on tabular data without GPU requirements
- More interpretable than deep learning for HR decision support

---

## Training Data

| Property | Value |
|---|---|
| Source | hr_features.csv (anonymized, preprocessed) |
| Split | 80% train / 20% test (stratified on target) |
| Train size | 248 samples |
| Test size | 63 samples |
| Class balance | ~33.4% positive (terminated) |

---

## Input Features

| Feature | Type | Description |
|---|---|---|
| `EngagementSurvey` | Numeric | Employee engagement score (1–5) |
| `EmpSatisfaction` | Numeric | Satisfaction rating (1–5) |
| `SpecialProjectsCount` | Numeric | Count of special projects |
| `Absences` | Numeric | Days absent in last year |
| `DaysLateLast30` | Numeric | Days late in last 30 days |
| `Department` | Categorical (encoded) | Employee department |
| `PerformanceScore` | Categorical (encoded) | Performance rating |
| `RecruitmentSource` | Categorical (encoded) | How employee was recruited |
| `MaritalDesc` | Categorical (encoded) | Marital status |

**Protected attributes (audited, NOT used as features):** `Sex`, `RaceDesc`

---

## Performance Metrics

| Metric | Baseline Model | Fair Model |
|---|---|---|
| Accuracy | 68.3% | **71.4%** |
| Train / Test split | 80/20 stratified | Same |

---

## Fairness Audit — IBM AIF360

### Gender (Sex) Audit

| Metric | Baseline | Fair Model | Threshold | Status |
|---|---|---|---|---|
| Disparate Impact | 1.1724 | 1.0747 | 0.8–1.2 | ✅ Both pass |
| Statistical Parity Diff | **0.1116** | **0.0527** | < 0.10 | ❌ → ✅ Fixed |
| Equal Opp. Difference | 0.0069 | -0.0801 | < 0.10 | ✅ Both pass |

**Key finding:** Statistical Parity Difference exceeded the 0.10 threshold in the baseline model. After Reweighing, it dropped from 0.1116 to 0.0527 — a 52.8% improvement. Accuracy simultaneously improved by +3.2%.

### Race (RaceDesc) Audit

| Metric | Baseline | Fair Model | Status |
|---|---|---|---|
| Disparate Impact | 0.9043 | 0.9043 | ✅ No change needed |
| Statistical Parity Diff | -0.0687 | -0.0687 | ✅ Within threshold |
| Equal Opp. Difference | 0.0238 | 0.0238 | ✅ Within threshold |

---

## Bias Mitigation Method — Reweighing

**Algorithm:** `aif360.algorithms.preprocessing.Reweighing`

**How it works:**
1. Compute the expected vs. observed probability of favorable outcome for each (group × label) combination
2. Assign instance weight = expected / observed
3. Pass weights to classifier via `sample_weight` parameter

**Why Reweighing over alternatives:**
- Pre-processing approach — no model architecture change
- Transparent and auditable
- Works with any scikit-learn estimator
- Compatible with EU AI Act requirements for High-Risk AI systems

---

## Ethical Considerations

**This model is advisory only.** It produces risk scores to support — not replace — human HR judgment.

**Known limitations:**
- Dataset contains 311 samples — small by production standards
- Binary gender encoding does not capture full gender spectrum
- Protected attribute encoding (LabelEncoder) may not preserve ordinal relationships for race categories
- Reweighing weight range [0.970, 1.043] was narrow — stronger bias would produce wider weights

**Recommended safeguards in production:**
- Human review required for all High-risk predictions before action
- Quarterly bias re-audit as employee population evolves
- Confidence intervals displayed alongside point estimates
- Opt-out mechanism for employees

---

## EU AI Act Compliance

This system falls under **Annex III, Category 4** (High-Risk AI in employment):
> *"AI systems used for recruitment, promotion decisions, task allocation, monitoring, evaluation of performance and behavior, or termination of work-related contractual relationships."*

Compliance measures implemented:
- ✅ Risk management system (bias audit pipeline)
- ✅ Data governance (GDPR anonymization pipeline)
- ✅ Transparency (model card, data card)
- ✅ Human oversight (advisory tool, no autonomous decisions)
- ✅ Accuracy and robustness documentation
- ✅ Technical documentation (this card + README)

---

## Model Lineage

```
hr_features.csv
    → bias_audit.py (GradientBoostingClassifier + AIF360 Reweighing)
    → model_fair.pkl      (primary model — used in dashboard)
    → predictions.csv     (test set predictions + risk scores)
    → fairness_metrics.json (all audit metrics)
```

---
*TrustedAI | Capgemini × ESILV Hackathon 2025*
