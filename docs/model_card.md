# 🧠 Model Card — TrustedAI Attrition Predictor

## Model Overview

| Property | Value |
|---|---|
| Name | TrustedAI Fair Attrition Model |
| Version | 1.0 |
| Type | Binary Classification |
| Algorithm | Gradient Boosting Classifier (scikit-learn) |
| Task | Predict employee attrition risk |
| Output | Risk score (0-1) + risk level (Low/Medium/High) |

## Architecture Choice

**Why Gradient Boosting?**
- Strong performance on tabular data with modest dataset sizes
- Native `sample_weight` parameter — required for AIF360 Reweighing
- Interpretable via SHAP TreeExplainer (exact computation, not approximation)
- Good balance of accuracy and inference speed (frugal AI consideration)
- No GPU required — can run on standard office hardware

**Why NOT deep learning?**
- Dataset is small (~3,200 rows) — deep learning would overfit
- Tabular data doesn't benefit from neural architectures
- Tree ensembles are more explainable for HR stakeholders
- Lower carbon footprint (frugal AI)

## Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| n_estimators | 150 | Sufficient for convergence without overfitting |
| max_depth | 3 | Prevents overfitting on small dataset |
| learning_rate | 0.08 | Conservative for stable training |
| random_state | 42 | Reproducibility |

## Features

### Model Features (used for prediction)
1. **EngagementSurvey** — Employee engagement score (1-5)
2. **EmpSatisfaction** — Job satisfaction (1-5)
3. **SpecialProjectsCount** — Number of special projects/companies
4. **Absences** — Absence count
5. **DaysLateLast30** — Late days in last 30 days
6. **YearsAtCompany** — Tenure
7. **OverTime** — Whether employee works overtime (0/1)
8. **WorkLifeBalance** — Work-life balance score (1-4)
9. **Department** — Encoded department
10. **PerformanceScore** — Encoded performance rating
11. **RecruitmentSource** — Encoded recruitment channel
12. **MaritalStatus** — Encoded marital status
13. **AgeBracket** — Generalized age group
14. **SalaryBand** — Generalized salary range
15. **satisfaction_trend** — Derived satisfaction trajectory

### Protected Attributes (audit only, NOT used as features)
- **Sex** — Gender (Male/Female)
- **RaceDesc** — Ethnicity

## Fairness

### Methodology
- **Tool**: IBM AIF360 (AI Fairness 360)
- **Technique**: Reweighing — pre-processing algorithm that adjusts training instance weights
- **Protected attributes tested**: Gender (Sex), Race (RaceDesc)
- **Threshold**: Statistical Parity Difference (SPD) < ±0.10

### How Reweighing Works
1. Compute expected vs observed probability of favorable outcomes per group
2. Calculate weight = expected / observed for each training instance
3. Pass weights via `sample_weight` to the classifier during training
4. The model learns more equitable decision boundaries

### Metrics
| Metric | Definition | Ideal |
|---|---|---|
| Statistical Parity Difference (SPD) | Difference in positive outcome rates between groups | 0.0 |
| Disparate Impact (DI) | Ratio of positive outcome rates | 1.0 |
| Equal Opportunity Difference (EOD) | Difference in true positive rates | 0.0 |

## Explainability

### SHAP (SHapley Additive exPlanations)
- **Method**: TreeExplainer (exact SHAP values for tree ensembles)
- **Global importance**: Mean |SHAP| across test set
- **Local explanations**: Per-employee feature contribution
- **Visualization**: Bar chart on dashboard + per-prediction breakdown

### Interpretation for HR
- "Employee X is flagged as high risk because their engagement score is 1.8/5 (below average) and they have 18 absences (above threshold)"
- Each prediction comes with the top factors driving the risk score

## Limitations

1. **Synthetic data**: Model trained on synthetic datasets — real-world performance may differ
2. **Class imbalance**: Attrition is the minority class (~16-33%)
3. **Missing race data**: IBM/Kaggle datasets lack race information
4. **No temporal dynamics**: Model is static — doesn't capture trends over time
5. **Advisory only**: Predictions require human HR review before action
6. **Proxy bias**: Even without protected attributes as features, proxy features (e.g., department) could encode bias — auditing addresses this

## Intended Use

| ✅ Intended | ❌ Not Intended |
|---|---|
| Advisory tool for HR planning | Autonomous firing/hiring decisions |
| Identifying at-risk departments | Individual performance evaluation |
| Supporting retention strategies | Salary determination |
| Compliance demonstration | Legal evidence |

## EU AI Act Compliance

This model falls under **Annex III, Category 4 (High Risk)** — AI systems in employment decisions.

All requirements from Articles 9-15 are addressed:
- Risk management system (bias audit)
- Data governance (GDPR pipeline)
- Transparency (this Model Card + SHAP)
- Human oversight (advisory mode)
- Accuracy and robustness (validation + injection protection)
