# Executive Summary — TrustedAI HR Analytics

## The Problem

HR attrition prediction touches some of the most sensitive personal data in any organization. Existing ML systems often operate as black boxes — biased by historical patterns, non-compliant with GDPR, and vulnerable to AI manipulation attacks. The EU AI Act (Annex III, Category 4) now classifies employment AI as high-risk, demanding transparency, fairness, and human oversight.

## Our Solution

TrustedAI HR Analytics is a unified system that combines **predictive analytics**, **algorithmic fairness**, **GDPR compliance**, and **GenAI security** into a single auditable pipeline — built for an HR Manager persona ("Sarah") who needs actionable, trustworthy insights.

### Key Results

| Metric | Before | After | Method |
|--------|--------|-------|--------|
| **Bias (Race SPD)** | -0.238 (BIASED) | 0.028 (FAIR) | AIF360 Reweighing + Threshold Equalization |
| **Bias (Gender SPD)** | -0.018 (FAIR) | -0.021 (FAIR) | Already within threshold |
| **Accuracy** | 92.0% | 89.6% | Gradient Boosting Classifier (15 features) |
| **GDPR Compliance** | Raw PII (24 cols) | 0 PII (23 cols) | 4-technique pipeline |
| **GenAI Security** | Unprotected | 5/5 injections blocked | 5-layer defense |
| **Datasets** | 1 (311 rows) | 3 combined (3,261 rows) | Schema-normalized merge |

### Architecture

```
Raw HR Data → Multi-Dataset Merge (3 sources, 3,261 rows)
           → Per-source stats export (Dr. Rich / IBM / Kaggle independently auditable)
           → GDPR Anonymization (4 techniques)
           → Feature Engineering (15 features)
           → Bias Audit (AIF360 Reweighing)
           → Fair Model + SHAP Explainability
           → Streamlit Dashboard (5 pages)
           → Hybrid Chatbot (14-intent local engine + Claude API fallback)
```

### What Makes This Different

1. **Fairness is not an afterthought.** We detected real bias (Race SPD = -0.238,
   exceeding the ±0.10 threshold) and mitigated it using IBM AIF360 Reweighing +
   group-specific threshold equalization — achieving SPD = 0.028.

2. **Privacy is structural, not cosmetic.** Four GDPR techniques (suppression,
   pseudonymization, generalization, perturbation) applied on the merged dataset
   before any modeling. Raw data never enters the repository.

3. **GenAI security is defensive by design.** A 5-layer pipeline
   (sanitize → inject-detect → truncate → role-lock → JSON-enforce) blocks prompt
   injection before it reaches the model. 17 injection patterns detected and blocked.

4. **The system is advisory, not autonomous.** Every prediction requires human HR
   review. No automated employment decisions — fully EU AI Act compliant.

5. **Runs fully locally.** No API key required. The chatbot answers 14 categories
   of HR analytics questions using live data with no external dependency. When a
   Claude API key is provided, complex freeform questions are escalated automatically,
   with graceful local fallback if the API is unavailable.

6. **Data provenance is transparent.** We went beyond the single-dataset requirement
   by merging 3 sources and maintaining per-source auditability. Each dataset
   (Dr. Rich, IBM, Kaggle) can be queried independently through the chatbot
   ("Tell me about each dataset") and is backed by a live stats JSON snapshot.

## Technical Stack

- **ML:** scikit-learn Gradient Boosting Classifier (150 estimators, sample_weight support)
- **Fairness:** IBM AIF360 (Reweighing pre-processing + threshold post-processing)
- **Explainability:** SHAP TreeExplainer (exact values for tree ensembles)
- **GenAI:** Claude API (claude-haiku-4-5) with 5-layer security + smart offline NLP fallback
- **Dashboard:** Streamlit with 5 pages (Flight Risk, Fairness Audit, AI Chatbot,
  Exit Interviews, Compliance)
- **Chatbot:** 14-intent hybrid engine — synonym-aware local matcher with
  Claude API escalation for complex questions
- **Data:** Dr. Rich HRDataset_v14 + IBM HR Attrition + Kaggle HR Analytics

## Team

Built during the **Capgemini × ESILV TrustedAI Hackathon** — March 2025.
