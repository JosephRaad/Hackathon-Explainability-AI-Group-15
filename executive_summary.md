# Executive Summary  TrustedAI HR Analytics

## The Problem

HR attrition prediction touches some of the most sensitive personal data in any organization. Existing ML systems often operate as black boxes   biased by historical patterns, non-compliant with GDPR, and vulnerable to AI manipulation attacks. The EU AI Act (Annex III, Category 4) now classifies employment AI as high-risk, demanding transparency, fairness, and human oversight.

## Our Solution

TrustedAI HR Analytics is a unified system that combines **predictive analytics**, **algorithmic fairness**, **GDPR compliance**, and **GenAI security** into a single auditable pipeline   built for an HR Manager persona ("Sarah") who needs actionable, trustworthy insights.

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
    → GDPR Anonymization (4 techniques) → Feature Engineering (15 features)
    → Bias Audit (AIF360 Reweighing) → Fair Model + SHAP Explainability
    → Streamlit Dashboard (5 pages) + Local NLP Chatbot
```

### What Makes This Different

1. **Fairness is not an afterthought.** We detected real bias (Race SPD = -0.238, exceeding the ±0.10 threshold) and mitigated it using IBM AIF360 Reweighing + group-specific threshold equalization   achieving SPD = 0.028.

2. **Privacy is structural, not cosmetic.** Four GDPR techniques (suppression, pseudonymization, generalization, perturbation) applied on the merged dataset before any modeling. Raw data never enters the repository.

3. **GenAI security is defensive by design.** A 5-layer pipeline (sanitize → inject-detect → truncate → role-lock → JSON-enforce) blocks prompt injection before it reaches the model. 17 injection patterns detected.

4. **The system is advisory, not autonomous.** Every prediction requires human HR review. No automated employment decisions   fully EU AI Act compliant.

5. **Runs fully locally.** No API key required. The chatbot and exit interview analyzer work entirely offline with local NLP fallback.

## Technical Stack

- **ML:** scikit-learn Gradient Boosting Classifier (150 estimators, sample_weight support)
- **Fairness:** IBM AIF360 (Reweighing pre-processing + threshold post-processing)
- **Explainability:** SHAP TreeExplainer (exact values for tree ensembles)
- **GenAI:** Claude API with 5-layer security + smart offline NLP fallback
- **Dashboard:** Streamlit with 5 pages (Flight Risk, Fairness Audit, AI Chatbot, Exit Interviews, Compliance)
- **Data:** Dr. Rich HRDataset_v14 + IBM HR Attrition + Kaggle HR Analytics

## Team

Built during the **Capgemini × ESILV TrustedAI Hackathon**   March 2025.
