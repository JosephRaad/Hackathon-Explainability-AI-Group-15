# Executive Summary — TrustedAI

## The Problem
A company faces a high resignation rate and needs to understand why employees are leaving and how to retain talent — using AI that is secure, fair, and transparent.

## Our Solution
TrustedAI is an end-to-end HR analytics platform built around two core themes: **AI & Cybersecurity** and **Ethical AI**.

### What It Does
- **Predicts** attrition risk for 3,261 employees with 92% accuracy
- **Explains** each prediction using SHAP feature importance
- **Audits** for gender and racial bias using IBM AIF360 Reweighing
- **Protects** data with 4 GDPR anonymization techniques
- **Defends** against prompt injection with 17-pattern detection
- **Empowers HR** with a data-driven chatbot that answers questions like "How many high-risk employees per department?"

### Key Results
- **385 employees** (11.8%) flagged as High Risk across 8 departments
- **Production** and **Sales** departments show highest attrition risk
- Top departure drivers: compensation (27%), career growth (22%), management (18%)
- Fairness audit passes all thresholds after Reweighing mitigation
- EU AI Act Annex III compliant (High Risk category, all articles addressed)

### Technical Stack
- Gradient Boosting Classifier with AIF360 Reweighing
- SHAP explainability + local NLP analysis
- 5-layer GenAI security pipeline
- Streamlit dashboard with data-driven chatbot
- Full GDPR pipeline: suppression, pseudonymization, generalization, perturbation

### Impact
The platform gives HR directors actionable, explainable, and bias-audited insights to make retention decisions — while ensuring every prediction can be justified, every data point is protected, and every recommendation is grounded in evidence.
