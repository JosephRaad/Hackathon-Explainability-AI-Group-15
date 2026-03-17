# 📊 Data Card  TrustedAI HR Dataset

## Dataset Overview

| Property | Value |
|---|---|
| Name | TrustedAI HR Combined Dataset |
| Version | 1.0 |
| Created | March 2025 |
| Purpose | Employee attrition prediction and retention analysis |
| License | Open-source (Kaggle datasets) |

## Data Sources

### 1. Dr. Rich Huebner HR Dataset (~311 rows)
- **Origin**: Kaggle  designed by Drs. Rich Huebner and Carla Patalano
- **Nature**: Synthetic dataset for HR analytics education
- **Key fields**: Employee demographics, performance, engagement, termination status
- **Sensitive attributes**: Sex (M/F), RaceDesc, MaritalDesc

### 2. IBM HR Attrition Dataset (~1,470 rows)
- **Origin**: IBM sample dataset on Kaggle
- **Nature**: Synthetic employee data
- **Key fields**: Attrition, satisfaction scores, overtime, years at company
- **Note**: No race data available  marked as "Unknown"

### 3. Kaggle HR Dataset (~1,480 rows)
- **Origin**: Kaggle community dataset
- **Nature**: Similar schema to IBM with additional fields
- **Key fields**: Same as IBM with EmpID and AgeGroup

## Unified Schema

After merge, the dataset contains these fields:

| Field | Type | Description | GDPR Treatment |
|---|---|---|---|
| employee_id | String | Unique identifier | Pseudonymized (SHA-256) |
| Termd | Binary | Target: 1=left, 0=active | N/A |
| Sex | Category | Gender | Kept for bias audit |
| RaceDesc | Category | Ethnicity | Kept for bias audit |
| MaritalStatus | Category | Marital status | Encoded |
| Age | Integer | Employee age | Generalized to bracket |
| Department | Category | Work department | Encoded |
| Position | Category | Job title | Kept for analysis |
| Salary | Float | Annual salary | Generalized to band |
| PerformanceScore | Category | Performance rating | Encoded |
| EngagementSurvey | Float | 1-5 engagement score | Perturbed |
| EmpSatisfaction | Integer | 1-5 satisfaction | Encoded |
| Absences | Integer | Number of absences | Perturbed |
| DaysLateLast30 | Integer | Late days in last month | N/A |
| YearsAtCompany | Float | Tenure in years | N/A |
| OverTime | Binary | Works overtime | N/A |
| WorkLifeBalance | Integer | 1-4 balance score | N/A |
| departure_cause | Category | Reason for leaving | Generated |
| exit_feedback | Text | Exit interview text | Generated |
| satisfaction_trend | Category | declining/stable/improving | Derived |
| source_dataset | Category | Data provenance | Tracking |

## Data Enrichment

The following fields were generated to increase analytical value:

- **departure_cause**: Mapped from TermReason (Rich dataset) or randomly assigned based on realistic distributions for IBM/Kaggle
- **exit_feedback**: Synthetic exit interview text generated from templates based on departure cause and tenure
- **satisfaction_trend**: Derived from engagement and satisfaction scores
- **WorkLifeBalance** (Rich dataset): Simulated from overtime and absence patterns

## Sensitive Data Handling

### Protected Attributes
- **Sex** and **RaceDesc** are classified as sensitive data under GDPR Art. 9
- They are **RETAINED** in the anonymized dataset for fairness auditing
- **Legal basis**: EU AI Act Art. 10(5)  bias testing requires protected attributes
- **GDPR basis**: Art. 9(2)(g)  processing for substantial public interest (non-discrimination)
- These attributes are **NEVER used as model features**  only for audit

### Anonymization Pipeline (Step 2)
1. **Suppression**: TermReason free text removed
2. **Pseudonymization**: employee_id → salted SHA-256 hash
3. **Generalization**: Age → brackets, Salary → bands
4. **Perturbation**: Deterministic noise on continuous variables

## Known Limitations

- All three datasets are **synthetic**  they simulate real HR data but are not from actual companies
- IBM and Kaggle datasets lack race/ethnicity data (marked "Unknown")
- Exit interview texts are template-based, not from real interviews
- Class imbalance exists (typically ~16-33% attrition rate depending on source)

## Ethical Considerations

- No real individual can be identified from this data
- The synthetic nature means any biases are artifacts of the data creation process
- The model should always be used as an **advisory tool**  never for automated decisions
- All predictions require human HR review before any action is taken
