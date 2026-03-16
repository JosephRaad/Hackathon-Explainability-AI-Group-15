"""
Turnover Prediction Model

Trains an XGBoost classifier to predict employee turnover (Termd = 0/1).
Generates SHAP explanations for model transparency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "hr_anonymized.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_turnover.json"
PLOTS_DIR = Path(__file__).parent.parent / "outputs" / "plots"

# Raw features from dataset
NUMERIC_FEATURES = [
    "Salary", "EngagementSurvey", "EmpSatisfaction",
    "SpecialProjectsCount", "DaysLateLast30", "Absences",
    "MarriedID", "GenderID", "FromDiversityJobFairID", "PerfScoreID",
]

CATEGORICAL_FEATURES = [
    "Department", "Position", "RecruitmentSource", "PerformanceScore",
]

TARGET = "Termd"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that capture turnover signals better."""
    df = df.copy()

    # Tenure: days between hire date and termination (or today)
    df["DateofHire"] = pd.to_datetime(df["DateofHire"], format="mixed")
    if "DateofTermination" in df.columns:
        df["DateofTermination"] = pd.to_datetime(df["DateofTermination"], format="mixed", errors="coerce")
        end_date = df["DateofTermination"].fillna(pd.Timestamp("2019-06-01"))
    else:
        end_date = pd.Timestamp("2019-06-01")
    df["Tenure_Days"] = (end_date - df["DateofHire"]).dt.days

    # Salary relative to department average
    dept_avg_salary = df.groupby("Department")["Salary"].transform("mean")
    df["Salary_vs_Dept"] = df["Salary"] / dept_avg_salary

    # Salary relative to position average
    pos_avg_salary = df.groupby("Position")["Salary"].transform("mean")
    df["Salary_vs_Position"] = df["Salary"] / pos_avg_salary

    # Manager turnover rate (how many people under this manager have left)
    mgr_turnover = df.groupby("ManagerID")["Termd"].transform("mean")
    df["Manager_Turnover_Rate"] = mgr_turnover

    # Engagement-satisfaction gap
    df["Engagement_Satisfaction_Gap"] = df["EngagementSurvey"] - df["EmpSatisfaction"]

    # Low performer flag
    df["Is_Low_Performer"] = (df["PerfScoreID"] <= 2).astype(int)

    # High absence flag
    df["High_Absences"] = (df["Absences"] > df["Absences"].median()).astype(int)

    return df


ENGINEERED_FEATURES = [
    "Tenure_Days", "Salary_vs_Dept", "Salary_vs_Position",
    "Manager_Turnover_Rate", "Engagement_Satisfaction_Gap",
    "Is_Low_Performer", "High_Absences",
]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = engineer_features(df)
    all_features = NUMERIC_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES
    X = df[all_features].copy()
    y = df[TARGET]

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True)

    return X, y


def train_model(X_train, y_train) -> xgb.XGBClassifier:
    # Handle class imbalance: 207 active vs 104 terminated
    scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def generate_shap_plots(model, X_test, feature_names):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Summary plot — global feature importance
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot — mean absolute SHAP values
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Force plot for first terminated employee
    terminated_indices = X_test.index[X_test.index.isin(
        X_test.index[:len(X_test)]
    )]
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"SHAP plots saved to {PLOTS_DIR}")


def main():
    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {X.shape[1]} (including {len(ENGINEERED_FEATURES)} engineered)")
    print(f"Class distribution (train): {dict(y_train.value_counts())}")

    # Cross-validation score first
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(train_model(X_train, y_train), X, y, cv=cv, scoring="f1")
    print(f"\n5-Fold CV F1 Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    model = train_model(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Active", "Terminated"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"\nModel saved to {MODEL_PATH}")

    # SHAP explanations
    generate_shap_plots(model, X_test, X.columns.tolist())

    return model, X_test, y_test


if __name__ == "__main__":
    main()
