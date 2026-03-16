"""
Fairness Audit Module — AI Ethics

Evaluates the XGBoost turnover prediction model for bias across
protected attributes (Sex, RaceDesc).

Metrics computed:
- Demographic Parity Difference: P(pred=1|unprivileged) - P(pred=1|privileged)
  Ideal = 0, negative = bias against unprivileged group
- Equalized Odds Difference: difference in TPR between groups
  Ideal = 0
- Disparate Impact Ratio: P(pred=1|unprivileged) / P(pred=1|privileged)
  Ideal = 1, legal threshold typically 0.8 (80% rule)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import prepare_features, TARGET

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "hr_anonymized.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_turnover.json"
PLOTS_DIR = Path(__file__).parent.parent / "outputs" / "plots"


def compute_fairness_metrics(y_true, y_pred, sensitive_attr, privileged_value):
    """Compute fairness metrics for a binary sensitive attribute split."""
    privileged_mask = sensitive_attr == privileged_value
    unprivileged_mask = ~privileged_mask

    # Prediction rates
    priv_pred_rate = y_pred[privileged_mask].mean()
    unpriv_pred_rate = y_pred[unprivileged_mask].mean()

    # Demographic Parity Difference
    dp_diff = unpriv_pred_rate - priv_pred_rate

    # Disparate Impact Ratio
    di_ratio = unpriv_pred_rate / priv_pred_rate if priv_pred_rate > 0 else float("inf")

    # True Positive Rates (Equalized Odds)
    priv_positives = (y_true[privileged_mask] == 1)
    unpriv_positives = (y_true[unprivileged_mask] == 1)

    priv_tpr = y_pred[privileged_mask][priv_positives].mean() if priv_positives.sum() > 0 else 0
    unpriv_tpr = y_pred[unprivileged_mask][unpriv_positives].mean() if unpriv_positives.sum() > 0 else 0
    eo_diff = unpriv_tpr - priv_tpr

    return {
        "privileged_pred_rate": priv_pred_rate,
        "unprivileged_pred_rate": unpriv_pred_rate,
        "demographic_parity_diff": dp_diff,
        "disparate_impact_ratio": di_ratio,
        "equalized_odds_diff": eo_diff,
        "privileged_count": privileged_mask.sum(),
        "unprivileged_count": unprivileged_mask.sum(),
    }


def print_fairness_report(name, metrics):
    print(f"\n{'='*50}")
    print(f"Fairness Audit: {name}")
    print(f"{'='*50}")
    print(f"  Privileged group size:   {metrics['privileged_count']}")
    print(f"  Unprivileged group size: {metrics['unprivileged_count']}")
    print(f"  Privileged pred rate:    {metrics['privileged_pred_rate']:.3f}")
    print(f"  Unprivileged pred rate:  {metrics['unprivileged_pred_rate']:.3f}")
    print(f"  Demographic Parity Diff: {metrics['demographic_parity_diff']:+.3f}  (ideal: 0)")
    print(f"  Disparate Impact Ratio:  {metrics['disparate_impact_ratio']:.3f}  (ideal: 1, threshold: 0.8)")
    print(f"  Equalized Odds Diff:     {metrics['equalized_odds_diff']:+.3f}  (ideal: 0)")

    # Flag issues
    if abs(metrics["demographic_parity_diff"]) > 0.1:
        print(f"  ⚠ WARNING: Demographic parity difference exceeds ±0.1 threshold")
    if metrics["disparate_impact_ratio"] < 0.8 or metrics["disparate_impact_ratio"] > 1.25:
        print(f"  ⚠ WARNING: Disparate impact ratio outside 0.8-1.25 range (80% rule)")


def plot_fairness_comparison(results: dict):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, results.items()):
        groups = ["Privileged", "Unprivileged"]
        rates = [metrics["privileged_pred_rate"], metrics["unprivileged_pred_rate"]]
        colors = ["#2196F3", "#FF5722"]
        bars = ax.bar(groups, rates, color=colors, width=0.5)
        ax.set_ylabel("Predicted Termination Rate")
        ax.set_title(f"Fairness: {name}")
        ax.set_ylim(0, 1)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{rate:.2f}", ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fairness_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFairness plot saved to {PLOTS_DIR / 'fairness_comparison.png'}")


def main():
    df = pd.read_csv(DATA_PATH)
    X, y = prepare_features(df)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    test_indices = X_test.index

    # Load model and predict
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    y_pred = model.predict(X_test)

    # Audit: Sex (M = privileged, F = unprivileged)
    sex_test = df.loc[test_indices, "Sex"]
    sex_metrics = compute_fairness_metrics(
        y_test.values, y_pred, sex_test.values, privileged_value="M"
    )
    print_fairness_report("Sex (M=privileged, F=unprivileged)", sex_metrics)

    # Audit: RaceDesc (White = privileged, Non-White = unprivileged)
    race_test = df.loc[test_indices, "RaceDesc"]
    race_binary = (race_test == "White").map({True: "White", False: "Non-White"})
    race_metrics = compute_fairness_metrics(
        y_test.values, y_pred, race_binary.values, privileged_value="White"
    )
    print_fairness_report("Race (White=privileged, Non-White=unprivileged)", race_metrics)

    plot_fairness_comparison({
        "Sex": sex_metrics,
        "Race": race_metrics,
    })

    return {"sex": sex_metrics, "race": race_metrics}


if __name__ == "__main__":
    main()
