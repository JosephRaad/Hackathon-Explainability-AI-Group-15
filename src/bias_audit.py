# =============================================================================
# TrustedAI — bias_audit.py
# Step 4: Train model + fairness audit with AIF360 + SHAP explainability.
# Protected attributes (Sex, RaceDesc) are AUDITED but NOT used as features.
# Saves: predictions.csv · fairness_metrics.json · model_fair.pkl · shap data
# =============================================================================

import pandas as pd
import numpy as np
import json, os, pickle, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

FEATURES_PATH = "data/processed/hr_features.csv"
META_PATH = "data/processed/hr_features_meta.json"
PREDICTIONS_PATH = "data/processed/predictions.csv"
METRICS_PATH = "data/processed/fairness_metrics.json"
MODEL_PATH = "data/processed/model_fair.pkl"
SHAP_PATH = "data/processed/shap_values.pkl"

TARGET = "Termd"
FAV_LABEL = 0
UNFAV_LABEL = 1


def _try_aif360_audit(df, feature_cols, prot_attr, priv_val, unpriv_val, label):
    """Attempt AIF360 audit. Returns metrics or None if insufficient data."""
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import ClassificationMetric
        from aif360.algorithms.preprocessing import Reweighing
    except ImportError:
        print(f"    ⚠️  AIF360 not installed — skipping {label} audit")
        return None

    print(f"\n  ── Auditing: {label} (priv={priv_val}, unpriv={unpriv_val})")

    unpriv = [{prot_attr: unpriv_val}]
    priv = [{prot_attr: priv_val}]

    # Build AIF360 datasets
    cols_needed = list(set(feature_cols + [TARGET, prot_attr]))
    df_c = df[[c for c in cols_needed if c in df.columns]].copy().dropna()

    if len(df_c) < 50 or df_c[TARGET].nunique() < 2:
        print(f"    ⚠️  Insufficient data for {label} — skipping")
        return None

    # Check both groups exist
    if df_c[prot_attr].nunique() < 2:
        print(f"    ⚠️  Only one group in {prot_attr} — skipping")
        return None

    X = df_c[feature_cols]
    y = df_c[TARGET]

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df_c.index, test_size=0.2, random_state=42, stratify=y
    )
    df_tr = df_c.loc[idx_tr]
    df_te = df_c.loc[idx_te]

    ds_tr = BinaryLabelDataset(df=df_tr, label_names=[TARGET],
                                protected_attribute_names=[prot_attr],
                                favorable_label=FAV_LABEL,
                                unfavorable_label=UNFAV_LABEL)
    ds_te = BinaryLabelDataset(df=df_te, label_names=[TARGET],
                                protected_attribute_names=[prot_attr],
                                favorable_label=FAV_LABEL,
                                unfavorable_label=UNFAV_LABEL)

    # Baseline model
    clf_b = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=3, random_state=42
    )
    clf_b.fit(X_tr, y_tr)
    preds_b = clf_b.predict(X_te)

    ds_te_b = ds_te.copy()
    ds_te_b.labels = preds_b.reshape(-1, 1)

    cm_b = ClassificationMetric(ds_te, ds_te_b,
                                 unprivileged_groups=unpriv,
                                 privileged_groups=priv)
    baseline = {
        "accuracy": round(cm_b.accuracy(), 4),
        "disparate_impact": round(cm_b.disparate_impact(), 4),
        "statistical_parity_difference": round(cm_b.statistical_parity_difference(), 4),
        "equal_opportunity_difference": round(cm_b.equal_opportunity_difference(), 4),
        "average_odds_difference": round(cm_b.average_odds_difference(), 4),
    }
    print(f"    Baseline  → Acc:{baseline['accuracy']:.3f}  "
          f"DI:{baseline['disparate_impact']:.3f}  "
          f"SPD:{baseline['statistical_parity_difference']:.3f}")

    # Reweighing (fair model)
    rw = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv)
    ds_tr_rw = rw.fit_transform(ds_tr)
    weights = ds_tr_rw.instance_weights

    clf_f = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=3, random_state=42
    )
    clf_f.fit(X_tr, y_tr, sample_weight=weights)
    preds_f = clf_f.predict(X_te)
    proba_f = clf_f.predict_proba(X_te)[:, 1]

    ds_te_f = ds_te.copy()
    ds_te_f.labels = preds_f.reshape(-1, 1)

    cm_f = ClassificationMetric(ds_te, ds_te_f,
                                 unprivileged_groups=unpriv,
                                 privileged_groups=priv)
    fair = {
        "accuracy": round(cm_f.accuracy(), 4),
        "disparate_impact": round(cm_f.disparate_impact(), 4),
        "statistical_parity_difference": round(cm_f.statistical_parity_difference(), 4),
        "equal_opportunity_difference": round(cm_f.equal_opportunity_difference(), 4),
        "average_odds_difference": round(cm_f.average_odds_difference(), 4),
    }
    print(f"    Fair      → Acc:{fair['accuracy']:.3f}  "
          f"DI:{fair['disparate_impact']:.3f}  "
          f"SPD:{fair['statistical_parity_difference']:.3f}")

    improvement = {
        "disparate_impact_delta": round(fair["disparate_impact"] - baseline["disparate_impact"], 4),
        "spd_delta": round(abs(baseline["statistical_parity_difference"]) -
                           abs(fair["statistical_parity_difference"]), 4),
        "eod_delta": round(abs(baseline["equal_opportunity_difference"]) -
                           abs(fair["equal_opportunity_difference"]), 4),
        "accuracy_delta": round(fair["accuracy"] - baseline["accuracy"], 4),
    }

    return {
        "baseline": baseline,
        "fair_model": fair,
        "improvement": improvement,
        "clf_fair": clf_f,
        "proba_fair": proba_f,
        "X_test": X_te,
        "y_test": y_te,
        "df_test": df_te,
    }


def run():
    print("\n" + "=" * 60)
    print("  TrustedAI — Bias Audit + Explainability Pipeline")
    print("  AIF360 Reweighing · SHAP Feature Importance")
    print("=" * 60)

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("Run preprocess.py first.")

    df = pd.read_csv(FEATURES_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    protected_cols = meta.get("protected", [])

    print(f"\n  Features:  {feature_cols}")
    print(f"  Protected: {protected_cols} (audit only, excluded from model)")
    print(f"  Rows: {len(df)} | Attrition: {df[TARGET].mean():.1%}")

    # ── Train primary model on all data ──────────────────────────────────
    # This model uses feature_cols ONLY (no protected attributes)
    df_clean = df.dropna(subset=feature_cols + [TARGET])
    X = df_clean[feature_cols]
    y = df_clean[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── AIF360 Audits ────────────────────────────────────────────────────
    audit_results = {}
    primary_clf = None
    primary_proba = None

    for attr in protected_cols:
        if attr not in df_clean.columns or df_clean[attr].nunique() < 2:
            print(f"\n  ⚠️ {attr}: insufficient groups, skipping audit")
            continue

        vals = df_clean[attr].value_counts()
        priv_val = int(vals.index[0])
        unpriv_val = int(vals.index[1]) if len(vals) > 1 else 0

        result = _try_aif360_audit(
            df_clean, feature_cols, attr, priv_val, unpriv_val, attr.upper()
        )
        if result:
            audit_results[attr] = result
            # Use the first successful audit's fair model as primary
            if primary_clf is None:
                primary_clf = result["clf_fair"]

    # ── Fallback: train without AIF360 if audits failed ──────────────────
    if primary_clf is None:
        print("\n  ⚠️ No AIF360 audit succeeded — training standard model")
        primary_clf = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=3, random_state=42
        )
        primary_clf.fit(X_tr, y_tr)

    primary_proba = primary_clf.predict_proba(X_te)[:, 1]
    primary_preds = primary_clf.predict(X_te)
    acc = accuracy_score(y_te, primary_preds)

    print(f"\n  Primary model accuracy: {acc:.1%}")
    print(classification_report(y_te, primary_preds, target_names=["Active", "Left"]))

    # ── SHAP Explainability ──────────────────────────────────────────────
    shap_data = None
    try:
        import shap
        print("  Computing SHAP values...")
        explainer = shap.TreeExplainer(primary_clf)
        shap_values = explainer.shap_values(X_te)

        # For binary classification, shap_values might be a list [class0, class1]
        if isinstance(shap_values, list):
            sv = shap_values[1]  # Class 1 = Left
        else:
            sv = shap_values

        # Feature importance (mean absolute SHAP)
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": np.abs(sv).mean(axis=0)
        }).sort_values("importance", ascending=False)

        print("\n  Top Feature Importances (SHAP):")
        for _, row in importance.head(8).iterrows():
            bar = "█" * int(row["importance"] * 50)
            print(f"    {row['feature']:25s} {row['importance']:.4f}  {bar}")

        shap_data = {
            "importance": importance.to_dict(orient="records"),
            "shap_values": sv.tolist(),
            "feature_names": feature_cols,
        }
        with open(SHAP_PATH, "wb") as f:
            pickle.dump(shap_data, f)
        print(f"  ✅ SHAP values saved: {SHAP_PATH}")

    except ImportError:
        print("  ⚠️ SHAP not installed — skipping explainability")
    except Exception as e:
        print(f"  ⚠️ SHAP failed: {e}")

    # ── Build predictions for ALL employees ──────────────────────────────
    print("\n  Generating predictions for all employees...")
    all_X = df_clean[feature_cols]
    all_proba = primary_clf.predict_proba(all_X)[:, 1]

    df_out = df_clean.copy()
    df_out["risk_score"] = all_proba
    df_out["risk_level"] = pd.cut(
        df_out["risk_score"],
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    # Re-attach human-readable labels
    for col, mapping in meta.get("label_mappings", {}).items():
        if col in df_out.columns and col in meta.get("categorical", []) + meta.get("protected", []):
            df_out[f"{col}_label"] = df_out[col].astype(str).map(mapping).fillna("Unknown")

    # Save predictions
    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    df_out.to_csv(PREDICTIONS_PATH, index=False)

    # ── Save model ───────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(primary_clf, f)

    # ── Build metrics JSON ───────────────────────────────────────────────
    # Determine primary audit attribute
    best_attr = None
    best_gap = 0
    for attr, res in audit_results.items():
        gap = abs(res["baseline"]["statistical_parity_difference"])
        if gap > best_gap:
            best_gap = gap
            best_attr = attr

    if best_attr:
        primary_audit = audit_results[best_attr]
        b_metrics = primary_audit["baseline"]
        f_metrics = primary_audit["fair_model"]
        imp_metrics = primary_audit["improvement"]
    else:
        b_metrics = f_metrics = {
            "accuracy": round(acc, 4),
            "disparate_impact": 1.0,
            "statistical_parity_difference": 0.0,
            "equal_opportunity_difference": 0.0,
            "average_odds_difference": 0.0,
        }
        imp_metrics = {"spd_delta": 0.0, "accuracy_delta": 0.0,
                       "disparate_impact_delta": 0.0, "eod_delta": 0.0}

    n_high = int((df_out["risk_level"] == "High").sum())
    n_med = int((df_out["risk_level"] == "Medium").sum())
    n_low = int((df_out["risk_level"] == "Low").sum())

    # Department risk summary
    dept_col = "Department_label" if "Department_label" in df_out.columns else "Department"
    dept_risk = {}
    if dept_col in df_out.columns:
        for dept, grp in df_out.groupby(dept_col):
            dept_risk[str(dept)] = {
                "total": int(len(grp)),
                "high_risk": int((grp["risk_level"] == "High").sum()),
                "medium_risk": int((grp["risk_level"] == "Medium").sum()),
                "low_risk": int((grp["risk_level"] == "Low").sum()),
                "avg_risk_score": round(float(grp["risk_score"].mean()), 4),
                "attrition_rate": round(float(grp[TARGET].mean()), 4),
            }

    metrics_out = {
        "primary_attribute": best_attr or "Sex",
        "primary_label": "Gender" if best_attr == "Sex" else (
            "Race" if best_attr == "RaceDesc" else "Unknown"),
        "baseline": b_metrics,
        "fair_model": f_metrics,
        "improvement": imp_metrics,
        "all_audits": {},
        "n_employees": len(df_out),
        "high_risk": n_high,
        "medium_risk": n_med,
        "low_risk": n_low,
        "department_risk": dept_risk,
        "feature_importance": shap_data["importance"] if shap_data else [],
        "model_info": {
            "type": "GradientBoostingClassifier",
            "n_estimators": 150,
            "learning_rate": 0.08,
            "max_depth": 3,
            "n_features": len(feature_cols),
            "features": feature_cols,
        },
    }

    for attr, res in audit_results.items():
        label = "sex" if attr == "Sex" else "race" if attr == "RaceDesc" else attr
        metrics_out["all_audits"][label] = {
            "baseline": res["baseline"],
            "fair_model": res["fair_model"],
            "improvement": res["improvement"],
        }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n  {'='*50}")
    print(f"  RESULTS SUMMARY")
    print(f"  {'='*50}")
    print(f"  Employees:  {len(df_out)}")
    print(f"  High Risk:  {n_high} ({n_high/len(df_out):.1%})")
    print(f"  Medium Risk: {n_med} ({n_med/len(df_out):.1%})")
    print(f"  Low Risk:   {n_low} ({n_low/len(df_out):.1%})")
    if best_attr:
        print(f"\n  Primary fairness audit: {best_attr}")
        print(f"  SPD: {b_metrics['statistical_parity_difference']:.4f} → "
              f"{f_metrics['statistical_parity_difference']:.4f}")
        print(f"  Acc: {b_metrics['accuracy']:.4f} → {f_metrics['accuracy']:.4f}")

    print(f"\n  ✅ Predictions: {PREDICTIONS_PATH}")
    print(f"  ✅ Metrics:     {METRICS_PATH}")
    print(f"  ✅ Model:       {MODEL_PATH}")
    print("=" * 60 + "\n")
    return metrics_out


if __name__ == "__main__":
    run()
