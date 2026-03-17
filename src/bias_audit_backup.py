# =============================================================================
# TrustedAI  bias_audit.py  (v4  multi-group + threshold equalization)
# Step 4: Train model + fairness audit with AIF360 + SHAP explainability.
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
        print(f"    ⚠️  AIF360 not installed  skipping {label} audit")
        return None

    print(f"\n  ── Auditing: {label} (priv={priv_val}, unpriv={unpriv_val})")

    unpriv = [{prot_attr: unpriv_val}]
    priv = [{prot_attr: priv_val}]

    # Build AIF360 datasets
    cols_needed = list(set(feature_cols + [TARGET, prot_attr]))
    df_c = df[[c for c in cols_needed if c in df.columns]].copy().dropna()

    if len(df_c) < 50 or df_c[TARGET].nunique() < 2:
        print(f"    ⚠️  Insufficient data for {label}  skipping")
        return None

    # Check both groups exist
    if df_c[prot_attr].nunique() < 2:
        print(f"    ⚠️  Only one group in {prot_attr}  skipping")
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

    # ── Baseline model (no fairness correction) ─────────────────────────
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

    # ── Stage 1: Reweighing (pre-processing) ────────────────────────────
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
    print(f"    Reweigh   → Acc:{fair['accuracy']:.3f}  "
          f"DI:{fair['disparate_impact']:.3f}  "
          f"SPD:{fair['statistical_parity_difference']:.3f}")

    # ── Stage 2: Group-specific threshold adjustment (post-processing) ──
    if abs(fair["statistical_parity_difference"]) >= 0.10:
        print(f"    SPD still {fair['statistical_parity_difference']:.3f} "
              f" applying group-threshold equalization...")
        try:
            n_te = len(X_te)
            cal_size = n_te // 2
            idx_all = np.arange(n_te)
            np.random.seed(42)
            np.random.shuffle(idx_all)
            cal_idx = idx_all[:cal_size]
            eval_idx = idx_all[cal_size:]

            proba_cal = proba_f[cal_idx]
            proba_eval = proba_f[eval_idx]
            y_cal = y_te.values[cal_idx]
            y_eval = y_te.values[eval_idx]
            prot_cal = df_te[prot_attr].values[cal_idx]
            prot_eval = df_te[prot_attr].values[eval_idx]

            overall_rate = (proba_cal >= 0.5).mean()
            thresholds = {}

            for group_val in [priv_val, unpriv_val]:
                g_mask = prot_cal == group_val
                g_proba = proba_cal[g_mask]
                if len(g_proba) == 0:
                    thresholds[group_val] = 0.5
                    continue
                best_t = 0.5
                best_diff = abs((g_proba >= 0.5).mean() - overall_rate)
                for t in np.linspace(0.05, 0.95, 200):
                    rate = (g_proba >= t).mean()
                    diff = abs(rate - overall_rate)
                    if diff < best_diff:
                        best_diff = diff
                        best_t = t
                thresholds[group_val] = best_t

            print(f"    Calibrated thresholds: priv={thresholds[priv_val]:.3f}, "
                  f"unpriv={thresholds[unpriv_val]:.3f}")

            preds_eq = np.zeros(len(eval_idx), dtype=int)
            for group_val in [priv_val, unpriv_val]:
                g_mask = prot_eval == group_val
                preds_eq[g_mask] = (proba_eval[g_mask] >= thresholds[group_val]).astype(int)
            other_mask = ~np.isin(prot_eval, [priv_val, unpriv_val])
            if other_mask.any():
                preds_eq[other_mask] = (proba_eval[other_mask] >= 0.5).astype(int)

            priv_rate = preds_eq[prot_eval == priv_val].mean()
            unpriv_rate = preds_eq[prot_eval == unpriv_val].mean()
            spd_eq = round(unpriv_rate - priv_rate, 4)
            di_eq = round(unpriv_rate / priv_rate, 4) if priv_rate > 0 else 1.0
            acc_eq = round(accuracy_score(y_eval, preds_eq), 4)

            priv_tp = preds_eq[(prot_eval == priv_val) & (y_eval == 1)].mean() if ((prot_eval == priv_val) & (y_eval == 1)).any() else 0
            unpriv_tp = preds_eq[(prot_eval == unpriv_val) & (y_eval == 1)].mean() if ((prot_eval == unpriv_val) & (y_eval == 1)).any() else 0
            eod_eq = round(unpriv_tp - priv_tp, 4)

            fair = {
                "accuracy": acc_eq,
                "disparate_impact": di_eq,
                "statistical_parity_difference": spd_eq,
                "equal_opportunity_difference": eod_eq,
                "average_odds_difference": round(eod_eq / 2, 4),
            }
            print(f"    PostProc  → Acc:{acc_eq:.3f}  "
                  f"DI:{di_eq:.3f}  "
                  f"SPD:{spd_eq:.3f}  (on held-out eval set)")
        except Exception as e:
            print(f"    ⚠️  Post-processing failed: {e}")
            import traceback; traceback.print_exc()

    # ── Compute improvement ──────────────────────────────────────────────
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
    print("  TrustedAI  Bias Audit + Explainability Pipeline")
    print("  AIF360 Reweighing + Threshold Equalization · SHAP")
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
    df_clean = df.dropna(subset=feature_cols + [TARGET])
    X = df_clean[feature_cols]
    y = df_clean[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── AIF360 Audits ────────────────────────────────────────────────────
    audit_results = {}
    primary_clf = None

    for attr in protected_cols:
        if attr not in df_clean.columns or df_clean[attr].nunique() < 2:
            print(f"\n  ⚠️ {attr}: insufficient groups, skipping audit")
            continue

        # ── For multi-group attributes: binarize to majority vs rest ─────
        original_col = None
        if df_clean[attr].nunique() > 2:
            majority_val = int(df_clean[attr].value_counts().index[0])
            n_groups = df_clean[attr].nunique()
            print(f"\n  ℹ️  {attr} has {n_groups} groups  binarizing: "
                  f"majority ({majority_val}) vs rest")
            original_col = df_clean[attr].copy()
            df_clean[attr] = (df_clean[attr] == majority_val).astype(int)
            priv_val = 1   # majority group
            unpriv_val = 0  # everyone else
        else:
            vals = df_clean[attr].value_counts()
            priv_val = int(vals.index[0])
            unpriv_val = int(vals.index[1]) if len(vals) > 1 else 0

        result = _try_aif360_audit(
            df_clean, feature_cols, attr, priv_val, unpriv_val, attr.upper()
        )
        if result:
            audit_results[attr] = result
            if primary_clf is None:
                primary_clf = result["clf_fair"]

        # Restore original multi-group column after audit
        if original_col is not None:
            df_clean[attr] = original_col

    # ── Fallback: manual Reweighing without AIF360 ──────────────────────
    if primary_clf is None:
        print("\n  ⚠️ No AIF360 audit succeeded  applying manual Reweighing")

        prot_attr = protected_cols[0] if protected_cols else None
        weights = np.ones(len(X_tr))

        if prot_attr and prot_attr in df_clean.columns:
            df_tr_local = df_clean.loc[X_tr.index]
            n = len(df_tr_local)

            for g in df_tr_local[prot_attr].unique():
                for label in [0, 1]:
                    mask = (df_tr_local[prot_attr] == g) & (df_tr_local[TARGET] == label)
                    n_gl = mask.sum()
                    if n_gl == 0:
                        continue
                    p_g = (df_tr_local[prot_attr] == g).sum() / n
                    p_l = (df_tr_local[TARGET] == label).sum() / n
                    expected = p_g * p_l
                    observed = n_gl / n
                    w = expected / observed if observed > 0 else 1.0
                    idx_mask = mask[mask].index
                    weights_idx = [list(X_tr.index).index(i) for i in idx_mask if i in X_tr.index]
                    weights[weights_idx] = w

            print(f"    Manual Reweighing on '{prot_attr}': "
                  f"weights [{weights.min():.3f}, {weights.max():.3f}]")

        clf_baseline = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=3, random_state=42)
        clf_baseline.fit(X_tr, y_tr)
        preds_base = clf_baseline.predict(X_te)

        primary_clf = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=3, random_state=42)
        primary_clf.fit(X_tr, y_tr, sample_weight=weights)
        preds_fair = primary_clf.predict(X_te)

        if prot_attr and prot_attr in df_clean.columns:
            df_te_local = df_clean.loc[X_te.index]
            vals = df_te_local[prot_attr].value_counts()
            pv = vals.index[0]
            uv = vals.index[1] if len(vals) > 1 else vals.index[0]

            def _spd(preds, df_t, attr, priv, unpriv):
                pr = preds[df_t[attr].values == priv].mean()
                ur = preds[df_t[attr].values == unpriv].mean()
                return round(float(ur - pr), 4)

            def _di(preds, df_t, attr, priv, unpriv):
                pr = (preds[df_t[attr].values == priv] == 1).mean()
                ur = (preds[df_t[attr].values == unpriv] == 1).mean()
                return round(float(ur / pr), 4) if pr > 0 else 1.0

            spd_b = _spd(preds_base, df_te_local, prot_attr, pv, uv)
            spd_f = _spd(preds_fair, df_te_local, prot_attr, pv, uv)
            acc_b = round(accuracy_score(y_te, preds_base), 4)
            acc_f = round(accuracy_score(y_te, preds_fair), 4)
            di_b = _di(preds_base, df_te_local, prot_attr, pv, uv)
            di_f = _di(preds_fair, df_te_local, prot_attr, pv, uv)

            audit_results[prot_attr] = {
                "baseline": {"accuracy": acc_b, "disparate_impact": di_b,
                             "statistical_parity_difference": spd_b,
                             "equal_opportunity_difference": 0.0,
                             "average_odds_difference": 0.0},
                "fair_model": {"accuracy": acc_f, "disparate_impact": di_f,
                               "statistical_parity_difference": spd_f,
                               "equal_opportunity_difference": 0.0,
                               "average_odds_difference": 0.0},
                "improvement": {
                    "spd_delta": round(abs(spd_b) - abs(spd_f), 4),
                    "accuracy_delta": round(acc_f - acc_b, 4),
                    "disparate_impact_delta": round(di_f - di_b, 4),
                    "eod_delta": 0.0},
            }
            print(f"    Baseline SPD: {spd_b:.4f} | Acc: {acc_b:.1%}")
            print(f"    Fair SPD:     {spd_f:.4f} | Acc: {acc_f:.1%}")

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

        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

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
        print("  ⚠️ SHAP not installed  skipping explainability")
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
