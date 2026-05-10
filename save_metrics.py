"""
save_metrics.py
───────────────
Run this ONCE after training to compute real metrics from your test set
and persist them to metrics.json.  MachineGuard AI loads this file at
startup so the dashboard always shows computed values, not hardcoded strings.

Usage:
    python save_metrics.py
"""

import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def main() -> None:
    # ── Load artifacts ──────────────────────────────────────────────────
    model         = joblib.load("machine_failure_model.pkl")
    feature_names = joblib.load("feature_names.pkl")

    # ── Load your held-out test split ───────────────────────────────────
    # Adjust the path / column name to match your actual test CSV.
    test_df = pd.read_csv("test_data.csv")
    X_test  = test_df[feature_names]
    y_test  = test_df["Machine failure"]          # ← change if different

    # ── Compute metrics ─────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, float] = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1 Score":  f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
    }

    # ── Save ─────────────────────────────────────────────────────────────
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅  metrics.json written:")
    for k, v in metrics.items():
        print(f"   {k}: {v*100:.2f}%")


if __name__ == "__main__":
    main()
