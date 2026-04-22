#!/usr/bin/env python3
"""
Task 5: Compare SVM vs Logistic Regression using GroupKFold (LOSO).
Task 6: Implement proper cross-validation with GroupKFold.

Uses sklearn's GroupKFold with subject_id as groups so each fold
holds out all recordings from one subject.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import json
import os

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "features_dataset.csv")
OUTPUT_DIR = os.path.dirname(__file__)


def run_grouped_cv(model_class, model_params, X, y, groups, model_name):
    """Run GroupKFold cross-validation and return per-fold + aggregate metrics."""
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    gkf = GroupKFold(n_splits=n_groups)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test_scaled)
        else:
            y_prob = y_pred.astype(float)

        held_out_subject = np.unique(groups[test_idx])[0]

        fold_results.append({
            "fold": fold_idx + 1,
            "held_out_subject": held_out_subject,
            "n_test": len(y_test),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        })

    # Aggregate
    all_y_test = np.concatenate([r["y_test"] for r in fold_results])
    all_y_pred = np.concatenate([r["y_pred"] for r in fold_results])
    all_y_prob = np.concatenate([r["y_prob"] for r in fold_results])

    agg = {
        "model": model_name,
        "accuracy": accuracy_score(all_y_test, all_y_pred),
        "precision": precision_score(all_y_test, all_y_pred, zero_division=0),
        "recall": recall_score(all_y_test, all_y_pred, zero_division=0),
        "f1": f1_score(all_y_test, all_y_pred, zero_division=0),
        "auc": roc_auc_score(all_y_test, all_y_prob),
        "confusion_matrix": confusion_matrix(all_y_test, all_y_pred),
        "classification_report": classification_report(all_y_test, all_y_pred,
                                                        target_names=["open", "closed"]),
        "fold_results": fold_results,
        "mean_accuracy": np.mean([r["accuracy"] for r in fold_results]),
        "std_accuracy": np.std([r["accuracy"] for r in fold_results]),
    }

    return agg


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)

    feature_cols = ["mean", "median", "std", "skewness", "kurtosis", "path_length"]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"])  # closed=0, open=1
    groups = df["subject_id"].values

    print(f"Dataset: {len(df)} samples, {len(np.unique(groups))} subjects")
    print(f"Labels: {le.classes_} → {list(range(len(le.classes_)))}")
    print(f"Features: {feature_cols}")
    print(f"Cross-validation: GroupKFold (leave-one-subject-out, {len(np.unique(groups))} folds)")
    print()

    # ============================================================
    # Task 5: Compare models
    # ============================================================
    models = {
        "SVM (RBF kernel)": (SVC, {"kernel": "rbf", "probability": True, "class_weight": "balanced"}),
        "SVM (Linear)": (SVC, {"kernel": "linear", "probability": True, "class_weight": "balanced"}),
        "Logistic Regression": (LogisticRegression, {"max_iter": 1000, "class_weight": "balanced"}),
    }

    results = {}
    for name, (cls, params) in models.items():
        print(f"{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        agg = run_grouped_cv(cls, params, X, y, groups, name)
        results[name] = agg

        print(f"\nPer-fold results:")
        for r in agg["fold_results"]:
            print(f"  Fold {r['fold']:2d} ({r['held_out_subject']}): "
                  f"acc={r['accuracy']:.3f}  prec={r['precision']:.3f}  "
                  f"rec={r['recall']:.3f}  f1={r['f1']:.3f}  "
                  f"(n={r['n_test']})")

        print(f"\nAggregate results:")
        print(f"  Accuracy:  {agg['accuracy']:.4f}")
        print(f"  Precision: {agg['precision']:.4f}")
        print(f"  Recall:    {agg['recall']:.4f}")
        print(f"  F1:        {agg['f1']:.4f}")
        print(f"  AUC:       {agg['auc']:.4f}")
        print(f"  Mean fold accuracy: {agg['mean_accuracy']:.4f} ± {agg['std_accuracy']:.4f}")
        print(f"\nConfusion matrix:")
        print(f"  {agg['confusion_matrix']}")
        print(f"\nClassification report:")
        print(agg["classification_report"])

    # ============================================================
    # Summary comparison
    # ============================================================
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25s} {'Accuracy':>8s} {'Precision':>9s} {'Recall':>8s} "
          f"{'F1':>8s} {'AUC':>8s} {'Mean±Std':>12s}")
    print("-" * 80)
    for name, agg in results.items():
        print(f"{name:<25s} {agg['accuracy']:>8.4f} {agg['precision']:>9.4f} "
              f"{agg['recall']:>8.4f} {agg['f1']:>8.4f} {agg['auc']:>8.4f} "
              f"{agg['mean_accuracy']:.4f}±{agg['std_accuracy']:.4f}")

    # Save comparison to CSV
    comp_rows = []
    for name, agg in results.items():
        comp_rows.append({
            "model": name,
            "accuracy": round(agg["accuracy"], 4),
            "precision": round(agg["precision"], 4),
            "recall": round(agg["recall"], 4),
            "f1": round(agg["f1"], 4),
            "auc": round(agg["auc"], 4),
            "mean_fold_accuracy": round(agg["mean_accuracy"], 4),
            "std_fold_accuracy": round(agg["std_accuracy"], 4),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\nComparison saved to: {comp_path}")

    # Save per-fold results
    fold_rows = []
    for name, agg in results.items():
        for r in agg["fold_results"]:
            fold_rows.append({
                "model": name,
                "fold": r["fold"],
                "held_out_subject": r["held_out_subject"],
                "n_test": r["n_test"],
                "accuracy": round(r["accuracy"], 4),
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1": round(r["f1"], 4),
            })
    fold_df = pd.DataFrame(fold_rows)
    fold_path = os.path.join(OUTPUT_DIR, "cv_fold_results.csv")
    fold_df.to_csv(fold_path, index=False)
    print(f"Fold results saved to: {fold_path}")


if __name__ == "__main__":
    main()
