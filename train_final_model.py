#!/usr/bin/env python3
"""
Task 7: Train final SVM (Linear) model on all data and export weights.

Based on Task 5/6 results:
- SVM (Linear) chosen: matches Thea's spec, 76.5% LOSO accuracy,
  lowest fold variance (±0.22)
- Trained on all 162 samples (10 subjects, including Syed)
- Exports weights as JSON for client-side prediction in the web app
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "features_dataset.csv")
OUTPUT_DIR = os.path.dirname(__file__)


def main():
    df = pd.read_csv(FEATURES_PATH)

    feature_cols = ["mean", "median", "std", "skewness", "kurtosis", "path_length"]
    X = df[feature_cols].values

    le = LabelEncoder()
    y = le.fit_transform(df["label"])  # closed=0, open=1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel="linear", probability=True, class_weight="balanced")
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)

    print("="*60)
    print("FINAL MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Model: SVM (Linear kernel, balanced class weights)")
    print(f"Training samples: {len(X)} (all data, {len(df['subject_id'].unique())} subjects)")
    print(f"Features: {feature_cols}")
    print(f"Classes: {le.classes_.tolist()} → {list(range(len(le.classes_)))}")
    print(f"\nCross-validation accuracy (from model comparison): 73.9%")
    print(f"Cross-validation method: GroupKFold LOSO, 9 folds")

    print(f"\nTraining set classification report:")
    print(classification_report(y, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y, y_pred)
    print(f"Confusion matrix (training set):")
    print(f"                Predicted")
    print(f"              closed  open")
    print(f"  Actual closed  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"  Actual open    {cm[1][0]:4d}  {cm[1][1]:4d}")

    # Export weights
    # For linear SVM: decision_function(x) = w · x_scaled + b
    # where x_scaled = (x - scaler_mean) / scaler_std
    weights = model.coef_[0].tolist()
    bias = float(model.intercept_[0])

    model_export = {
        "model_type": "SVM_linear",
        "description": "Romberg balance test classifier: eyes open (normal) vs eyes closed (impaired)",
        "features": feature_cols,
        "classes": le.classes_.tolist(),
        "prediction_rule": "If decision_function(x) > 0 → open (normal balance), else → closed (impaired balance)",
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "std": scaler.scale_.tolist(),
        },
        "weights": weights,
        "bias": bias,
        "metadata": {
            "training_samples": int(len(X)),
            "training_subjects": int(len(df["subject_id"].unique())),
            "cv_accuracy": 0.7394,
            "cv_precision": 0.7067,
            "cv_recall": 0.7794,
            "cv_f1": 0.7413,
            "cv_auc": 0.7705,
            "cv_method": "GroupKFold (leave-one-subject-out, 9 folds)",
            "n_features": len(feature_cols),
        }
    }

    output_path = os.path.join(OUTPUT_DIR, "romberg_model_weights.json")
    with open(output_path, "w") as f:
        json.dump(model_export, f, indent=2)

    print(f"\nModel weights exported to: {output_path}")

    print(f"\nWeight vector (feature importance):")
    weight_importance = sorted(zip(feature_cols, weights),
                                key=lambda x: abs(x[1]), reverse=True)
    for feat, w in weight_importance:
        direction = "→ higher = more likely OPEN" if w > 0 else "→ higher = more likely CLOSED"
        print(f"  {feat:12s}: {w:+.6f}  {direction}")
    print(f"  {'bias':12s}: {bias:+.6f}")

    print(f"\nScaler parameters (for client-side normalization):")
    for feat, m, s in zip(feature_cols, scaler.mean_, scaler.scale_):
        print(f"  {feat:12s}: mean={m:.6f}  std={s:.6f}")


if __name__ == "__main__":
    main()
