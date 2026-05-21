#!/usr/bin/env python3
"""
Task 7: Train final KNN (k=7) model on all data and export for browser inference.

KNN requires exporting the full scaled training set + scaler params,
since prediction = majority vote of k nearest neighbors.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "features_dataset.csv")
OUTPUT_DIR = os.path.dirname(__file__)


def main():
    df = pd.read_csv(FEATURES_PATH)

    feature_cols = ["mean", "median", "std", "skewness", "kurtosis", "path_length",
                    "temporal_mean", "temporal_median", "temporal_std",
                    "temporal_skewness", "temporal_kurtosis", "temporal_path_length"]
    X = df[feature_cols].values

    le = LabelEncoder()
    y = le.fit_transform(df["label"])  # closed=0, open=1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=7, weights="distance")
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)

    print("="*60)
    print("FINAL MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Model: KNN (k=7, distance-weighted)")
    print(f"Training samples: {len(X)} (all data, {len(df['subject_id'].unique())} subjects)")
    print(f"Features: {feature_cols}")
    print(f"Classes: {le.classes_.tolist()} → {list(range(len(le.classes_)))}")
    print(f"\nCross-validation accuracy (from model comparison): 81.6% weighted on meaningful folds")
    print(f"Cross-validation method: GroupKFold LOSO, 22 folds")

    print(f"\nTraining set classification report:")
    print(classification_report(y, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y, y_pred)
    print(f"Confusion matrix (training set):")
    print(f"                Predicted")
    print(f"              closed  open")
    print(f"  Actual closed  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"  Actual open    {cm[1][0]:4d}  {cm[1][1]:4d}")

    training_points = []
    for i in range(len(X_scaled)):
        training_points.append({
            "features": X_scaled[i].tolist(),
            "label": int(y[i]),
        })

    model_export = {
        "model_type": "KNN",
        "description": "Romberg balance test classifier: eyes open (normal) vs eyes closed (impaired)",
        "features": feature_cols,
        "classes": le.classes_.tolist(),
        "k": 7,
        "weights": "distance",
        "prediction_rule": "Classify by distance-weighted majority vote of 7 nearest neighbors. class 0=closed (impaired), 1=open (normal)",
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "std": scaler.scale_.tolist(),
        },
        "training_data": training_points,
        "metadata": {
            "training_samples": int(len(X)),
            "training_subjects": int(len(df["subject_id"].unique())),
            "cv_accuracy_weighted": 0.816,
            "cv_aggregate_accuracy": 0.676,
            "cv_method": "GroupKFold (leave-one-subject-out, 22 folds)",
            "n_features": len(feature_cols),
        }
    }

    output_path = os.path.join(OUTPUT_DIR, "romberg_model_weights.json")
    with open(output_path, "w") as f:
        json.dump(model_export, f, indent=2)

    print(f"\nModel exported to: {output_path}")
    print(f"Training points exported: {len(training_points)}")

    print(f"\nScaler parameters (for client-side normalization):")
    for feat, m, s in zip(feature_cols, scaler.mean_, scaler.scale_):
        print(f"  {feat:25s}: mean={m:.6f}  std={s:.6f}")


if __name__ == "__main__":
    main()
