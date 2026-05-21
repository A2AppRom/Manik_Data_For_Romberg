#!/usr/bin/env python3
"""
Retrain the KNN model using local training data + new labeled samples from Supabase.

Steps:
  1. Load existing training data from results/features_dataset.csv (111 samples)
  2. Pull labeled samples from Supabase (label IS NOT NULL)
  3. For Supabase samples missing temporal features, download the raw CSV
     from Supabase Storage and re-extract all 12 features
  4. Combine datasets and retrain KNN (k=7, distance-weighted)
  5. Run LOSO cross-validation and report accuracy
  6. Export new model weights to model/romberg_model_weights.json
  7. Update the embedded MODEL object in ui/index.html

Usage:
    export SB_URL="https://xmxyschvrqcqhqznjdgu.supabase.co"
    export SB_SERVICE_ROLE_KEY="..."
    python pipeline/retrain_from_supabase.py
"""

import os
import sys
import io
import json
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from supabase import create_client

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
FEATURES_PATH = os.path.join(PROJECT_ROOT, "results", "features_dataset.csv")
MODEL_OUTPUT = os.path.join(PROJECT_ROOT, "model", "romberg_model_weights.json")
HTML_PATH = os.path.join(PROJECT_ROOT, "ui", "index.html")

FEATURE_COLS = [
    "mean", "median", "std", "skewness", "kurtosis", "path_length",
    "temporal_mean", "temporal_median", "temporal_std",
    "temporal_skewness", "temporal_kurtosis", "temporal_path_length",
]

NUM_WINDOWS = 6


def compute_window_features(mag):
    n = len(mag)
    return {
        "mean": float(np.mean(mag)),
        "median": float(np.median(mag)),
        "std": float(np.std(mag)),
        "skewness": float(stats.skew(mag)),
        "kurtosis": float(stats.kurtosis(mag)),
        "path_length": float(np.sum(np.abs(np.diff(mag))) / (n - 1)) if n > 1 else 0.0,
    }


def extract_features_from_csv_text(csv_text):
    df = pd.read_csv(io.StringIO(csv_text))

    accel_cols = None
    if {"x", "y", "z"}.issubset(df.columns):
        accel_cols = ("x", "y", "z")
    elif {"userAcceleration.x", "userAcceleration.y", "userAcceleration.z"}.issubset(df.columns):
        accel_cols = ("userAcceleration.x", "userAcceleration.y", "userAcceleration.z")
    else:
        for cx in df.columns:
            if re.search(r"accel.*x", cx, re.IGNORECASE):
                cy = next((c for c in df.columns if re.search(r"accel.*y", c, re.IGNORECASE)), None)
                cz = next((c for c in df.columns if re.search(r"accel.*z", c, re.IGNORECASE)), None)
                if cy and cz:
                    accel_cols = (cx, cy, cz)
                    break

    if accel_cols is None:
        return None

    ax = pd.to_numeric(df[accel_cols[0]], errors="coerce")
    ay = pd.to_numeric(df[accel_cols[1]], errors="coerce")
    az = pd.to_numeric(df[accel_cols[2]], errors="coerce")

    valid = ax.notna() & ay.notna() & az.notna()
    ax, ay, az = ax[valid].values, ay[valid].values, az[valid].values
    n = len(ax)
    if n < 10:
        return None

    mag = np.sqrt(ax**2 + ay**2 + az**2)
    if np.mean(mag) > 5:
        ax = ax - np.mean(ax)
        ay = ay - np.mean(ay)
        az = az - np.mean(az)
        mag = np.sqrt(ax**2 + ay**2 + az**2)

    global_feats = compute_window_features(mag)

    window_size = n // NUM_WINDOWS
    window_results = {k: [] for k in global_feats}
    for w in range(NUM_WINDOWS):
        start = w * window_size
        end = start + window_size
        wf = compute_window_features(mag[start:end])
        for k in wf:
            window_results[k].append(wf[k])

    temporal_feats = {}
    for k in window_results:
        temporal_feats[f"temporal_{k}"] = float(np.std(window_results[k]))

    return {**global_feats, **temporal_feats}


def fetch_supabase_samples(sb):
    result = sb.table("samples").select("*").not_.is_("label", "null").execute()
    return result.data


def download_raw_csv(sb, storage_path):
    try:
        resp = sb.storage.from_("csv-uploads").download(storage_path)
        return resp.decode("utf-8")
    except Exception as e:
        print(f"  WARNING: Could not download {storage_path}: {e}")
        return None


def has_all_features(extracted_features):
    return all(f"temporal_{k}" in extracted_features for k in ["mean", "median", "std", "skewness", "kurtosis", "path_length"])


def run_loso_cv(X, y, groups, k=7):
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        acc = np.mean(y_pred == y_test)
        fold_results.append({
            "subject": groups[test_idx[0]],
            "accuracy": acc,
            "n_test": len(test_idx),
        })

    meaningful = [f for f in fold_results if f["n_test"] > 2]
    if meaningful:
        total_samples = sum(f["n_test"] for f in meaningful)
        weighted_acc = sum(f["accuracy"] * f["n_test"] for f in meaningful) / total_samples
    else:
        weighted_acc = np.mean([f["accuracy"] for f in fold_results])

    aggregate_acc = np.mean([f["accuracy"] for f in fold_results])
    return fold_results, weighted_acc, aggregate_acc


def update_html_model(model_export, fold_results):
    with open(HTML_PATH, "r") as f:
        html = f.read()

    scaler_mean_str = "[" + ", ".join(f"{v:.8f}" for v in model_export["scaler"]["mean"]) + "]"
    scaler_std_str = "[" + ", ".join(f"{v:.8f}" for v in model_export["scaler"]["std"]) + "]"

    feature_names_str = json.dumps(model_export["features"])

    training_features = []
    training_labels = []
    for pt in model_export["training_data"]:
        tf = "[" + ", ".join(f"{v:.6f}" for v in pt["features"]) + "]"
        training_features.append(tf)
        training_labels.append(str(pt["label"]))

    training_features_str = "[" + ", ".join(training_features) + "]"
    training_labels_str = "[" + ", ".join(training_labels) + "]"

    per_subject_entries = []
    for fr in fold_results:
        per_subject_entries.append(
            f'{{subject:"{fr["subject"]}",accuracy:{fr["accuracy"]:.3f},n:{fr["n_test"]}}}'
        )
    per_subject_str = "[\n    " + ",".join(per_subject_entries) + "\n  ]"

    weighted_acc = model_export["metadata"]["cv_accuracy_weighted"]
    n_folds = model_export["metadata"]["training_subjects"]

    new_model = f"""const MODEL = {{
  feature_names: {feature_names_str},
  scaler_mean: {scaler_mean_str},
  scaler_std: {scaler_std_str},
  k: 7,
  classes: {json.dumps(model_export["classes"])},
  training_features: {training_features_str},
  training_labels: {training_labels_str},
  per_subject_accuracy: {per_subject_str},
  cv_accuracy: {weighted_acc},
  cv_method: "GroupKFold (leave-one-subject-out, {n_folds} folds, weighted)"
}};"""

    pattern = r"const MODEL = \{.*?\};"
    html_new = re.sub(pattern, new_model, html, count=1, flags=re.DOTALL)

    with open(HTML_PATH, "w") as f:
        f.write(html_new)

    print(f"  Updated: {HTML_PATH}")


def main():
    sb_url = os.environ.get("SB_URL")
    sb_key = os.environ.get("SB_SERVICE_ROLE_KEY")
    if not sb_url or not sb_key:
        sys.exit("ERROR: Set SB_URL and SB_SERVICE_ROLE_KEY environment variables")

    sb = create_client(sb_url, sb_key)

    # Step 1: Load local training data
    print("=" * 60)
    print("STEP 1: Loading local training data")
    print("=" * 60)
    local_df = pd.read_csv(FEATURES_PATH)
    print(f"  Local samples: {len(local_df)}")
    print(f"  Local subjects: {local_df['subject_id'].nunique()}")

    # Step 2: Pull labeled samples from Supabase
    print(f"\n{'=' * 60}")
    print("STEP 2: Fetching labeled samples from Supabase")
    print("=" * 60)
    sb_rows = fetch_supabase_samples(sb)
    print(f"  Labeled samples in Supabase: {len(sb_rows)}")

    new_rows = []
    for row in sb_rows:
        feats = row["extracted_features"]
        label = row["label"]
        session_id = row.get("subject_session_id", row["sample_id"])

        if has_all_features(feats):
            feature_row = {col: feats[col] for col in FEATURE_COLS}
            print(f"  Sample {row['sample_id'][:8]}... has all 12 features")
        else:
            print(f"  Sample {row['sample_id'][:8]}... missing temporal features, downloading raw CSV...")
            csv_text = download_raw_csv(sb, row["storage_path"])
            if csv_text is None:
                print(f"    SKIP: could not download raw CSV")
                continue
            feature_row = extract_features_from_csv_text(csv_text)
            if feature_row is None:
                print(f"    SKIP: could not extract features from raw CSV")
                continue
            feature_row = {col: feature_row[col] for col in FEATURE_COLS}
            print(f"    Re-extracted 12 features from raw CSV")

        feature_row["subject_id"] = f"web_{session_id[:8]}"
        feature_row["session_id"] = 0
        feature_row["label"] = label
        new_rows.append(feature_row)

    print(f"  Usable new samples: {len(new_rows)}")

    # Step 3: Combine datasets
    print(f"\n{'=' * 60}")
    print("STEP 3: Combining datasets")
    print("=" * 60)
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([local_df, new_df], ignore_index=True)
    else:
        combined_df = local_df.copy()
        print("  No new labeled samples — retraining on local data only")

    print(f"  Total samples: {len(combined_df)}")
    print(f"  Total subjects: {combined_df['subject_id'].nunique()}")
    print(f"  Eyes open: {len(combined_df[combined_df['label'] == 'open'])}")
    print(f"  Eyes closed: {len(combined_df[combined_df['label'] == 'closed'])}")

    # Step 4: Train final model
    print(f"\n{'=' * 60}")
    print("STEP 4: Training KNN (k=7, distance-weighted)")
    print("=" * 60)
    X = combined_df[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(combined_df["label"])  # closed=0, open=1
    groups = combined_df["subject_id"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=7, weights="distance")
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    print(f"  Training accuracy: {np.mean(y_pred == y):.4f}")
    print(f"  Classes: {le.classes_.tolist()}")

    # Step 5: LOSO cross-validation
    print(f"\n{'=' * 60}")
    print("STEP 5: LOSO cross-validation")
    print("=" * 60)
    fold_results, weighted_acc, aggregate_acc = run_loso_cv(X, y, groups, k=7)

    for fr in fold_results:
        marker = " *" if fr["n_test"] <= 2 else ""
        print(f"  {fr['subject']:20s}: acc={fr['accuracy']:.3f}  n={fr['n_test']}{marker}")

    print(f"\n  Weighted accuracy (folds with n>2): {weighted_acc:.3f}")
    print(f"  Aggregate accuracy (all folds):      {aggregate_acc:.3f}")

    # Step 6: Export model weights
    print(f"\n{'=' * 60}")
    print("STEP 6: Exporting model weights")
    print("=" * 60)

    training_points = []
    for i in range(len(X_scaled)):
        training_points.append({
            "features": X_scaled[i].tolist(),
            "label": int(y[i]),
        })

    model_export = {
        "model_type": "KNN",
        "description": "Romberg balance test classifier: eyes open (normal) vs eyes closed (impaired)",
        "features": FEATURE_COLS,
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
            "training_subjects": int(combined_df["subject_id"].nunique()),
            "cv_accuracy_weighted": round(weighted_acc, 3),
            "cv_aggregate_accuracy": round(aggregate_acc, 3),
            "cv_method": f"GroupKFold (leave-one-subject-out, {combined_df['subject_id'].nunique()} folds)",
            "n_features": len(FEATURE_COLS),
        },
    }

    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    with open(MODEL_OUTPUT, "w") as f:
        json.dump(model_export, f, indent=2)
    print(f"  Saved: {MODEL_OUTPUT}")

    # Step 7: Update HTML
    print(f"\n{'=' * 60}")
    print("STEP 7: Updating UI with new model weights")
    print("=" * 60)
    update_html_model(model_export, fold_results)

    # Summary
    print(f"\n{'=' * 60}")
    print("RETRAINING COMPLETE")
    print("=" * 60)
    print(f"  Training samples:  {len(X)} ({len(local_df)} local + {len(new_rows)} from Supabase)")
    print(f"  Subjects:          {combined_df['subject_id'].nunique()}")
    print(f"  LOSO weighted acc: {weighted_acc:.3f}")
    print(f"  Model exported to: {MODEL_OUTPUT}")
    print(f"  HTML updated:      {HTML_PATH}")


if __name__ == "__main__":
    main()
