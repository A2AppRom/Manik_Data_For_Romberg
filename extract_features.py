#!/usr/bin/env python3
"""
Task 4: Extract features from each 30-second chunk.

Reads from romberg_data_final/, computes summary statistics on sway
magnitude using a windowed approach, outputs features_dataset.csv.

Features (6 global + 6 temporal = 12 total):
  Global (computed over the whole chunk):
    1. Mean — average sway magnitude
    2. Median — robust center of sway
    3. Standard Deviation — sway variability
    4. Skewness — directional bias in sway
    5. Kurtosis — spikes in movement
    6. Path Length — average per-step displacement (normalized by sample count)

  Temporal (std of each feature across 6 equal time windows):
    7-12. temporal_* — captures how each feature varies over the 30-second period
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

INPUT_DIR = os.path.join(os.path.dirname(__file__), "romberg_data_final")
OUTPUT_DIR = os.path.dirname(__file__)

NUM_WINDOWS = 6


def compute_window_features(mag):
    """Compute the 6 features for a single window of magnitude data."""
    n = len(mag)
    return {
        "mean": np.mean(mag),
        "median": np.median(mag),
        "std": np.std(mag),
        "skewness": stats.skew(mag),
        "kurtosis": stats.kurtosis(mag),
        "path_length": np.sum(np.abs(np.diff(mag))) / (n - 1) if n > 1 else 0,
    }


def extract_features(filepath):
    """Extract 12 features (6 global + 6 temporal) from a single CSV chunk."""
    df = pd.read_csv(filepath)
    mag = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    n = len(mag)

    global_feats = compute_window_features(mag)

    window_size = n // NUM_WINDOWS
    window_results = {k: [] for k in global_feats}

    for w in range(NUM_WINDOWS):
        start = w * window_size
        end = start + window_size
        win_feats = compute_window_features(mag[start:end])
        for k in win_feats:
            window_results[k].append(win_feats[k])

    temporal_feats = {}
    for k in window_results:
        temporal_feats[f"temporal_{k}"] = np.std(window_results[k])

    return {**global_feats, **temporal_feats}


FEATURE_COLS = [
    "mean", "median", "std", "skewness", "kurtosis", "path_length",
    "temporal_mean", "temporal_median", "temporal_std",
    "temporal_skewness", "temporal_kurtosis", "temporal_path_length",
]


def main():
    rows = []

    for subject_dir in sorted(os.listdir(INPUT_DIR)):
        subject_path = os.path.join(INPUT_DIR, subject_dir)
        if not os.path.isdir(subject_path) or not subject_dir.startswith("subject_"):
            continue

        for session_dir in sorted(os.listdir(subject_path), key=lambda s: int(s.split("_")[1])):
            session_path = os.path.join(subject_path, session_dir)
            if not os.path.isdir(session_path) or not session_dir.startswith("session_"):
                continue

            for csv_file in sorted(os.listdir(session_path)):
                if not csv_file.endswith(".csv"):
                    continue

                label = "open" if "open" in csv_file else "closed"
                filepath = os.path.join(session_path, csv_file)

                features = extract_features(filepath)
                features["subject_id"] = subject_dir
                features["session_id"] = int(session_dir.split("_")[1])
                features["label"] = label

                rows.append(features)

    col_order = ["subject_id", "session_id", "label"] + FEATURE_COLS
    df = pd.DataFrame(rows)[col_order]

    output_path = os.path.join(OUTPUT_DIR, "features_dataset.csv")
    df.to_csv(output_path, index=False)

    print(f"{'='*60}")
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total data points: {len(df)}")
    print(f"  Eyes open:   {len(df[df['label'] == 'open'])}")
    print(f"  Eyes closed: {len(df[df['label'] == 'closed'])}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Features: {len(FEATURE_COLS)} (6 global + 6 temporal)")
    print(f"Output: {output_path}")

    print(f"\nFeature statistics:")
    for feat in FEATURE_COLS:
        vals = df[feat]
        print(f"  {feat:20s}: min={vals.min():.4f}  max={vals.max():.4f}  "
              f"mean={vals.mean():.4f}  std={vals.std():.4f}")

    print(f"\nLabel means (open vs closed):")
    for feat in FEATURE_COLS:
        open_mean = df[df["label"] == "open"][feat].mean()
        closed_mean = df[df["label"] == "closed"][feat].mean()
        diff_pct = ((closed_mean - open_mean) / open_mean * 100) if open_mean != 0 else 0
        print(f"  {feat:20s}: open={open_mean:.4f}  closed={closed_mean:.4f}  "
              f"diff={diff_pct:+.1f}%")


if __name__ == "__main__":
    main()
