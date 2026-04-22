#!/usr/bin/env python3
"""
Task 2: Clean raw data — remove phone-handling artifacts.

For each recording in romberg_data/, detect and trim acceleration spikes
at the start/end caused by picking up or placing the phone.

Output: romberg_data_cleaned/ (same structure, cleaned CSVs)
        reports/cleaning_log.csv (per-file trimming stats)
"""

import os
import numpy as np
import pandas as pd

INPUT_DIR = os.path.join(os.path.dirname(__file__), "romberg_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "romberg_data_cleaned")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")

SAMPLE_RATE = 100  # approximate Hz
SAFETY_BUFFER_S = 1.5  # trim this many seconds from start/end unconditionally
SPIKE_THRESHOLD_STD = 3  # trim rows deviating > N std from median magnitude


def compute_magnitude(df):
    return np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)


def find_stable_start(mag, median_mag, std_mag, sample_rate):
    """Find the first index where magnitude stabilizes (within threshold of median)."""
    threshold = SPIKE_THRESHOLD_STD * std_mag
    safety_samples = int(SAFETY_BUFFER_S * sample_rate)

    # Start after safety buffer
    start = safety_samples

    # Then scan forward past any remaining spikes
    window = int(0.5 * sample_rate)  # 0.5s window
    while start < len(mag) - window:
        chunk = mag[start:start + window]
        if np.all(np.abs(chunk - median_mag) < threshold):
            break
        start += int(0.1 * sample_rate)  # advance 0.1s

    return start


def find_stable_end(mag, median_mag, std_mag, sample_rate):
    """Find the last index where magnitude is still stable."""
    threshold = SPIKE_THRESHOLD_STD * std_mag
    safety_samples = int(SAFETY_BUFFER_S * sample_rate)

    # End before safety buffer
    end = len(mag) - safety_samples

    # Then scan backward past any remaining spikes
    window = int(0.5 * sample_rate)
    while end > window:
        chunk = mag[end - window:end]
        if np.all(np.abs(chunk - median_mag) < threshold):
            break
        end -= int(0.1 * sample_rate)

    return end


def clean_recording(filepath):
    """Clean a single CSV file. Returns (cleaned_df, original_rows, cleaned_rows)."""
    df = pd.read_csv(filepath)
    original_rows = len(df)

    mag = compute_magnitude(df)

    # Use the middle 60% of the recording to estimate stable median/std
    # (avoids contamination from start/end artifacts)
    n = len(mag)
    mid_start = int(n * 0.2)
    mid_end = int(n * 0.8)
    mid_mag = mag[mid_start:mid_end]
    median_mag = np.median(mid_mag)
    std_mag = np.std(mid_mag)

    # Estimate actual sample rate from the data
    if "seconds_elapsed" in df.columns and len(df) > 1:
        total_time = df["seconds_elapsed"].iloc[-1] - df["seconds_elapsed"].iloc[0]
        if total_time > 0:
            actual_rate = len(df) / total_time
        else:
            actual_rate = SAMPLE_RATE
    else:
        actual_rate = SAMPLE_RATE

    stable_start = find_stable_start(mag.values, median_mag, std_mag, actual_rate)
    stable_end = find_stable_end(mag.values, median_mag, std_mag, actual_rate)

    if stable_start >= stable_end:
        # Fallback: just trim the safety buffer
        safety = int(SAFETY_BUFFER_S * actual_rate)
        stable_start = min(safety, len(df) // 4)
        stable_end = max(len(df) - safety, len(df) * 3 // 4)

    cleaned_df = df.iloc[stable_start:stable_end].reset_index(drop=True)
    return cleaned_df, original_rows, len(cleaned_df)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    log_rows = []
    flagged = []

    for subject_dir in sorted(os.listdir(INPUT_DIR)):
        subject_path = os.path.join(INPUT_DIR, subject_dir)
        if not os.path.isdir(subject_path) or not subject_dir.startswith("subject_"):
            continue

        for session_dir in sorted(os.listdir(subject_path)):
            session_path = os.path.join(subject_path, session_dir)
            if not os.path.isdir(session_path) or not session_dir.startswith("session_"):
                continue

            for csv_file in sorted(os.listdir(session_path)):
                if not csv_file.endswith(".csv"):
                    continue

                src = os.path.join(session_path, csv_file)
                rel_path = os.path.join(subject_dir, session_dir, csv_file)

                dst_dir = os.path.join(OUTPUT_DIR, subject_dir, session_dir)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, csv_file)

                cleaned_df, orig, cleaned = clean_recording(src)
                cleaned_df.to_csv(dst, index=False)

                trimmed = orig - cleaned
                pct = round(trimmed / orig * 100, 1) if orig > 0 else 0

                log_rows.append({
                    "file": rel_path,
                    "original_rows": orig,
                    "cleaned_rows": cleaned,
                    "rows_trimmed": trimmed,
                    "pct_trimmed": pct,
                })

                status = "FLAGGED" if pct > 20 else "ok"
                if pct > 20:
                    flagged.append(rel_path)

                print(f"  {rel_path}: {orig} → {cleaned} ({trimmed} trimmed, {pct}%) [{status}]")

    # Write cleaning log
    log_df = pd.DataFrame(log_rows)
    log_path = os.path.join(OUTPUT_DIR, "cleaning_log.csv")
    log_df.to_csv(log_path, index=False)

    print(f"\n{'='*60}")
    print(f"CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {len(log_rows)}")
    print(f"Total rows before: {sum(r['original_rows'] for r in log_rows):,}")
    print(f"Total rows after:  {sum(r['cleaned_rows'] for r in log_rows):,}")
    print(f"Total trimmed:     {sum(r['rows_trimmed'] for r in log_rows):,}")
    print(f"Cleaning log: {log_path}")

    if flagged:
        print(f"\nFLAGGED (>20% trimmed) — review these:")
        for f in flagged:
            print(f"  - {f}")
    else:
        print(f"\nNo files flagged (all <20% trimmed)")


if __name__ == "__main__":
    main()
