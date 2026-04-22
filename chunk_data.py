#!/usr/bin/env python3
"""
Task 3: Split cleaned recordings into 30-second chunks.

Reads from romberg_data_cleaned/, outputs to romberg_data_final/.
- Long recordings (>3500 rows): split into 30-second chunks, discard remainder
- Short recordings (<=3500 rows): keep as-is (already ~28-35s)

Each chunk becomes its own session in the final structure.
"""

import os
import pandas as pd
import numpy as np

INPUT_DIR = os.path.join(os.path.dirname(__file__), "romberg_data_cleaned")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "romberg_data_final")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")

CHUNK_SECONDS = 30
CHUNK_ROWS_MIN = 2500  # minimum rows to accept as a valid chunk (~25s)


def estimate_sample_rate(df):
    if "seconds_elapsed" in df.columns and len(df) > 1:
        total_time = df["seconds_elapsed"].iloc[-1] - df["seconds_elapsed"].iloc[0]
        if total_time > 0:
            return len(df) / total_time
    return 100.0


def chunk_file(df, sample_rate):
    """Split a dataframe into 30-second chunks. Returns list of dataframes."""
    chunk_size = int(CHUNK_SECONDS * sample_rate)
    chunks = []
    for start in range(0, len(df) - CHUNK_ROWS_MIN + 1, chunk_size):
        end = start + chunk_size
        if end > len(df):
            remaining = len(df) - start
            if remaining >= CHUNK_ROWS_MIN:
                chunks.append(df.iloc[start:len(df)].reset_index(drop=True))
            break
        chunks.append(df.iloc[start:end].reset_index(drop=True))
    return chunks


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    manifest_rows = []
    # Track per-subject session counters for the final output
    subject_sessions = {}

    # Collect all files grouped by (subject, original_session, label)
    all_files = []
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
                label = "open" if "open" in csv_file else "closed"
                all_files.append({
                    "subject": subject_dir,
                    "orig_session": session_dir,
                    "label": label,
                    "path": os.path.join(session_path, csv_file),
                })

    # Process files grouped by subject and original session
    # For each original session that has both open and closed,
    # we chunk them and pair the chunks into new sessions
    subjects = sorted(set(f["subject"] for f in all_files))

    for subject in subjects:
        subj_files = [f for f in all_files if f["subject"] == subject]
        orig_sessions = sorted(set(f["orig_session"] for f in subj_files))

        final_session_counter = 0

        for orig_sess in orig_sessions:
            sess_files = [f for f in subj_files if f["orig_session"] == orig_sess]
            open_file = next((f for f in sess_files if f["label"] == "open"), None)
            closed_file = next((f for f in sess_files if f["label"] == "closed"), None)

            # Chunk each file
            open_chunks = []
            closed_chunks = []

            if open_file:
                df = pd.read_csv(open_file["path"])
                rate = estimate_sample_rate(df)
                if len(df) > int(CHUNK_SECONDS * rate * 1.15):
                    open_chunks = chunk_file(df, rate)
                else:
                    open_chunks = [df]

            if closed_file:
                df = pd.read_csv(closed_file["path"])
                rate = estimate_sample_rate(df)
                if len(df) > int(CHUNK_SECONDS * rate * 1.15):
                    closed_chunks = chunk_file(df, rate)
                else:
                    closed_chunks = [df]

            # Write chunks as separate sessions
            max_chunks = max(len(open_chunks), len(closed_chunks))

            for i in range(max_chunks):
                dst_dir = os.path.join(OUTPUT_DIR, subject,
                                       f"session_{final_session_counter}")
                os.makedirs(dst_dir, exist_ok=True)

                if i < len(open_chunks):
                    chunk = open_chunks[i]
                    dst = os.path.join(dst_dir, "eyes_open.csv")
                    chunk.to_csv(dst, index=False)
                    manifest_rows.append({
                        "subject_id": subject,
                        "session_id": final_session_counter,
                        "label": "open",
                        "row_count": len(chunk),
                        "duration_approx_s": round(len(chunk) / rate, 1),
                        "source": f"{orig_sess} chunk {i+1}/{len(open_chunks)}",
                    })

                if i < len(closed_chunks):
                    chunk = closed_chunks[i]
                    dst = os.path.join(dst_dir, "eyes_closed.csv")
                    chunk.to_csv(dst, index=False)
                    manifest_rows.append({
                        "subject_id": subject,
                        "session_id": final_session_counter,
                        "label": "closed",
                        "row_count": len(chunk),
                        "duration_approx_s": round(len(chunk) / rate, 1),
                        "source": f"{orig_sess} chunk {i+1}/{len(closed_chunks)}",
                    })

                final_session_counter += 1

        print(f"{subject}: {final_session_counter} sessions in final output")

    # Write manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    # Summary
    total_open = sum(1 for r in manifest_rows if r["label"] == "open")
    total_closed = sum(1 for r in manifest_rows if r["label"] == "closed")

    print(f"\n{'='*60}")
    print(f"CHUNKING COMPLETE")
    print(f"{'='*60}")
    print(f"Total recordings (data points): {len(manifest_rows)}")
    print(f"  Eyes open:   {total_open}")
    print(f"  Eyes closed: {total_closed}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")

    print(f"\nPer-subject breakdown:")
    for subject in subjects:
        entries = [r for r in manifest_rows if r["subject_id"] == subject]
        n_open = sum(1 for e in entries if e["label"] == "open")
        n_closed = sum(1 for e in entries if e["label"] == "closed")
        sessions = len(set(e["session_id"] for e in entries))
        print(f"  {subject}: {sessions} sessions, {n_open} open + {n_closed} closed = {len(entries)} data points")


if __name__ == "__main__":
    main()
