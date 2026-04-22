#!/usr/bin/env python3
"""
Task 1: Consolidate all Romberg test data into a single standardized structure.

Duplicate analysis (verified by matching timestamps):
  - Jack Person1-4 == sophia subjects 0-3  → skip Person folders
  - Barbara == sophia subject_4            → skip Barbara repo
  - Igor short (Mar 11) == sophia subject_5 sessions 0-1 → skip Data_For_ML Igor old
  - Syed short (Mar 11) == sophia subject_6 sessions 0-1 → skip Data_For_ML Syed old
  - Sadaf == sophia subject_7              → skip Data_For_ML Sadaf
  - Manik Sophia long == sophia subject_8  → skip Manik Sophia
  - Data_For_ML Syed zips == Manik Syed long → skip zips

Unique data sources:
  1. sophia-romberg-data subjects 0-8 (canonical copy of all short recordings)
  2. Jack's own trials (Jack_Trial1-6, Jack_Impaired_1-8, Jack_Data subfolder)
  3. Igor new long recordings (Apr 22) — 4 files
  4. Syed long recordings (Apr 5, from Manik balance_data)
"""

import os
import pandas as pd

BASE = "/Users/taswarmahbub"
OUTPUT = os.path.join(BASE, "Manik_Data_For_Romberg", "romberg_data")

os.makedirs(OUTPUT, exist_ok=True)

manifest_rows = []
subject_counter = 0


def copy_csv(src, dst):
    df = pd.read_csv(src)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    expected = ["time", "seconds_elapsed", "z", "y", "x"]
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {src}")
    df = df[expected]
    df.to_csv(dst, index=False)
    return len(df)


def add_to_manifest(subject_id, subject_name, session_id, label, source_repo,
                    source_path, row_count):
    duration = round(row_count / 100.0, 1)
    manifest_rows.append({
        "subject_id": f"subject_{subject_id:02d}",
        "subject_name": subject_name,
        "session_id": session_id,
        "label": label,
        "source_repo": source_repo,
        "source_path": os.path.basename(os.path.dirname(source_path)),
        "row_count": row_count,
        "duration_approx_s": duration,
    })


def make_session_dir(subject_id, session_id):
    d = os.path.join(OUTPUT, f"subject_{subject_id:02d}", f"session_{session_id}")
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================
# 1. sophia-romberg-data: subjects 0-8
#    This is the canonical copy. Includes the people also stored
#    redundantly in Jack_Data, Barbara, Data_For_ML, and Manik.
#    subject_0 = Person1, subject_1 = Person2, subject_2 = Person3,
#    subject_3 = Person4, subject_4 = Barbara, subject_5 = Igor,
#    subject_6 = Syed, subject_7 = Sadaf, subject_8 = Sophia (chunked)
# ============================================================
print("=== sophia-romberg-data (subjects 0-8) ===")
SOPHIA = os.path.join(BASE, "sophia-romberg-data", "data")

SOPHIA_NAMES = {
    0: "Person1", 1: "Person2", 2: "Person3", 3: "Person4",
    4: "Barbara", 5: "Igor", 6: "Syed", 7: "Sadaf", 8: "Sophia",
}

for subj_dir in sorted(os.listdir(SOPHIA)):
    if not subj_dir.startswith("subject_"):
        continue
    sophia_idx = int(subj_dir.split("_")[1])
    name = SOPHIA_NAMES.get(sophia_idx, f"sophia_{sophia_idx}")
    subj_path = os.path.join(SOPHIA, subj_dir)

    for sess_dir in sorted(os.listdir(subj_path)):
        if not sess_dir.startswith("session_"):
            continue
        sess_idx = int(sess_dir.split("_")[1])
        sess_path = os.path.join(subj_path, sess_dir)
        dst = make_session_dir(subject_counter, sess_idx)

        files = os.listdir(sess_path)
        open_files = [f for f in files if "eyes_open" in f.lower() and f.endswith(".csv")]
        closed_files = [f for f in files if "eyes_closed" in f.lower() and f.endswith(".csv")]

        for of in open_files:
            src = os.path.join(sess_path, of)
            rows = copy_csv(src, os.path.join(dst, "eyes_open.csv"))
            add_to_manifest(subject_counter, name, sess_idx, "open",
                            "sophia-romberg-data", src, rows)
            print(f"  subject_{subject_counter:02d}/session_{sess_idx} eyes_open: {rows} rows")

        for cf in closed_files:
            src = os.path.join(sess_path, cf)
            rows = copy_csv(src, os.path.join(dst, "eyes_closed.csv"))
            add_to_manifest(subject_counter, name, sess_idx, "closed",
                            "sophia-romberg-data", src, rows)
            print(f"  subject_{subject_counter:02d}/session_{sess_idx} eyes_closed: {rows} rows")

    subject_counter += 1

# ============================================================
# 2. Jack's own trials (NOT duplicated in sophia)
#    Jack_Trial1-6: normal, Jack_Impaired_1-8: impaired,
#    Jack_Data/Jack_Data: 1 long session
# ============================================================
print("\n=== Jack_Data (Jack's own trials) ===")
JACK = os.path.join(BASE, "Jack_Data")
jack_subject = subject_counter
session_idx = 0

for i in range(1, 7):
    trial_dir = os.path.join(JACK, f"Jack_Trial{i}")
    if not os.path.isdir(trial_dir):
        continue
    dst = make_session_dir(jack_subject, session_idx)

    for fname, label in [("Eyes_Open.csv", "open"), ("Eyes_Closed.csv", "closed")]:
        src = os.path.join(trial_dir, fname)
        if os.path.exists(src):
            rows = copy_csv(src, os.path.join(dst, f"eyes_{label}.csv"))
            add_to_manifest(jack_subject, "Jack", session_idx, label,
                            "Jack_Data", src, rows)
            print(f"  subject_{jack_subject:02d}/session_{session_idx} eyes_{label} (Trial{i}): {rows} rows")
    session_idx += 1

for i in range(1, 9):
    trial_dir = os.path.join(JACK, f"Jack_Impaired_{i}")
    if not os.path.isdir(trial_dir):
        continue
    dst = make_session_dir(jack_subject, session_idx)

    for fname, label in [("Eyes_Open.csv", "open"), ("Eyes_Closed.csv", "closed")]:
        src = os.path.join(trial_dir, fname)
        if os.path.exists(src):
            rows = copy_csv(src, os.path.join(dst, f"eyes_{label}.csv"))
            add_to_manifest(jack_subject, "Jack", session_idx, label,
                            "Jack_Data", src, rows)
            print(f"  subject_{jack_subject:02d}/session_{session_idx} eyes_{label} (Impaired_{i}): {rows} rows")
    session_idx += 1

jj_dir = os.path.join(JACK, "Jack_Data")
if os.path.isdir(jj_dir):
    dst = make_session_dir(jack_subject, session_idx)
    for fname, label in [("Jack_Eyes_Open.csv", "open"), ("Jack_Eyes_Closed.csv", "closed")]:
        src = os.path.join(jj_dir, fname)
        if os.path.exists(src):
            rows = copy_csv(src, os.path.join(dst, f"eyes_{label}.csv"))
            add_to_manifest(jack_subject, "Jack", session_idx, label,
                            "Jack_Data", src, rows)
            print(f"  subject_{jack_subject:02d}/session_{session_idx} eyes_{label} (Jack_Data): {rows} rows")
    session_idx += 1

subject_counter += 1

# ============================================================
# 3. Igor new long recordings (Apr 22) — unique, not in sophia
#    Add as additional sessions under the existing Igor subject
#    Igor is subject_05 (sophia subject_5). We need to add sessions
#    after sophia's existing ones.
# ============================================================
print("\n=== Igor new long recordings (Apr 22) ===")
DML = os.path.join(BASE, "Data_For_Machine_Learning_Model")
igor_subject = 5  # sophia subject_5 = Igor

# Find the next available session index for Igor
existing_igor_sessions = [r for r in manifest_rows if r["subject_id"] == "subject_05"]
next_sess = max(r["session_id"] for r in existing_igor_sessions) + 1 if existing_igor_sessions else 0

# Session: eyesOpenIgor + igorEyesClosed (pair 1)
dst = make_session_dir(igor_subject, next_sess)
src = os.path.join(DML, "eyesOpenIgor-2026-04-22_03-13-26", "Accelerometer.csv")
rows = copy_csv(src, os.path.join(dst, "eyes_open.csv"))
add_to_manifest(igor_subject, "Igor", next_sess, "open", "Data_For_ML", src, rows)
print(f"  subject_{igor_subject:02d}/session_{next_sess} eyes_open (long ~7min): {rows} rows")

src = os.path.join(DML, "igorEyesClosed-2026-04-22_03-26-13", "Accelerometer.csv")
rows = copy_csv(src, os.path.join(dst, "eyes_closed.csv"))
add_to_manifest(igor_subject, "Igor", next_sess, "closed", "Data_For_ML", src, rows)
print(f"  subject_{igor_subject:02d}/session_{next_sess} eyes_closed (long ~2min): {rows} rows")

next_sess += 1

# Session: igorEyesOpen2 + eyesclosedigor (pair 2)
dst = make_session_dir(igor_subject, next_sess)
src = os.path.join(DML, "igorEyesOpen2-2026-04-22_03-21-34", "Accelerometer.csv")
rows = copy_csv(src, os.path.join(dst, "eyes_open.csv"))
add_to_manifest(igor_subject, "Igor", next_sess, "open", "Data_For_ML", src, rows)
print(f"  subject_{igor_subject:02d}/session_{next_sess} eyes_open (long ~3.5min): {rows} rows")

src = os.path.join(DML, "eyesclosedigor-2026-04-22_03-29-48", "Accelerometer.csv")
rows = copy_csv(src, os.path.join(dst, "eyes_closed.csv"))
add_to_manifest(igor_subject, "Igor", next_sess, "closed", "Data_For_ML", src, rows)
print(f"  subject_{igor_subject:02d}/session_{next_sess} eyes_closed (long ~9min): {rows} rows")

# ============================================================
# 4. Syed long recordings (Apr 5, from Manik balance_data) — unique
#    Syed is subject_06 (sophia subject_6). Add as extra session.
# ============================================================
print("\n=== Syed long recordings (Manik balance_data) ===")
syed_subject = 6  # sophia subject_6 = Syed

existing_syed_sessions = [r for r in manifest_rows if r["subject_id"] == "subject_06"]
next_sess = max(r["session_id"] for r in existing_syed_sessions) + 1 if existing_syed_sessions else 0

dst = make_session_dir(syed_subject, next_sess)
src = os.path.join(BASE, "Manik_Data_For_Romberg", "balance_data", "eyes_open",
                   "Accelerometer_Syed_Open.csv")
rows = copy_csv(src, os.path.join(dst, "eyes_open.csv"))
add_to_manifest(syed_subject, "Syed", next_sess, "open", "Manik_balance_data", src, rows)
print(f"  subject_{syed_subject:02d}/session_{next_sess} eyes_open (long ~5min): {rows} rows")

src = os.path.join(BASE, "Manik_Data_For_Romberg", "balance_data", "eyes_closed",
                   "Accelerometer_Syed_Closed.csv")
rows = copy_csv(src, os.path.join(dst, "eyes_closed.csv"))
add_to_manifest(syed_subject, "Syed", next_sess, "closed", "Manik_balance_data", src, rows)
print(f"  subject_{syed_subject:02d}/session_{next_sess} eyes_closed (long ~5min): {rows} rows")

# ============================================================
# Write manifest
# ============================================================
manifest_path = os.path.join(OUTPUT, "manifest.csv")
df_manifest = pd.DataFrame(manifest_rows)
df_manifest.to_csv(manifest_path, index=False)

print(f"\n{'='*60}")
print(f"CONSOLIDATION COMPLETE (duplicates removed)")
print(f"{'='*60}")
print(f"Total unique subjects: {subject_counter}")
print(f"Total recordings: {len(manifest_rows)}")
print(f"  Eyes open:   {sum(1 for r in manifest_rows if r['label'] == 'open')}")
print(f"  Eyes closed: {sum(1 for r in manifest_rows if r['label'] == 'closed')}")
print(f"Output directory: {OUTPUT}")
print(f"Manifest: {manifest_path}")

print(f"\nSubject summary:")
unique_subjects = sorted(set(r["subject_id"] for r in manifest_rows))
for sid in unique_subjects:
    entries = [r for r in manifest_rows if r["subject_id"] == sid]
    name = entries[0]["subject_name"]
    n_open = sum(1 for e in entries if e["label"] == "open")
    n_closed = sum(1 for e in entries if e["label"] == "closed")
    total_rows = sum(e["row_count"] for e in entries)
    long_marker = " *has long recordings*" if any(e["row_count"] > 10000 for e in entries) else ""
    print(f"  {sid} ({name}): {n_open} open + {n_closed} closed, "
          f"{total_rows:,} total rows{long_marker}")
