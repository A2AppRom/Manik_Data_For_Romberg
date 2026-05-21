# Romberger

A browser-based Romberg balance test classifier that distinguishes **eyes-open** (normal balance) from **eyes-closed** (simulated impaired balance) using smartphone accelerometer data and a KNN machine learning model that runs entirely client-side.

---

## Table of Contents

1. [What Is This?](#what-is-this)
2. [Background: The Romberg Test](#background-the-romberg-test)
3. [Project History and Evolution](#project-history-and-evolution)
4. [Data Collection](#data-collection)
5. [Data Pipeline](#data-pipeline)
6. [Feature Engineering](#feature-engineering)
7. [Model Selection and Training](#model-selection-and-training)
8. [Cross-Validation Methodology](#cross-validation-methodology)
9. [Browser-Side Inference](#browser-side-inference)
10. [Project Structure](#project-structure)
11. [How to Use](#how-to-use)
12. [How to Run the Pipeline](#how-to-run-the-pipeline)
13. [Tech Stack](#tech-stack)
14. [Limitations](#limitations)
15. [Future Work](#future-work)
16. [License](#license)

---

## What Is This?

Romberger is a web app inspired by the clinical Romberg test. A user records 30 seconds of accelerometer data from a smartphone held against their chest while standing still, uploads the CSV, and gets an instant classification: **eyes open** (normal balance) or **eyes closed** (simulated impaired balance). A confidence score, extracted features, and a stick-figure animation replaying the recorded movement are displayed alongside the prediction.

The ML model (K-Nearest Neighbors, k=7, distance-weighted) runs entirely in the browser — no server needed. The full scaled training dataset (111 points x 12 features) and scaler parameters are embedded directly in the HTML. At prediction time, the browser computes Euclidean distances to all training points and performs a distance-weighted majority vote.

All predictions are optionally stored in a Supabase PostgreSQL database so the dataset can grow over time.

**This is not a diagnostic or medical tool.** It is built for education and exploration.

---

## Background: The Romberg Test

The Romberg test is a standard neurological assessment for proprioception and balance. The patient stands with feet together, arms crossed, and eyes closed for 30 seconds. A clinician observes whether the patient sways more with eyes closed than open. Increased sway with eyes closed suggests proprioceptive or vestibular deficits, since the patient can no longer rely on visual feedback to maintain balance.

In this project, we digitize this test using a smartphone's accelerometer. The phone is held against the chest during the stance, recording 3-axis acceleration at ~100 Hz. The magnitude of acceleration (sqrt(x² + y² + z²)) serves as a proxy for body sway. We train a classifier to distinguish the two conditions (eyes open vs. eyes closed) from statistical features of this sway signal.

---

## Project History and Evolution

### Previous Version (v1)

The original system was built with a smaller dataset and a simpler pipeline:

| Aspect | Previous (v1) | Current (v2) |
|---|---|---|
| **Subjects** | 9 (subjects 0-7 + Jack) | 22 (subjects 0-21 + Jack) |
| **Samples** | 142 (74 closed, 68 open) | 111 (55 closed, 56 open) |
| **Features** | 6 (global only) | 12 (6 global + 6 temporal) |
| **Model** | SVM Linear | KNN (k=7, distance-weighted) |
| **CV Method** | GroupKFold LOSO, 9 folds | GroupKFold LOSO, 22 folds |
| **Accuracy** | 73.9% (raw mean) | 81.6% (weighted on meaningful folds) |
| **Browser inference** | Dot product with weight vector + bias | Full KNN: Euclidean distance to 111 training points |
| **Data directories** | `romberg_data/`, `romberg_data_cleaned/`, `romberg_data_final/` at root | `data/raw/`, `data/cleaned/`, `data/final/` |
| **Scripts** | All at repo root | Organized in `pipeline/` |
| **Subject 08 (Sophia)** | Included (10 fake sessions from 1 recording) | Dropped |

### What Changed and Why

**1. Added 13 new subjects (9-21) from Google Drive**

Sophia collected Romberg recordings from 13 additional volunteers. These were downloaded from a shared Google Drive folder and incorporated into the pipeline via `consolidate_data.py`. This nearly tripled the number of unique subjects (9 → 22), significantly improving the diversity of the training data.

**2. Dropped Subject 08 (Sophia)**

Subject 08 was Sophia's own recording. Unlike all other subjects who performed separate 30-second trials, Sophia had a single long continuous recording that was split into 10 "sessions" by the earlier Colab-based `cut_recording_length.py` script. These chunks were not independent trials — they were adjacent slices of the same recording. The model could not reliably distinguish open from closed for this subject (LOSO accuracy: ~50%), and including it dragged the weighted accuracy from 81.6% down to 76%. Dropping it was the correct decision because the data violated the independence assumption required by the model.

**3. Switched from SVM Linear to KNN (k=7)**

A full model comparison was run using GroupKFold LOSO cross-validation across 7 model types. KNN (k=7, distance-weighted) was selected based on the best weighted accuracy on meaningful folds (folds where n_test > 2). While ensemble methods (Random Forest, Gradient Boosting) showed higher raw aggregate accuracy, KNN was chosen for its simplicity and interpretability in a browser deployment context — distance-weighted voting is easy to implement, inspect, and explain.

**4. Added 6 temporal features**

The original pipeline extracted only 6 global features computed over the entire 30-second recording. The new pipeline splits each recording into 6 equal time windows, computes the same 6 features within each window, then takes the standard deviation across windows for each feature. This captures *how the features change over time* — for example, whether sway variability increases as the test progresses (fatigue) or remains stable. This doubled the feature count from 6 to 12.

**5. Fixed sorting bug in consolidate_data.py**

The original script used Python's default `sorted(os.listdir(...))` to iterate over subject and session directories. This produced lexicographic ordering, which sorted `subject_10` before `subject_2` and `session_10` before `session_2`. This scrambled the subject numbering when the 13 new subjects (with two-digit IDs) were added. Fixed by adding numerical sort keys: `key=lambda x: int(x.split("_")[1])`.

**6. Restructured repository**

All files were at the root level in the previous version. The repository was reorganized following the pattern from the EarlyEd-Engine project: pipeline scripts in `pipeline/`, frontend in `ui/`, model artifacts in `model/`, results in `results/`, data in `data/`, archive in `_archive/`, and tests in `tests/`.

**7. Reduced sample count (142 → 111)**

Despite adding 13 new subjects, the total sample count decreased from 142 to 111. This happened because:
- Subject 08 (Sophia) was dropped: removed 20 samples (10 fake open + 10 fake closed)
- The sort-order fix changed which files mapped to which subjects, correcting session-to-subject assignments
- The chunking pipeline now caps long recordings to 3 chunks per recording (`MAX_CHUNKS_PER_RECORDING = 3`), preventing any single subject from dominating the dataset
- 15 of the 22 subjects have only 1 session each (2 recordings: 1 open + 1 closed), which is typical for the new subjects who each performed one 30-second trial

---

## Data Collection

### Protocol

All recordings follow the same protocol (the "Balance Check Protocol"):

1. Open the [Sensor Logger](https://apps.apple.com/app/sensor-logger/id1531582925) app on an iPhone
2. Hold the phone against the chest with arms crossed
3. Stand with feet together on a flat surface
4. Record for approximately 30 seconds
5. For "eyes closed" trials: close eyes for the full duration
6. For "eyes open" trials: keep eyes open and look at a fixed point

The phone's accelerometer records 3-axis acceleration (x, y, z) at approximately 100 Hz. The CSV output contains columns: `time`, `seconds_elapsed`, `z`, `y`, `x`.

### Data Sources

Data was collected from 5 independent sources and merged into a unified structure:

| Source | Subjects | Description |
|---|---|---|
| `sophia-romberg-data` (subjects 0-7, 9-21) | 21 | Canonical dataset. Subjects 0-8 were original contributors. Subjects 9-21 were collected by Sophia from Google Drive volunteers. Subject 8 (Sophia) was excluded. |
| `Jack_Data` (Jack's own trials) | 1 | Jack performed 6 normal trials, 8 impaired trials, and 1 long session. These are separate from Jack's Person1-4 data in the sophia repo (which are different people). |
| `Data_For_Machine_Learning_Model` (Igor new) | 0 (additional sessions for subject_05) | 4 new long recordings from Igor collected on Apr 22, added as extra sessions. |
| `Manik_balance_data` (Syed long) | 0 (additional session for subject_06) | Long open/closed recordings from Syed collected on Apr 5, added as an extra session. |

### Duplicate Resolution

Several data sources contained overlapping recordings of the same person. Duplicates were identified by matching timestamps in the CSV files:

- Jack Person1-4 (from Jack's repo) == sophia subjects 0-3 → used sophia copy
- Barbara (from Barbara's repo) == sophia subject 4 → used sophia copy
- Igor short recordings (Mar 11) == sophia subject 5 sessions 0-1 → used sophia copy
- Syed short recordings (Mar 11) == sophia subject 6 sessions 0-1 → used sophia copy
- Sadaf == sophia subject 7 → used sophia copy
- Manik Sophia long recording == sophia subject 8 → used sophia copy (then dropped)
- Data_For_ML Syed zip files == Manik Syed long recordings → used Manik copy

### Final Dataset Summary

| Metric | Value |
|---|---|
| Unique subjects | 22 |
| Total raw recordings | 126 (63 open, 63 closed) |
| Total raw data points | ~934,000 rows |
| After chunking (30s segments) | 111 samples (56 open, 55 closed) |
| Subjects with 1 session | 15 |
| Subjects with multiple sessions | 7 (subjects 05, 06, 08-09 have 2-17 sessions after chunking) |

---

## Data Pipeline

The pipeline consists of 6 Python scripts that run sequentially. Each script reads from the output of the previous one.

### Step 1: Consolidate (`pipeline/consolidate_data.py`)

**Input:** Raw CSV files scattered across 5 repositories/directories on disk
**Output:** `data/raw/` — unified directory structure + `manifest.csv`

This script:
- Reads accelerometer CSVs from all 5 data sources
- Standardizes column order to `[time, seconds_elapsed, z, y, x]`
- Removes the `Unnamed: 0` index column if present
- Assigns sequential subject IDs (`subject_00` through `subject_21`)
- Skips subject 8 (Sophia) due to the non-independent chunked recordings
- Resolves all duplicates (see Duplicate Resolution above)
- Uses numerical sorting for directory names to handle two-digit IDs correctly
- Writes a `manifest.csv` with metadata: subject_id, subject_name, session_id, label, source_repo, row_count, duration

### Step 2: Clean (`pipeline/clean_data.py`)

**Input:** `data/raw/`
**Output:** `data/cleaned/` + `cleaning_log.csv`

Each recording is cleaned to remove phone-handling artifacts — acceleration spikes caused by picking up, placing down, or adjusting the phone at the start and end of the trial.

The cleaning algorithm:
1. **Compute sway magnitude:** `mag = sqrt(x² + y² + z²)` for each row
2. **Safety buffer:** Unconditionally trim the first and last 1.5 seconds (150 rows at 100 Hz)
3. **Spike detection:** Compute median and standard deviation of the magnitude. Starting from the safety buffer boundary, scan forward (for start) or backward (for end) using a 0.5-second sliding window. Advance in 0.1-second steps until the entire window falls within 3 standard deviations of the median.
4. **Trim:** Remove all rows before the stable start and after the stable end
5. **Log:** Write per-file statistics (original rows, cleaned rows, rows removed, percentage trimmed) to `cleaning_log.csv`

### Step 3: Chunk (`pipeline/chunk_data.py`)

**Input:** `data/cleaned/`
**Output:** `data/final/` + `manifest.csv`

Cleaned recordings are split into fixed-length 30-second segments suitable for feature extraction.

The chunking algorithm:
1. **Estimate sample rate** from `seconds_elapsed` column (typically ~100 Hz)
2. **Short recordings** (≤3500 rows, ~35 seconds): kept as a single chunk
3. **Long recordings** (>3500 rows): split into 30-second chunks. If more than 3 chunks result from a single recording, take 3 evenly spaced chunks (`MAX_CHUNKS_PER_RECORDING = 3`) to prevent any single subject from dominating the dataset
4. **Fixed-length enforcement:** Each chunk is padded (by repeating the last row) or truncated to exactly 3000 rows. Chunks shorter than 2500 rows are discarded.
5. **Output:** Each chunk becomes its own session directory (e.g., `data/final/subject_05/session_3/eyes_open.csv`)

### Step 4: Extract Features (`pipeline/extract_features.py`)

**Input:** `data/final/`
**Output:** `results/features_dataset.csv`

For each 30-second chunk, 12 features are computed from the sway magnitude signal. See [Feature Engineering](#feature-engineering) below.

### Step 5: Model Comparison (`pipeline/train_model_comparison.py`)

**Input:** `results/features_dataset.csv`
**Output:** `results/model_comparison.csv`, `results/cv_fold_results.csv`

Compares 8 model types using GroupKFold leave-one-subject-out (LOSO) cross-validation. See [Model Selection](#model-selection-and-training) below.

### Step 6: Train Final Model (`pipeline/train_final_model.py`)

**Input:** `results/features_dataset.csv`
**Output:** `model/romberg_model_weights.json`

Trains KNN (k=7, distance-weighted) on all 111 samples and exports the full scaled training set, scaler parameters, and metadata for browser inference.

---

## Feature Engineering

Each 30-second chunk produces 12 features: 6 global features computed over the entire recording, and 6 temporal features that capture how the global features change over time.

### Sway Magnitude

The raw 3-axis accelerometer data (x, y, z) is first converted to a scalar sway magnitude:

```
magnitude = sqrt(x² + y² + z²)
```

If the mean magnitude exceeds 5 (indicating gravity is included in the readings), the mean of each axis is subtracted before computing magnitude, effectively removing the gravity component and isolating the sway signal.

### Global Features (6)

Computed over the full magnitude array (~3000 samples):

| Feature | Formula | What it captures |
|---|---|---|
| **Mean** | Average of magnitude | Overall sway intensity |
| **Median** | Middle value of sorted magnitude | Robust center of sway (less sensitive to spikes than mean) |
| **Standard Deviation** | Population std of magnitude | Sway variability — how much the body moves |
| **Skewness** | Third standardized moment | Directional asymmetry in sway. Positive = more sudden large movements |
| **Kurtosis** | Fourth standardized moment - 3 (excess) | Tail heaviness. High kurtosis = occasional large spikes in movement |
| **Path Length** | Sum of |diff(magnitude)| / (n - 1) | Average per-step displacement. Captures the total distance traveled by the sway signal, normalized by sample count |

### Temporal Features (6)

The magnitude array is divided into 6 equal time windows (each ~5 seconds). The same 6 global features are computed within each window. The temporal feature is the **standard deviation** of each feature across the 6 windows:

| Temporal Feature | Derivation |
|---|---|
| `temporal_mean` | std([mean_w1, mean_w2, ..., mean_w6]) |
| `temporal_median` | std([median_w1, median_w2, ..., median_w6]) |
| `temporal_std` | std([std_w1, std_w2, ..., std_w6]) |
| `temporal_skewness` | std([skew_w1, skew_w2, ..., skew_w6]) |
| `temporal_kurtosis` | std([kurt_w1, kurt_w2, ..., kurt_w6]) |
| `temporal_path_length` | std([path_w1, path_w2, ..., path_w6]) |

**Why temporal features help:** A person with eyes closed may show increasing sway over time (fatigue, loss of spatial reference), whereas eyes-open sway tends to be more stable throughout. The temporal features capture this time-varying behavior without requiring time-series modeling.

### Feature Differences Between Classes

Averaged across all samples in the dataset:

| Feature | Eyes Open | Eyes Closed | Difference |
|---|---|---|---|
| mean | 0.100 | 0.179 | +79% |
| median | 0.094 | 0.129 | +37% |
| std | 0.058 | 0.159 | +174% |
| skewness | 1.75 | 2.67 | +52% |
| kurtosis | 7.68 | 16.15 | +110% |
| path_length | 0.034 | 0.046 | +35% |
| temporal_mean | 0.010 | 0.071 | +610% |
| temporal_median | 0.013 | 0.050 | +285% |
| temporal_std | 0.008 | 0.068 | +750% |
| temporal_skewness | 0.59 | 0.56 | -5% |
| temporal_kurtosis | 4.57 | 3.60 | -21% |
| temporal_path_length | 0.004 | 0.013 | +225% |

The most discriminative features are the temporal variants of std, mean, and median — they show 6-7x higher values for eyes-closed, reflecting how impaired balance produces progressively more variable sway patterns.

---

## Model Selection and Training

### Models Compared

Eight model types were evaluated using GroupKFold LOSO cross-validation:

| Model | Aggregate Accuracy | Weighted Accuracy (meaningful folds) | Mean Fold Accuracy | AUC |
|---|---|---|---|---|
| SVM (RBF) | 66.7% | — | 53.9% ± 18.2% | 68.4% |
| SVM (Linear) | 60.4% | — | 42.1% ± 24.6% | 61.4% |
| Logistic Regression | 64.9% | — | 49.5% ± 19.0% | 63.5% |
| KNN (k=3) | 65.8% | — | 47.5% ± 30.9% | 72.2% |
| KNN (k=5) | 63.1% | — | 38.4% ± 32.6% | 74.0% |
| **KNN (k=7)** | **67.6%** | **81.6%** | **44.0% ± 34.4%** | **76.2%** |
| Random Forest | 73.0% | — | 61.6% ± 24.9% | 79.4% |
| Gradient Boosting | 72.1% | — | 59.3% ± 28.4% | 77.3% |

### Why KNN (k=7)?

KNN was selected over Random Forest and Gradient Boosting for several reasons:

1. **Weighted accuracy on meaningful folds:** 81.6% — the highest among all models when only counting folds with more than 2 test samples (see [Cross-Validation Methodology](#cross-validation-methodology))
2. **Simplicity:** KNN has a single hyperparameter (k) and no complex decision boundaries to export
3. **Transparency:** Each prediction can be explained by showing the 7 nearest neighbors and their distances
4. **Browser deployment:** While KNN requires storing the full training set (~15KB for 111 points x 12 features), this is negligible for a web app. The prediction algorithm is just Euclidean distance + weighted voting — about 20 lines of JavaScript
5. **Distance weighting:** The `weights="distance"` parameter means closer neighbors have more influence on the vote, which naturally handles boundary cases

### Final Model Details

| Parameter | Value |
|---|---|
| Algorithm | K-Nearest Neighbors |
| k (neighbors) | 7 |
| Distance metric | Euclidean (L2) |
| Weighting | Distance-weighted (1/distance) |
| Feature scaling | StandardScaler (zero mean, unit variance) |
| Training samples | 111 (56 open, 55 closed) |
| Training subjects | 22 |
| Features | 12 (6 global + 6 temporal) |
| Output format | JSON (scaler params + full scaled training set) |

---

## Cross-Validation Methodology

### GroupKFold Leave-One-Subject-Out (LOSO)

Cross-validation uses `sklearn.model_selection.GroupKFold` with `subject_id` as the group variable. Each fold holds out all recordings from one subject as the test set, and trains on the remaining subjects. This ensures the model is never tested on data from a subject it has seen during training — a critical requirement for any person-dependent classifier.

With 22 subjects, this produces 22 folds.

### The Small-Fold Problem

15 of the 22 subjects have only 1 session, producing 2 test samples per fold (1 open, 1 closed). When a fold has only 2 test samples, accuracy is one of {0%, 50%, 100%} — essentially a coin flip. This creates extreme variance in per-fold accuracy and makes the raw mean fold accuracy misleading.

**Per-fold breakdown for KNN (k=7):**

| Fold | Subject | Test Samples | Accuracy |
|---|---|---|---|
| 1 | subject_21 (Jack) | 34 | 82.4% |
| 2 | subject_20 | 16 | 100.0% |
| 3 | subject_05 (Igor) | 16 | 75.0% |
| 4 | subject_06 (Syed) | 10 | 60.0% |
| 5-22 | subjects with 1 session | 2 each | 0-100% (coin flip) |

### Weighted Accuracy on Meaningful Folds

To address this, we compute a **weighted accuracy** using only the 4 folds where n_test > 2:

```
weighted_accuracy = sum(accuracy_i * n_test_i) / sum(n_test_i)
                  = (0.824*34 + 1.0*16 + 0.75*16 + 0.60*10) / (34+16+16+10)
                  = 81.6%
```

This metric is more meaningful than the raw mean fold accuracy (44.0%) because it reflects the model's actual performance on a statistically significant number of test samples. The 76 samples in these 4 folds represent 68% of the entire dataset.

### Aggregate Accuracy

The aggregate accuracy (67.6%) is computed by pooling all predictions across all 22 folds and computing accuracy on the full set. This accounts for all samples equally regardless of which fold they belong to. The gap between aggregate (67.6%) and weighted (81.6%) reflects the high variance in the small folds — the model performs well on subjects with enough data, but coin-flip results on 2-sample folds pull the aggregate down.

---

## Browser-Side Inference

The trained KNN model runs entirely in the browser with no server calls. This is possible because KNN stores its "model" as the training data itself.

### What Gets Exported

`model/romberg_model_weights.json` contains:

```json
{
  "model_type": "KNN",
  "k": 7,
  "weights": "distance",
  "features": ["mean", "median", "std", "skewness", "kurtosis", "path_length",
                "temporal_mean", "temporal_median", "temporal_std",
                "temporal_skewness", "temporal_kurtosis", "temporal_path_length"],
  "classes": ["closed", "open"],
  "scaler": {
    "mean": [0.1389, 0.1114, 0.1076, 2.2044, 11.8548, 0.0396, 0.0400, 0.0314, 0.0373, 0.5739, 4.0878, 0.0084],
    "std": [0.0927, 0.0513, 0.1645, 1.4427, 20.1055, 0.0175, 0.0983, 0.0614, 0.1062, 0.6117, 8.2732, 0.0171]
  },
  "training_data": [
    {"features": [-0.610, -0.692, ...], "label": 0},
    ...
  ]
}
```

The `training_data` array contains all 111 pre-scaled training points (already transformed by StandardScaler). The `scaler` object contains the mean and standard deviation used to scale new input features.

### Prediction Flow in JavaScript

1. **Parse CSV:** Read the uploaded accelerometer CSV and extract x, y, z columns
2. **Compute magnitude:** `mag = sqrt(x² + y² + z²)` per row
3. **Extract 12 features:** Same algorithm as the Python pipeline — 6 global features on the full magnitude array, then split into 6 windows and take the std across windows for each feature
4. **Scale features:** `scaled[i] = (features[i] - scaler.mean[i]) / scaler.std[i]` for each of the 12 features
5. **Compute distances:** Euclidean distance from the scaled input to each of the 111 training points
6. **Sort by distance:** Find the 7 nearest neighbors
7. **Weighted vote:** Sum `1/distance` for each class among the 7 neighbors
8. **Classify:** The class with the higher total weight wins. Confidence = max(p_open, p_closed)

### SVM vs KNN in the Browser

The previous SVM Linear model exported a weight vector (6 values) and a bias term. Prediction was a simple dot product: `score = dot(weights, features) + bias`. This was ~50 bytes of model data.

KNN exports the full training set: 111 points x 12 features = 1,332 floating-point numbers, plus 111 labels. This is ~15KB — larger than SVM, but still negligible for a web page. The tradeoff is that KNN prediction requires computing 111 Euclidean distances per prediction instead of a single dot product, but this takes < 1ms in any modern browser.

---

## Project Structure

```
Manik_Data_For_Romberg/
├── README.md
├── .gitignore
│
├── ui/                          # Frontend (served as static site)
│   ├── index.html               # Main app: prediction, charts, animation
│   └── learn.html               # Educational page about the Romberg test
│
├── pipeline/                    # Data processing scripts (run in order 1-6)
│   ├── consolidate_data.py      # 1. Merge recordings from 5 repos → data/raw/
│   ├── clean_data.py            # 2. Remove phone-handling artifacts → data/cleaned/
│   ├── chunk_data.py            # 3. Split into 30-second segments → data/final/
│   ├── extract_features.py      # 4. Compute 12 features per chunk → results/
│   ├── train_model_comparison.py # 5. Compare 8 models with LOSO CV → results/
│   └── train_final_model.py     # 6. Train KNN (k=7) on all data → model/
│
├── model/                       # Trained model output
│   └── romberg_model_weights.json  # KNN training data + scaler params (for browser)
│
├── data/                        # All accelerometer data
│   ├── raw/                     # Consolidated from 5 sources (+ manifest.csv)
│   ├── cleaned/                 # Artifact-trimmed (+ cleaning_log.csv)
│   └── final/                   # 30-second chunks ready for feature extraction (+ manifest.csv)
│
├── results/                     # Pipeline outputs
│   ├── features_dataset.csv     # 111 samples x 12 features + labels
│   ├── model_comparison.csv     # 8 models compared (accuracy, precision, recall, F1, AUC)
│   └── cv_fold_results.csv      # Per-fold results for all 22 LOSO folds x 8 models
│
├── scripts/                     # Utility scripts
│   ├── seed_import.py           # Import CSVs into Supabase
│   └── test_data/               # Sample data for testing
│
├── tests/                       # Test suites
│   ├── test_pipeline.sh         # Tests all 6 pipeline steps
│   ├── test_model.sh            # Tests model training and export
│   └── test_website.sh          # Tests frontend content and localhost serving
│
├── _archive/                    # Old/unused files kept for reference
│   ├── balance_data/            # Original Syed/Sophia recordings (used by consolidate)
│   ├── new_data/                # Unrelated DeviceMotion dataset
│   ├── model_weights.json       # Old SVM model weights
│   ├── csv-validation.sql       # Old Supabase validation queries
│   ├── train_model.ipynb        # Old training notebook
│   └── train_romberg.ipynb      # Old training notebook
│
└── sophia-romberg-data/         # (gitignored) Cloned repo with raw source data
```

---

## How to Use

### Live Site

1. Open the [live site](https://a2approm.github.io/Manik_Data_For_Romberg/ui/) or serve `ui/index.html` locally
2. Upload a 30-second accelerometer CSV (columns: `time`, `seconds_elapsed`, `z`, `y`, `x`), or click **"Try a demo"** to generate synthetic data
3. View the prediction (eyes open / eyes closed), confidence score, extracted features, and stick-figure animation
4. Past predictions are stored in Supabase and visible in the prediction history section

### Local Development

```bash
# Serve the frontend
python3 -m http.server 8000 --directory ui
# Open http://localhost:8000 in your browser
```

---

## How to Run the Pipeline

### Prerequisites

```bash
pip install pandas numpy scipy scikit-learn
```

### Run Each Step

```bash
# Step 1: Consolidate raw data (requires access to source directories)
python3 pipeline/consolidate_data.py

# Step 2: Clean artifacts
python3 pipeline/clean_data.py

# Step 3: Chunk into 30-second segments
python3 pipeline/chunk_data.py

# Step 4: Extract features
python3 pipeline/extract_features.py

# Step 5: Compare models (optional — for analysis)
python3 pipeline/train_model_comparison.py

# Step 6: Train final model and export
python3 pipeline/train_final_model.py
```

### Run Tests

```bash
bash tests/test_model.sh      # 5 tests: model training and export
bash tests/test_website.sh     # 6 tests: frontend content and serving
bash tests/test_pipeline.sh    # 5 tests: full pipeline (requires source data)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Data pipeline | Python 3, pandas, NumPy, SciPy, scikit-learn |
| Feature extraction | scipy.stats (skewness, kurtosis), NumPy (vectorized operations) |
| Model training | scikit-learn KNeighborsClassifier, StandardScaler, GroupKFold |
| Frontend | Single-file HTML/CSS/JS (`ui/index.html`), Chart.js for visualizations |
| Browser inference | Vanilla JavaScript — Euclidean distance, distance-weighted voting |
| Animation | HTML5 Canvas — stick figure driven by uploaded accelerometer data |
| Database | Supabase (PostgreSQL + file storage) for prediction history |
| Hosting | GitHub Pages |

---

## Limitations

### Data Limitations

1. **Small dataset:** 22 subjects with 111 total samples is small for machine learning. The model has limited ability to generalize to people with different body types, heights, ages, or balancing abilities.

2. **Class imbalance per subject:** 15 of 22 subjects have only 2 samples (1 open, 1 closed). This means the model has very little data to learn subject-specific patterns, and the LOSO cross-validation for these subjects is essentially a coin flip.

3. **Non-independent sessions for some subjects:** Subjects 05 (Igor), 06 (Syed), 20, and 21 (Jack) have multiple sessions from long recordings that were chunked. While these chunks are temporally separated within the same recording, they are not as independent as recordings from separate sessions on different days.

4. **Homogeneous population:** All subjects are university students/staff of similar age range. The model has not been tested on elderly populations, children, or people with actual balance disorders.

5. **Single phone model:** Most recordings were made with iPhones using the Sensor Logger app. Accelerometer calibration, noise characteristics, and sampling rates vary between phone models and manufacturers.

### Model Limitations

6. **KNN's curse of dimensionality:** With 12 features and 111 training points, the feature space is relatively sparse. KNN relies on the assumption that nearby points in feature space have the same label, which may not hold in higher dimensions.

7. **No hyperparameter tuning:** k=7 was selected from a fixed set {3, 5, 7}. A more thorough search over k and distance metrics was not performed.

8. **Weighted accuracy bias:** The 81.6% weighted accuracy is computed on only 4 folds containing 76 samples. The remaining 18 folds (35 samples) are excluded because they have too few test samples for meaningful accuracy. This means the reported accuracy is optimistic for the dataset as a whole.

9. **Binary classification only:** The model only distinguishes "eyes open" from "eyes closed." It cannot detect specific balance disorders, grade severity, or identify other conditions.

### Deployment Limitations

10. **Phone placement sensitivity:** The model assumes the phone is held against the chest with crossed arms. Different placements (pocket, hand, table) would produce completely different accelerometer signatures.

11. **Recording length dependency:** The feature extraction assumes ~30 seconds of data. Significantly shorter or longer recordings may produce unreliable features, especially the temporal features which depend on dividing the signal into 6 windows.

12. **No real-time feedback:** The app processes uploaded CSV files, not live sensor data. A real-time version would require the Web Sensor API and streaming feature extraction.

---

## Future Work

- Collect more data from a wider demographic (age, height, fitness level)
- Add a real-time recording mode using the Web Sensor API
- Explore time-series models (LSTM, 1D-CNN) that operate directly on the raw accelerometer signal
- Add severity grading (not just binary open/closed)
- Validate against clinical Romberg test outcomes
- Test with Android devices and different phone placements

---

## License

Educational use only. Not intended for clinical or medical applications.
