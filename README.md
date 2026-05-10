# Romberger

A browser-based Romberg balance test web app that classifies eyes-open vs eyes-closed stance using smartphone accelerometer data.

---

## What is it?

Romberger is a web app inspired by the clinical Romberg test. Users upload a 30-second accelerometer CSV recorded with a phone held against their chest, and the app classifies whether they were standing with eyes open (normal balance) or eyes closed (simulated impaired balance).

The ML model (SVM Linear) runs entirely in the browser using JSON-exported weights. After prediction, a stick figure animation replays the recorded movement. All predictions are stored in a Supabase database so the dataset grows over time.

This is not a diagnostic tool — it is built for education and exploration.

---

## Data

### Self-Collected Balance Data

Collected using the [Sensor Logger](https://apps.apple.com/app/sensor-logger/id1531582925) app on iPhone following the Balance Check Protocol (phone on chest, crossed-arm hold, 30-second trials).

Location: `romberg_data/`

- 9 subjects, 142 samples (74 eyes-closed, 68 eyes-open)
- Columns: `time`, `seconds_elapsed`, `z`, `y`, `x` (raw accelerometer)
- Long recordings chunked into 30-second segments
- Subject 08 (Sophia) removed due to corrupted data (identical open/closed files)

---

## ML Pipeline

1. **Consolidate** — merge recordings from team repos into `romberg_data/`
2. **Clean** — remove sensor artifacts (startup spikes, pauses) → `romberg_data_cleaned/`
3. **Chunk** — split long recordings into 30-second segments → `romberg_data_final/`
4. **Extract features** — 6 summary statistics per sample → `features_dataset.csv`
   - mean, median, std, skewness, kurtosis, path_length (all computed from sway magnitude)
5. **Train** — SVM Linear with GroupKFold leave-one-subject-out cross-validation (9 folds)
6. **Export** — model weights to `romberg_model_weights.json` for client-side inference

### Results (LOSO CV)

| Metric    | Value |
|-----------|-------|
| Accuracy  | 73.9% |
| Precision | 70.7% |
| Recall    | 77.9% |
| F1 Score  | 74.1% |
| AUC       | 77.1% |

---

## Tech Stack

- **Python** — pandas, scikit-learn, scipy for pipeline and training
- **HTML / CSS / JS** — single-file frontend (`index.html`), client-side prediction, canvas animation
- **Supabase** — PostgreSQL database + file storage for predictions and uploaded CSVs

---

## How to Use

1. Open the [live site](https://a2approm.github.io/Manik_Data_For_Romberg/) or `index.html` locally
2. Upload a 30-second accelerometer CSV, or click "Try a demo"
3. View the prediction, confidence, extracted features, and stick figure replay
4. Past predictions are stored and visible in the prediction history section

---

## License

Educational use only. Not intended for clinical or medical applications.
