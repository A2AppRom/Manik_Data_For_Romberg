# Romberger

A browser-based posture classification and Romberg balance test tool using smartphone accelerometer data.

---

## What is it?

Romberger is a single-page web app (`index.html`) that runs two ML classifiers entirely in the browser:

1. **Standing vs Sitting** — classifies posture from accelerometer CSV data (91.5% LOSO accuracy, 24 subjects)
2. **Eyes Open vs Closed (Romberg Test)** — detects whether a person had their eyes open or closed while standing still (62.5% LOSO accuracy, 2 subjects — needs more data)

Both models are logistic regression trained in Python and exported as JSON weights embedded in the frontend. After prediction, a stick figure animation replays the recorded movement.

This is not a diagnostic tool — it's built for education and exploration.

---

## Data

### Standing vs Sitting — MotionSense Dataset (Kaggle)

Source: [MotionSense Dataset](https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset) by Malekzadeh et al.

Location: `Manik_Data_For_Romberg/new_data/A_DeviceMotion_data/A_DeviceMotion_data/`

- 24 subjects, 17 activity folders (standing: `std_6/`, `std_14/`; sitting: `sit_5/`, `sit_13/`)
- Each folder contains `sub_1.csv` through `sub_24.csv`
- Columns: `attitude.roll/pitch/yaw`, `gravity.x/y/z`, `rotationRate.x/y/z`, `userAcceleration.x/y/z`
- Sample rate: ~50 Hz

Training notebook: `train_model.ipynb` → exports `model_weights.json`

### Eyes Open vs Closed — Self-Collected Balance Data

Collected using the [Sensor Logger](https://apps.apple.com/app/sensor-logger/id1531582925) app on iPhone.

Location: `balance_data/`

```
balance_data/
  eyes_open/
    Accelerometer_Sophia_Open.csv
    Accelerometer_Syed_Open.csv
  eyes_closed/
    Accelerometer_Sophia_Closed.csv
    Accelerometer_Syed_Closed.csv
```

- 2 subjects (Sophia, Syed), ~5 minutes each condition
- Columns: `time`, `seconds_elapsed`, `z`, `y`, `x` (raw accelerometer including gravity)
- Sample rate: ~100 Hz

Training notebook: `train_romberg.ipynb` → exports `romberg_model_weights.json`

---

## Tech Stack

- **Python / Jupyter** — feature extraction (8 features: rms_sway, std_x/y/z, mean_jerk, path_length, sway_mean, sway_peak), LOSO cross-validation, logistic regression training
- **HTML / CSS / JS** — single-file frontend (`index.html`), client-side prediction, canvas stick figure animation

---

## How to Use

1. Open `index.html` in a browser
2. Upload a CSV from your phone's accelerometer, or click "Try a demo"
3. View the prediction and stick figure movement replay

---

## License

Educational use only. Not intended for clinical or medical applications.
