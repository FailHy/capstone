# 💪 AI-Based Exercise Evaluation System

## Biceps & Triceps Movement Analysis
**MediaPipe + XGBoost**

---

## 📋 Overview

This project implements a production-grade, computer vision-based system to evaluate exercise movements (focused on biceps and triceps) using:

- **MediaPipe Pose Estimation** → 2D Keypoint extraction
- **Biomechanical Feature Engineering** → Spatial (Joint Angles) & Temporal (Velocity/Duration) representation
- **Hybrid Gatekeeper (Rule-Based)** → Absolute biomechanical limit validation
- **XGBoost** → Complex pattern classification (Expert-validated error types)

---

## 🚀 Project Goals

- Detect and evaluate exercise form from video input in real-time
- Classify movement quality into specific, expert-validated categories (e.g., *Body Swing*, *Not Full Up*)
- Provide structured, explainable feedback mimicking a personal trainer
- Build a reproducible and robust ML pipeline (collection → cleaning → training → inference) immune to **Subject Bias**

---

## ⚠️ Current Status

| Aspect | Description |
|--------|-------------|
| **Phase** | *Data Scaling & Real-World Validation* |
| **Dataset** | Successfully synthesized and cleaned ~337 repetitions. Currently expanding dataset with natural multi-subject recordings (S02, S03, etc.) |
| **Error Labeling** | Upgraded to **SME** (*Subject Matter Expert*) validated labels |
| **Model** | Mild overfitting tightly controlled. Strong evaluation pipeline (GroupShuffleSplit + Learning Curves) is active |

---

## 📁 Project Structure

```
scripts/
├── config.py              # Global configuration & biomechanical thresholds
├── pose_utils.py          # MediaPipe spatial & coordinate processing
├── feature_utils.py       # Feature extraction (Core Four + Temporal) & Gatekeeper
├── collect_data.py        # Interactive dataset collection tool
├── clean_dataset.py       # Safe data filtering (Anti-Threshold Leakage)
├── train_model.py         # Training pipeline (GroupShuffleSplit + Metrics Report)
├── live_evaluator.py      # Real-time inference & UI feedback
├── data_eda.py            # Exploratory Data Analysis (EDA)
└── csvtodb.py             # PostgreSQL database uploader

dataset/                   # Raw, Synthesized, & Processed CSV data
models/                    # Trained .pkl models, encoders, and JSON schema
logs/                      # Runtime logs (Rejected samples, Live predictions)
reports/                   # EDA plots, Confusion Matrix, and Learning Curves
```

---

## ⚙️ ML Pipeline Overview

```
Video Input (Live Webcam / Video)
           ↓
MediaPipe Pose Estimation (Landmark detection & visibility check)
           ↓
Feature Engineering (Extracting 12 Spatial & Temporal features)
           ↓
Hybrid Gatekeeper (Rule-based filtering for absurd/extreme movements)
           ↓
XGBoost Classification (Multi-class movement evaluation)
           ↓
Output (Movement Quality + Specific Actionable Feedback)
```

---

## 🛠️ Step-by-Step Usage

### 🧪 1. Dataset Collection (Interactive Mode)

Run the data collection script. It now uses a clean, interactive terminal prompt (No need for complex CLI arguments):

```bash
python scripts/collect_data.py
```

**Setup:**
- Enter **Subject ID** (1-10) and **Session ID** (1-5) directly in the terminal
- **Action:** Press `SPACE` to start recording exactly 1 repetition

---

### 🎯 Expert-Validated Labeling Scheme

| Key | Label / Error Type | Description |
|-----|--------------------|-------------|
| `c` | **Correct** | Perfect form, isolated muscle |
| `1` | **Body Swing** | Torso swaying to create momentum |
| `2` | **Elbow Swing** | Elbow shifting forward/backward (unlocked shoulder) |
| `3` | **Not Full Up** | Incomplete Range of Motion (Half rep) |
| `4` | **Too Fast** | Dropping weight / zero eccentric control |
| `i` | **Unknown Incorrect** | Bad form (unspecified) |
| `SPACE` | **Start** | Begin tracking one repetition |
| `s` | **Skip** | Discard the current repetition |
| `q` | **Quit** | Exit the collector |

---

### 🧹 2. Dataset Cleaning

Clean the collected data safely:

```bash
python scripts/clean_dataset.py
```

**✔️ Cleaning Strategy (Anti-Threshold Leakage):**

Only removes **technically invalid** data, NOT biomechanically bad forms:

- Missing labels or NaN/infinite values
- Low keypoint visibility (Camera tracking loss)
- Physically impossible angle ranges (e.g., ROM < 5°)

---

### 🤖 3. Model Training & Evaluation

Train the XGBoost classifier and automatically generate evaluation reports:

```bash
python scripts/train_model.py
```

| Aspect | Description |
|--------|-------------|
| **Split Strategy** | Defaults to `GroupShuffleSplit (--split group_subject)` to prevent model memorization of a single person's body proportions |
| **Terminal Output** | Generates a detailed classification report (Accuracy, Precision, Recall, F1-Score) |

**📦 Saved Artifacts:**

- `models/biceps_xgboost_model.pkl` → Trained model
- `reports/xgboost_confusion_matrix.png` → Class separation performance
- `reports/xgboost_learning_curve.png` → Overfitting/Underfitting detection

---

### 🎥 4. Realtime Evaluation

Run the Hybrid Model on live video feed:

```bash
python scripts/live_evaluator.py
```

---

### 🗄️ 5. Database Integration (PostgreSQL)

Set your environment variable, then upload the dataset:

```bash
export DATABASE_URL='postgresql+psycopg2://user:password@localhost:5432/db_capstone'
python scripts/csvtodb.py --input dataset/data_biceps_clean.csv --table data_biceps --if-exists replace
```

> ⚠️ **Security Warning:** Never hardcode database credentials directly inside the scripts.

---

## 🧠 Feature Engineering

The model relies on a robust **12-feature array**, categorized into:

### Spatial (*The Core Four*)
- ROM Elbow
- Upper Arm Angle Std
- Torso Sway Range
- Shoulder Angle Range
- Elbow Drift

### Temporal & Kinematic
- Rep Duration
- Up/Down Phase Duration
- Elbow Velocity (Mean/Std)
- Motion Smoothness

---

## 🧪 Evaluation Notes (for Capstone/Thesis)

> ⚠️ **Never rely only on overall Accuracy.** Look at the **F1-Score (Macro Average)** to prove the model can detect minority error classes.

> 📊 The `xgboost_learning_curve.png` is the primary proof that the model is **generalized** and not suffering from *threshold leakage*.

---

## 📌 Development Roadmap

- [x] Refactor Codebase (Modular Architecture)
- [x] Eradicate Threshold Leakage from Clean Script
- [x] Adopt Expert-Validated Error Types (SME alignment)
- [x] Add Temporal & Kinematic Features
- [x] Synthesize & Rescue Legacy Dataset
- [ ] Expand dataset with natural variations (Subject S02 - S05) ⏳ *(In Progress)*
- [ ] Test Real-time UI robustness with Unseen Subjects

---

## 📜 License

For academic and research purposes (**Capstone Project**).

---

> 📌 **Note:** Ensure all dependencies are installed before running the scripts. Use a virtual environment to maintain project consistency.
```