from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    classification_report, 
    confusion_matrix, 
    f1_score
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

# Mengambil konfigurasi path dari config.py
from config import (
    CLEAN_DATASET_PATH, CONFUSION_MATRIX_PATH, LEARNING_CURVE_PATH,
    FEATURE_COLUMNS, FEATURE_COLUMNS_PATH, LABEL_ENCODER_PATH,
    MODEL_DIR, MODEL_JSON_PATH, MODEL_PKL_PATH, MODEL_VERSION_PATH,
    TRAINING_METRICS_PATH,
)

# -----------------------------------------------------------------------------
# 1. FUNGSI PERSIAPAN DATA
# -----------------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    """Membaca dataset yang sudah bersih."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset bersih tidak ditemukan di: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset bersih kosong!")
    return df

def split_dataset(X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Memecah data menjadi Latih (Train) dan Uji (Test).
    Menggunakan GroupShuffleSplit jika subject_id ada, untuk mencegah model
    'menghafal' tubuh satu orang tertentu (Mencegah Subject Bias).
    """
    if "subject_id" in df.columns and len(df["subject_id"].unique()) > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=df["subject_id"]))
        return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# -----------------------------------------------------------------------------
# 2. FUNGSI PEMBUATAN & VISUALISASI MODEL
# -----------------------------------------------------------------------------
def make_model(num_classes: int, random_state: int) -> xgb.XGBClassifier:
    """Inisialisasi arsitektur model XGBoost dengan hyperparameter anti-overfitting."""
    common = dict(
        n_estimators=150,           # Jumlah iterasi (pohon)
        max_depth=4,                # Kedalaman pohon moderat
        learning_rate=0.05,         # Kecepatan belajar (rendah = lebih stabil)
        subsample=0.8,              # Gunakan 80% baris per pohon (Anti-overfit)
        colsample_bytree=0.8,       # Gunakan 80% fitur per pohon (Anti-overfit)
        random_state=random_state,
        eval_metric="logloss" if num_classes == 2 else "mlogloss",
    )
    if num_classes == 2:
        return xgb.XGBClassifier(objective="binary:logistic", **common)
    return xgb.XGBClassifier(objective="multi:softprob", num_class=num_classes, **common)

def save_learning_curve(evals_result: dict, output: Path) -> None:
    """Menggambar grafik Train Error vs Test Error untuk mendeteksi Overfitting."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metric_name = list(evals_result['validation_0'].keys())[0]
    train_errors = evals_result['validation_0'][metric_name]
    test_errors = evals_result['validation_1'][metric_name]
    epochs = range(len(train_errors))
    
    ax.plot(epochs, train_errors, label='Train Error (Data Latih)', color='blue', linewidth=2)
    ax.plot(epochs, test_errors, label='Test Error (Data Uji)', color='red', linewidth=2)
    
    ax.set_title('XGBoost Learning Curve (Deteksi Overfitting)')
    ax.set_xlabel('Epoch (Jumlah Pohon)')
    ax.set_ylabel(f'Error ({metric_name})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

def save_confusion_matrix(y_true, y_pred, label_names, output: Path) -> None:
    """Menggambar matriks tebakan model vs kunci jawaban asli."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    
    plt.title('Confusion Matrix Evaluasi Gerakan')
    plt.ylabel('Kunci Jawaban Asli (Actual)')
    plt.xlabel('Tebakan Model (Predicted)')
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

# -----------------------------------------------------------------------------
# 3. PIPELINE UTAMA (TRAINING & EVALUATION)
# -----------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    df = load_dataset(args.input)
    
    # Target menggunakan 'error_type' agar menghasilkan evaluasi Multi-Kelas
    target_col = "error_type"
    
    # Ambil fitur yang tersedia saja
    features = [col for col in FEATURE_COLUMNS if col in df.columns]
    
    # Bersihkan sisa data kosong
    df = df.dropna(subset=[target_col] + features).reset_index(drop=True)
    X = df[features].astype(float)

    # Encode label string (correct, elbow_swing, dll) menjadi angka (0, 1, 2, dll)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_col].astype(str))
    label_names = list(label_encoder.classes_)
    num_classes = len(label_names)

    # Pecah Dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y, df, args.test_size, args.random_state)

    # Inisialisasi & Latih Model
    model = make_model(num_classes=num_classes, random_state=args.random_state)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    print("\nMemulai proses training XGBoost...")
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    # Evaluasi (Prediksi Data Uji)
    y_pred = model.predict(X_test)
    
    # Pembuatan Visualisasi
    save_learning_curve(model.evals_result(), LEARNING_CURVE_PATH)
    save_confusion_matrix(y_test, y_pred, label_names, CONFUSION_MATRIX_PATH)

    # =========================================================================
    # LAPORAN METRIK TERMINAL (DITAMBAHKAN UNTUK KEBUTUHAN ANALISIS SKRIPSI)
    # =========================================================================
    print("\n" + "="*60)
    print(" LAPORAN EVALUASI KINERJA MODEL AI (TESTING DATA)")
    print("="*60)
    
    acc = accuracy_score(y_test, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_test, y_pred) * 100
    f1_macro = f1_score(y_test, y_pred, average='macro') * 100
    
    print(f"1. Akurasi Keseluruhan (Accuracy) : {acc:.2f}%")
    print(f"2. Akurasi Seimbang (Balanced)    : {bal_acc:.2f}%")
    print(f"3. F1-Score (Macro Average)       : {f1_macro:.2f}%")
    
    print("\n--- DETAIL PER KELAS GERAKAN (Classification Report) ---")
    print(classification_report(y_test, y_pred, labels=range(num_classes), target_names=label_names, zero_division=0))
    
    print("="*60)
    print(" PANDUAN MEMBACA METRIK (UNTUK SKRIPSI):")
    print(" - Precision : Dari semua yang ditebak 'X' oleh AI, berapa % yang benar-benar 'X'?")
    print("               (Precision rendah = Model sering menuduh gerakan benar jadi salah).")
    print(" - Recall    : Dari semua data asli 'X', berapa % yang berhasil dideteksi AI?")
    print("               (Recall rendah = Model sering kecolongan gerakan salah / lolos).")
    print(" - F1-Score  : Nilai keseimbangan antara Precision dan Recall.")
    print(" - Support   : Jumlah data uji untuk kelas tersebut.")
    print("="*60 + "\n")
    # =========================================================================

    # Simpan Artefak Model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_JSON_PATH))
    with MODEL_PKL_PATH.open("wb") as f: pickle.dump(model, f)
    with FEATURE_COLUMNS_PATH.open("w") as f: json.dump(features, f, indent=2)
    with LABEL_ENCODER_PATH.open("wb") as f: pickle.dump(label_encoder, f)

    print(f"Training Selesai! Model & Grafik telah disimpan di folder '/models' dan '/reports'.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latih model evaluasi gerakan XGBoost.")
    parser.add_argument("--input", type=Path, default=CLEAN_DATASET_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    main_args = parse_args()
    train(main_args)