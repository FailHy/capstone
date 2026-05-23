from __future__ import annotations

import argparse
import json
import pickle
import shutil
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
    CLEAN_DATASET_PATH, 
    REPORT_DIR,
    FEATURE_COLUMNS, 
    MODEL_DIR, 
    LABEL_ENCODER_PATH,
)

# -----------------------------------------------------------------------------
# 1. FUNGSI PERSIAPAN DATA
# -----------------------------------------------------------------------------
def clean_previous_artifacts() -> None:
    """Membersihkan artefak model lama agar tidak terjadi penumpukan/konflik."""
    for path in [MODEL_DIR, REPORT_DIR]:
        if path.exists():
            for file in path.glob("*"):
                try:
                    if file.is_file(): file.unlink()
                    elif file.is_dir(): shutil.rmtree(file)
                except Exception as e:
                    print(f"Gagal menghapus {file}: {e}")

def load_and_filter_dataset(path: Path, exercise_type: str) -> pd.DataFrame:
    """Membaca dataset, memfilter jenis latihan, DAN membuang data sintesis lama."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset bersih tidak ditemukan di: {path}")
    
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset bersih kosong!")
        
    if "exercise_type" not in df.columns:
        print("WARNING: Kolom 'exercise_type' tidak ditemukan. Mengasumsikan semua data adalah Biceps.")
        df["exercise_type"] = "biceps"
        
    # --- PERBAIKAN KRUSIAL: FILTER DATA LEGACY (S00_Legacy) ---
    if "subject_id" in df.columns:
        legacy_count = len(df[df["subject_id"] == "S00_Legacy"])
        if legacy_count > 0:
            df = df[df["subject_id"] != "S00_Legacy"]
            print(f"   -> Mengabaikan {legacy_count} baris data 'S00_Legacy'. Murni menggunakan data organik.")
    # ----------------------------------------------------------

    df_filtered = df[df["exercise_type"] == exercise_type].copy()
    return df_filtered

def split_dataset(X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Memecah data menjadi Latih (Train) dan Uji (Test)."""
    # Gunakan GroupShuffleSplit hanya jika subjek lebih dari 1
    if "subject_id" in df.columns and len(df["subject_id"].unique()) > 1:
        print("   -> Menggunakan GroupShuffleSplit (Evaluasi berbasis Subjek)")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=df["subject_id"]))
        return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    else:
        print("   -> Menggunakan Stratified Random Split (Subjek kurang bervariasi)")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# -----------------------------------------------------------------------------
# 2. FUNGSI PEMBUATAN & VISUALISASI MODEL
# -----------------------------------------------------------------------------
def make_model(num_classes: int, random_state: int) -> xgb.XGBClassifier:
    """Inisialisasi arsitektur model XGBoost dengan hyperparameter anti-overfitting."""
    common = dict(
        n_estimators=150,           
        max_depth=4,                
        learning_rate=0.05,         
        subsample=0.8,              
        colsample_bytree=0.8,       
        random_state=random_state,
        eval_metric="logloss" if num_classes == 2 else "mlogloss",
    )
    if num_classes == 2:
        return xgb.XGBClassifier(objective="binary:logistic", **common)
    return xgb.XGBClassifier(objective="multi:softprob", num_class=num_classes, **common)

def save_learning_curve(evals_result: dict, exercise_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"xgboost_learning_curve_{exercise_name}.png"
    
    fig, ax = plt.subplots(figsize=(8, 5))
    metric_name = list(evals_result['validation_0'].keys())[0]
    train_errors = evals_result['validation_0'][metric_name]
    test_errors = evals_result['validation_1'][metric_name]
    epochs = range(len(train_errors))
    
    ax.plot(epochs, train_errors, label='Train Error', color='blue', linewidth=2)
    ax.plot(epochs, test_errors, label='Test Error', color='red', linewidth=2)
    
    ax.set_title(f'XGBoost Learning Curve - {exercise_name.upper()}')
    ax.set_xlabel('Epoch (Jumlah Pohon)')
    ax.set_ylabel(f'Error ({metric_name})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def save_confusion_matrix(y_true, y_pred, label_names, exercise_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"xgboost_confusion_matrix_{exercise_name}.png"
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    
    plt.title(f'Confusion Matrix - {exercise_name.upper()}')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# -----------------------------------------------------------------------------
# 3. PIPELINE PELATIHAN
# -----------------------------------------------------------------------------
def train_single_model(df_filtered: pd.DataFrame, exercise_name: str, args: argparse.Namespace) -> None:
    print(f"\n{'='*60}")
    print(f" 🚀 MEMULAI PELATIHAN MODEL: {exercise_name.upper()}")
    print(f"{'='*60}")
    
    target_col = "error_type"
    features = [col for col in FEATURE_COLUMNS if col in df_filtered.columns]
    
    df_filtered = df_filtered.dropna(subset=[target_col] + features).reset_index(drop=True)
    if len(df_filtered) < 20:
        print(f"⚠️ Melewati {exercise_name.upper()} karena datanya organik barunya terlalu sedikit ({len(df_filtered)} baris). Kumpulkan lebih banyak data!")
        return
        
    X = df_filtered[features].astype(float)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_filtered[target_col].astype(str))
    label_names = list(label_encoder.classes_)
    num_classes = len(label_names)

    X_train, X_test, y_train, y_test = split_dataset(X, y, df_filtered, args.test_size, args.random_state)

    model = make_model(num_classes=num_classes, random_state=args.random_state)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    print(f"   -> Training XGBoost ({len(X_train)} train, {len(X_test)} test)...")
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    y_pred = model.predict(X_test)
    
    save_learning_curve(model.evals_result(), exercise_name, REPORT_DIR)
    save_confusion_matrix(y_test, y_pred, label_names, exercise_name, REPORT_DIR)

    acc = accuracy_score(y_test, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_test, y_pred) * 100
    
    print("\n   📊 LAPORAN EVALUASI (TESTING DATA)")
    print(f"   - Akurasi (Accuracy) : {acc:.2f}%")
    print(f"   - Akurasi (Balanced) : {bal_acc:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model.save_model(str(MODEL_DIR / f"xgboost_{exercise_name}_model.json"))
    with (MODEL_DIR / f"{exercise_name}_xgboost_model.pkl").open("wb") as f: 
        pickle.dump(model, f)
        
    with LABEL_ENCODER_PATH.open("wb") as f: 
        pickle.dump(label_encoder, f)
        
    with (MODEL_DIR / f"feature_columns_{exercise_name}.json").open("w") as f: 
        json.dump(features, f, indent=2)

    print(f"✅ Selesai! Model {exercise_name.upper()} organik tersimpan.")


def train_all_models(args: argparse.Namespace) -> None:
    if not args.input.exists():
         print(f"ERROR: Dataset tidak ditemukan di: {args.input}")
         return

    print("\n🧹 Membersihkan artefak lama...")
    clean_previous_artifacts()
    print()

    df_full = pd.read_csv(args.input)
    if "exercise_type" not in df_full.columns:
        train_single_model(df_full, "biceps", args)
        return
        
    available_exercises = df_full["exercise_type"].dropna().unique()
    
    for exercise in available_exercises:
        df_filtered = load_and_filter_dataset(args.input, exercise)
        train_single_model(df_filtered, exercise, args)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=CLEAN_DATASET_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    main_args = parse_args()
    train_all_models(main_args)