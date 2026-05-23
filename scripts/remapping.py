import pandas as pd
import numpy as np
from pathlib import Path

def harmonize_dataset():
    input_file = Path("dataset/data_training_biceps.csv")
    output_file = Path("dataset/data_training_harmonized.csv")
    
    # 1. Definisikan kolom target yang final (23 kolom)
    target_cols = [
        "sample_id", "timestamp", "subject_id", "session_id", "camera_position",
        "lighting_condition", "exercise_type", "active_arm", "rom_elbow",
        "upper_arm_angle_std", "torso_sway_range", "shoulder_angle_range",
        "rep_duration", "up_phase_duration", "down_phase_duration",
        "elbow_velocity_mean", "elbow_velocity_std", "motion_smoothness",
        "elbow_drift_range", "mean_visibility", "label", "error_type", "notes"
    ]
    
    df = pd.read_csv(input_file)
    print(f"Dataset awal: {len(df)} baris. Memulai harmonisasi...")

    # 2. Imputasi fitur temporal (Nilai rata-rata dari distribusi data baru)
    # Gunakan nilai yang masuk akal secara biomekanik sebagai pengisi data lama
    defaults = {
        "timestamp": "2026-05-19T00:00:00",
        "subject_id": "S00_Legacy",
        "session_id": "session_legacy",
        "camera_position": "side",
        "lighting_condition": "normal",
        "exercise_type": "biceps",
        "rep_duration": 1.5,
        "up_phase_duration": 0.75,
        "down_phase_duration": 0.75,
        "elbow_velocity_mean": 140.0,
        "elbow_velocity_std": 120.0,
        "motion_smoothness": 0.05,
        "elbow_drift_range": 0.1,
        "mean_visibility": 0.95
    }

    for col, val in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else:
            df[col] = val

    # 3. Sintesis label biomekanik (mengisi error_type jika kosong)
    def infer_error(row):
        if pd.isna(row['error_type']) or row['error_type'] == 'correct':
            if row['label'] == 'correct': return 'correct'
            # Heuristik sederhana
            if row['rom_elbow'] < 100: return 'not_full_up'
            if row['shoulder_angle_range'] > 20: return 'elbow_swing'
            return 'unknown_incorrect'
        return row['error_type']

    df['error_type'] = df.apply(infer_error, axis=1)
    df['notes'] = df['notes'].fillna('auto_synthesized')

    # 4. Reorder kolom agar konsisten
    df = df[target_cols]
    
    df.to_csv(output_file, index=False)
    print(f"Harmonisasi selesai! Data tersimpan di: {output_file}")
    print(f"Total baris: {len(df)}")
    print(f"Kolom yang dihasilkan: {list(df.columns)}")

if __name__ == "__main__":
    harmonize_dataset()