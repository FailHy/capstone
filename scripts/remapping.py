import pandas as pd
import numpy as np

def run_anomaly_audit():
    file_path = 'dataset/data_training_biceps.csv'
    
    print("="*50)
    print("  AUDIT ANOMALI DATASET BICEPS")
    print("="*50)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Gagal memuat dataset: {e}")
        return

    # Pastikan tipe data numerik
    num_cols = ['rom_elbow', 'upper_arm_angle_std', 'torso_sway_range', 'shoulder_angle_range',
                'rep_duration', 'elbow_velocity_mean']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    total_rows = len(df)
    print(f"Total Data Dianalisis: {total_rows} baris\n")

    # 1. AUDIT BATAS LOGIS BIOMEKANIK (Logical Bounds)
    print("--- 1. AUDIT BATAS LOGIS (IMPOSSIBLE MOVEMENT) ---")
    absurd_rom = df[(df['rom_elbow'] < 10) | (df['rom_elbow'] > 160)]
    absurd_duration = df[(df['rep_duration'] < 0.2) | (df['rep_duration'] > 10.0)]
    absurd_std = df[df['upper_arm_angle_std'] > 80]

    print(f"[{len(absurd_rom)}] baris memiliki ROM Siku tidak masuk akal (<10 atau >160 derajat)")
    print(f"[{len(absurd_duration)}] baris memiliki Durasi Repetisi tidak wajar (<0.2s atau >10s)")
    print(f"[{len(absurd_std)}] baris memiliki getaran lengan atas ekstrem (Std > 80)")

    # 2. AUDIT KONTRIBUSI KELAS (Logical Contradiction)
    print("\n--- 2. AUDIT KONTRADIKSI LABEL ---")
    # Kasus: Labelnya 'correct' tapi ROM-nya sangat kecil (Half rep)
    contradiction_1 = df[(df['error_type'] == 'correct') & (df['rom_elbow'] < 90)]
    print(f"[{len(contradiction_1)}] baris berlabel 'Correct' TAPI rentang siku (ROM) di bawah 90 derajat")

    # Kasus: Labelnya 'correct' tapi badan goyang parah
    contradiction_2 = df[(df['error_type'] == 'correct') & (df['torso_sway_range'] > 25)]
    print(f"[{len(contradiction_2)}] baris berlabel 'Correct' TAPI badan berayun ekstrem (>25 derajat)")

    # 3. DETEKSI OUTLIER STATISTIK (Metode IQR)
    print("\n--- 3. DETEKSI OUTLIER STATISTIK (IQR METHOD) ---")
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_pct = (len(outliers) / total_rows) * 100
        
        # Beri peringatan jika outlier di suatu fitur melebihi 10%
        status = "PERLU DICEK" if outlier_pct > 10.0 else "AMAN"
        print(f"{col:<22}: {len(outliers):>3} outliers ({outlier_pct:>4.1f}%) -> {status}")

    print("\nKESIMPULAN:")
    total_warnings = len(absurd_rom) + len(absurd_duration) + len(contradiction_1) + len(contradiction_2)
    if total_warnings == 0:
        print("Dataset sintesis Anda SANGAT SEHAT. Silakan langsung jalankan clean_dataset.py dan train_model.py!")
    else:
        print("Ditemukan beberapa anomali/kontradiksi. clean_dataset.py akan secara otomatis")
        print("membuang anomali ini saat Anda menjalankannya nanti.")
    print("="*50)

if __name__ == "__main__":
    run_anomaly_audit()