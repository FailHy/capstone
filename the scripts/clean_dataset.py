# import lib
import pandas as pd 

# setup path file sesuai dengan nama file terbaru
input = 'dataset/data_training_biceps.csv'
output = 'dataset/data_biceps_clean.csv'

try:
    dataset = pd.read_csv(input)
    print(f"Total data awal: {len(dataset)}")
except FileNotFoundError:
    print(f"ERROR: File {input} tidak ditemukan. Pastikan path benar.")
    exit()

# pisahkan data berdasarkan label
correct_data = dataset[dataset['label'] == 'correct']
incorrect_data = dataset[dataset['label'] == 'incorrect']

print(f"Data correct mentah  : {len(correct_data)}")
print(f"Data incorrect mentah: {len(incorrect_data)}")

# 1. Filtrasi Data Correct (Syarat Ketat)
clean_correct = correct_data[
    (correct_data["upper_arm_angle_std"] <= 4.0) &  # Maksimal Std 4.0 (di atas ini, anggap salah)
    (correct_data["shoulder_angle_range"] <= 15.0) & # Maksimal Range 15
    (correct_data["rom_elbow"] >= 110.0)
]

# 2. Filtrasi Data Incorrect (Syarat Terbalik)
clean_incorrect = incorrect_data[
    (incorrect_data["upper_arm_angle_std"] > 4.0) | 
    (incorrect_data["shoulder_angle_range"] > 15.0) |
    (incorrect_data["rom_elbow"] < 110.0) |
    (incorrect_data["torso_sway_range"] >= 10.0)
]

print("-" * 30)
print(f"Data correct bersih  : {len(clean_correct)} (Dibuang: {len(correct_data) - len(clean_correct)})")
print(f"Data incorrect bersih: {len(clean_incorrect)} (Dibuang: {len(incorrect_data) - len(clean_incorrect)})")

# gabungkan data yang sudah bersih
clean_dataset = pd.concat([clean_correct, clean_incorrect], ignore_index=True)

# shuffle (acak) data agar saat training model tidak menghafal urutan
clean_dataset = clean_dataset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# simpan clean data
clean_dataset.to_csv(output, index=False)

print("-" * 30)
print(f"INFO: Dataset berhasil dibersihkan dan disimpan ke '{output}'")
print(f"Total data siap training: {len(clean_dataset)}")