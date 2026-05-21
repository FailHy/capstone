# import library data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# load data
file_path = 'dataset/data_biceps_clean.csv'

try:
    data = pd.read_csv(file_path)
    print(f'info: dataset loaded successfully, total {len(data)} rows')
except FileNotFoundError:
    print(f'ERR: file {file_path} not found')
    exit()
    
# set style seaborn untuk visualisasi
sns.set_theme(style='whitegrid', palette='muted')

# analyzed features
features = [
    'rom_elbow',
    'upper_arm_angle_std', 
    'torso_sway_range', 
    'shoulder_angle_range'
]

# encode label ke numeric untuk correlation heatmap
le = LabelEncoder()
data['label_encoded'] = le.fit_transform(data['label'])

# cek sanity dan target distribusi
print("\n -- Info dataset --")
print(data.info())
print("\n -- Deskripsi --")
print(data.describe())

# distribusi kelas target
plt.figure(figsize=(6,4))
sns.countplot(data=data, x='label', hue='label', palette={'correct': 'blue', 'incorrect': 'red'}, legend=False)   
plt.title('Distribusi Kelas Target (Correct vs Incorrect)')
plt.ylabel('Jumlah Repetisi')
plt.savefig('eda_1_class_distribution.png')
plt.close()

# boxplot deteksi outlier
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Boxplot Distribusi Fitur Berdasarkan Kelas', fontsize=16)

for i, feature in enumerate(features):
    row, col = i // 2, i % 2
    sns.boxplot(data=data, x='label', y=feature, hue='label', ax=axes[row, col], palette={'correct': '#2ecc71', 'incorrect': '#e74c3c'}, legend=False)
    axes[row, col].set_title(f'Distribusi {feature}')

plt.tight_layout()
plt.savefig('eda_2_boxplots.png')
plt.close()

# histogram / density plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Density Plot (Melihat Overlap Antar Kelas)', fontsize=16)

for i, feature in enumerate(features):
    row, col = i // 2, i % 2
    # perbaikan: dikembalikan ke kdeplot, bukan boxplot
    sns.kdeplot(data=data, x=feature, hue='label', fill=True, alpha=0.5, ax=axes[row, col], palette={'correct': '#2ecc71', 'incorrect': '#e74c3c'})
    axes[row, col].set_title(f'Density {feature}')

plt.tight_layout()
plt.savefig('eda_3_kdeplots.png')
plt.close()

# scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='upper_arm_angle_std', y='shoulder_angle_range', hue='label', 
                palette={'correct': '#2ecc71', 'incorrect': '#e74c3c'}, s=100, alpha=0.7)
plt.title('Pemisahan Kelas: Upper Arm Std vs Shoulder Range')
plt.xlabel('Upper Arm Angle Standard Deviation')
plt.ylabel('Shoulder Angle Range')

# tambahkan garis batas keputusan sederhana secara visual (estimasi)
plt.axvline(x=10, color='gray', linestyle='--', label='Batas Threshold Clean (Std=10)')
plt.axhline(y=35, color='gray', linestyle=':', label='Batas Threshold Clean (Shoulder=35)')
plt.legend()
plt.savefig('eda_4_scatterplot.png')
plt.close()

# correlation heatmap
plt.figure(figsize=(8, 6))

# korelasi antara fitur dan label_encoded
corr_matrix = data[features + ['label_encoded']].corr()

# perbaikan: hapus 'pd.np' dan gunakan 'np' secara langsung agar tidak error
# buat mask agar heatmap tidak duplikat (bentuk segitiga)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, vmin=-1, vmax=1)
plt.title('Heatmap Korelasi Fitur Biomekanik')
plt.savefig('eda_5_correlation.png')
plt.close()

print("\nINFO: EDA Selesai! Semua grafik telah disimpan sebagai file PNG di folder saat ini.")