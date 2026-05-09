# import library
import pickle
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import xgboost as xgb
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load dataset clean
file_path = 'dataset/data_biceps_clean.csv'

# try except
try:
    data = pd.read_csv(file_path)
    print(f"Total data awal: {len(data)}")
except FileNotFoundError:
    print(f"ERROR: File {file_path} tidak ditemukan. Pastikan path benar.")
    exit()
    
# pisahkan fitur dan target
# buang semua kolom string/metadata agar tidak merusak perhitungan XGBoost
drop_cols = ['sample_id', 'exercise', 'active_arm', 'label', 'notes']
X = data.drop(columns=drop_cols)

# mapping label
label_mapping = {'correct': 0, 'incorrect': 1}
y = data['label'].map(label_mapping)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# setup dengan xgboost
model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=3,
    learning_rate=0.1, 
    random_state=42,
    objective='binary:logistic',
    eval_metric='logloss'
)

# validasi kekuatan model dengan 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')

print(f"skor akurasi per fold : {cv_scores}")
print(f"rata-rata akurasi     : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 200:.2f}%)")

# training model
model.fit(X_train, y_train)

# testing model
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n=> Akurasi Final pada Data Testing: {test_acc * 100:.2f}% <=")

# define target names agar output text mudah dibaca manusia
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['correct', 'incorrect']))

# visualisasi confusion matrix
confusMat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
# perbaikan typo cmap dan nama palet
sns.heatmap(confusMat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['correct', 'incorrect'], 
            yticklabels=['correct', 'incorrect'])
plt.title('Confusion Matrix Evaluasi Biceps')
plt.ylabel('Label Asli (Actual)')
plt.xlabel('Tebakan Model (Predicted)')
plt.tight_layout()
plt.savefig('models/xgboost_confusion_matrix.png')
plt.close()

# simpan model untuk inferensi realtime
model.save_model('models/xgboost_biceps_model.json')

# backup pakai pickle
with open('models/biceps_xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("\nModel berhasil disimpan pada 'models/xgboost_biceps_model.json' dan 'models/biceps_xgboost_model.pkl'")