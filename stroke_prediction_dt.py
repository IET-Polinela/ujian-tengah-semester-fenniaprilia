
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd']

# 1. Load Dataset
print("1. Loading dan eksplorasi dataset...")
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Tampilkan informasi dasar dataset
print(f"\nJumlah data: {df.shape[0]}")
print(f"Jumlah fitur: {df.shape[1] - 1}")  # -1 karena 'stroke' adalah target
print("\nDistribusi target 'stroke':")
print(df['stroke'].value_counts())
print(f"Persentase kasus stroke: {df['stroke'].mean()*100:.2f}%")

# Statistik deskriptif untuk fitur numerik
print("\nStatistik deskriptif fitur numerik:")
print(df[['age', 'avg_glucose_level', 'bmi']].describe())

# 2. Data Preprocessing
print("\n2. Melakukan preprocessing data...")

# Hapus kolom ID yang tidak relevan
df.drop('id', axis=1, inplace=True)

# Menangani missing values
print(f"\nMissing values sebelum preprocessing:\n{df.isnull().sum()}")
df['bmi'].fillna(df['bmi'].median(), inplace=True)
print(f"\nMissing values setelah preprocessing:\n{df.isnull().sum()}")

# Encode fitur kategorikal
label_enc = LabelEncoder()
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_features:
    df[col] = label_enc.fit_transform(df[col])
    
# 3. Visualisasi Eksplorasi Data
print("\n3. Membuat visualisasi eksplorasi data...")

# Heatmap korelasi
plt.figure(figsize=(12, 10))
correlation = df.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm', mask=mask, linewidths=0.5)
plt.title('Heatmap Korelasi Antar Fitur', fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_korelasi.png')

# Visualisasi distribusi umur berdasarkan stroke
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='stroke', multiple='stack', palette=['#1f77b4', '#d62728'])
plt.title('Distribusi Umur berdasarkan Status Stroke', fontsize=16)
plt.xlabel('Umur', fontsize=14)
plt.ylabel('Jumlah Pasien', fontsize=14)
plt.savefig('distribusi_umur_stroke.png')

# Visualisasi distribusi glukosa berdasarkan stroke
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='avg_glucose_level', hue='stroke', multiple='stack', palette=['#1f77b4', '#d62728'])
plt.title('Distribusi Level Glukosa berdasarkan Status Stroke', fontsize=16)
plt.xlabel('Level Glukosa Rata-rata', fontsize=14)
plt.ylabel('Jumlah Pasien', fontsize=14)
plt.savefig('distribusi_glukosa_stroke.png')

# Analisis fitur kategorikal
plt.figure(figsize=(18, 15))

# Gender
plt.subplot(3, 2, 1)
sns.countplot(x='gender', hue='stroke', data=df, palette=['#1f77b4', '#d62728'])
plt.title('Distribusi Stroke berdasarkan Gender', fontsize=14)

# Hypertension
plt.subplot(3, 2, 2)
sns.countplot(x='hypertension', hue='stroke', data=df, palette=['#1f77b4', '#d62728'])
plt.title('Distribusi Stroke berdasarkan Hipertensi', fontsize=14)
plt.xticks([0, 1], ['Tidak', 'Ya'])

# Heart Disease
plt.subplot(3, 2, 3)
sns.countplot(x='heart_disease', hue='stroke', data=df, palette=['#1f77b4', '#d62728'])
plt.title('Distribusi Stroke berdasarkan Penyakit Jantung', fontsize=14)
plt.xticks([0, 1], ['Tidak', 'Ya'])

# Ever Married
plt.subplot(3, 2, 4)
sns.countplot(x='ever_married', hue='stroke', data=df, palette=['#1f77b4', '#d62728'])
plt.title('Distribusi Stroke berdasarkan Status Pernikahan', fontsize=14)

# Work Type
plt.subplot(3, 2, 5)
sns.countplot(x='work_type', hue='stroke', data=df, palette=['#1f77b4', '#d62728'])
plt.title('Distribusi Stroke berdasarkan Jenis Pekerjaan', fontsize=14)

# Smoking Status
plt.subplot(3, 2, 6)
sns.countplot(x='smoking_status', hue='stroke', data=df, palette=['#1f77b4', '#d62728'])
plt.title('Distribusi Stroke berdasarkan Status Merokok', fontsize=14)

plt.tight_layout()
plt.savefig('analisis_kategorikal.png')

# 4. Pemisahan fitur dan target
print("\n4. Memisahkan fitur dan target...")
X = df.drop('stroke', axis=1)
y = df['stroke']

# Normalisasi data numerik
scaler = StandardScaler()
X[['age', 'avg_glucose_level', 'bmi']] = scaler.fit_transform(X[['age', 'avg_glucose_level', 'bmi']])

# 5. Mengatasi ketidakseimbangan kelas dengan SMOTE
print("\n5. Mengatasi ketidakseimbangan kelas dengan SMOTE...")
print(f"Distribusi kelas sebelum SMOTE: {pd.Series(y).value_counts()}")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"Distribusi kelas setelah SMOTE: {pd.Series(y_res).value_counts()}")

# 6. Split data menjadi training dan testing
print("\n6. Membagi data menjadi training dan testing...")
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
print(f"Jumlah data training: {X_train.shape[0]}")
print(f"Jumlah data testing: {X_test.shape[0]}")

# 7. Model Selection dan Hyperparameter Tuning
print("\n7. Melakukan hyperparameter tuning untuk Random Forest...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"Parameter terbaik: {grid_search.best_params_}")
print(f"Akurasi cross-validation terbaik: {grid_search.best_score_:.4f}")

# 8. Training final model
print("\n8. Training model Random Forest dengan parameter optimal...")
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

# 9. Evaluasi Model
print("\n9. Evaluasi model...")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi: {accuracy:.4f}")

# 10. Visualisasi hasil model
print("\n10. Membuat visualisasi hasil model...")

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.savefig('confusion_matrix.png')

# ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

# Precision-Recall Curve
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16)
plt.savefig('precision_recall_curve.png')

# 11. Feature Importance
print("\n11. Analisis feature importance...")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance - Random Forest', fontsize=16)
plt.tight_layout()
plt.savefig('feature_importance.png')

print("\nProgram selesai! Output visualisasi telah disimpan.")
