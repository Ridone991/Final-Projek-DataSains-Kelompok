# --------------------------------------------
# FINAL PROJECT: PREDIKSI PENYAKIT KARDIOVASKULAR
# --------------------------------------------

# 1. Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

# 2. Load Dataset
# Gantilah path dengan file lokal kamu jika diperlukan
df = pd.read_csv("cardio_train.csv", sep=';')
df.drop(columns=['id'], inplace=True)
df['age'] = (df['age'] / 365).astype(int)

# 3. Split Fitur dan Target
X = df.drop('cardio', axis=1)
y = df['cardio']

# 4. Train-Test Split dan Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Training Model (Ganti ke RandomForestClassifier atau XGBClassifier)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 6. Evaluasi
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Akurasi:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))

# 7. Simpan Model dan Scaler
joblib.dump(model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 8. Streamlit App (Simpan bagian ini di file app.py jika terpisah)
import streamlit as st

st.title("Prediksi Risiko Penyakit Kardiovaskular")

# Load model dan scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Input pengguna
age = st.slider("Umur (tahun)", 30, 100, 50)
gender = st.radio("Jenis Kelamin", ["Perempuan", "Laki-laki"])
height = st.number_input("Tinggi badan (cm)", 120, 220, 160)
weight = st.number_input("Berat badan (kg)", 30, 200, 70)
ap_hi = st.number_input("Tekanan darah sistolik (ap_hi)", 90, 200, 120)
ap_lo = st.number_input("Tekanan darah diastolik (ap_lo)", 60, 150, 80)
chol = st.selectbox("Kolesterol", [1, 2, 3])
gluc = st.selectbox("Glukosa", [1, 2, 3])
smoke = st.radio("Merokok?", ["Tidak", "Ya"])
alco = st.radio("Konsumsi alkohol?", ["Tidak", "Ya"])
active = st.radio("Aktif secara fisik?", ["Tidak", "Ya"])

gender_val = 2 if gender == "Laki-laki" else 1
smoke_val = 1 if smoke == "Ya" else 0
alco_val = 1 if alco == "Ya" else 0
active_val = 1 if active == "Ya" else 0

input_data = np.array([[age, gender_val, height, weight, ap_hi, ap_lo,
                        chol, gluc, smoke_val, alco_val, active_val]])

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

if st.button("Prediksi"):
    if prediction[0] == 1:
        st.error("❗ Pasien kemungkinan TERDIAGNOSIS penyakit kardiovaskular.")
    else:
        st.success("✅ Pasien TIDAK terdiagnosis penyakit kardiovaskular.")
