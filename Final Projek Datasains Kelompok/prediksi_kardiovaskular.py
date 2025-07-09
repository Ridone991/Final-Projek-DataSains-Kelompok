import joblib
import numpy as np
import pemodelan as st
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
