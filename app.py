import streamlit as st
import pandas as pd
import joblib

# ==========================================
# LOAD MODEL
# ==========================================

data = joblib.load("random_forest_tuning.joblib")

model = data["model"]
encoder = data["encoder"]
num_cols = data["num_cols"]
cat_cols = data["cat_cols"]
prediction = model.predict(final_data)[0]
# ==========================================
# JUDUL WEBSITE
# ==========================================

st.set_page_config(page_title="Prediksi Kanker Paru")

st.title("Prediksi Kanker Paru")
st.write("Implementasi Algoritma Random Forest untuk Klasifikasi Kanker Paru")

st.write("Silakan masukkan data pasien:")

# ==========================================
# INPUT USER (BAHASA INDONESIA)
# ==========================================

gender = st.selectbox(
    "Jenis Kelamin",
    options=["M", "F"],
    format_func=lambda x: "Laki-laki" if x == "M" else "Perempuan"
)

age = st.number_input(
    "Umur",
    min_value=1,
    max_value=120,
    value=30
)

smoking = st.selectbox(
    "Merokok",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

yellow_fingers = st.selectbox(
    "Jari Menguning",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

anxiety = st.selectbox(
    "Kecemasan",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

peer_pressure = st.selectbox(
    "Tekanan Sosial",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

chronic_disease = st.selectbox(
    "Penyakit Kronis",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

fatigue = st.selectbox(
    "Kelelahan",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

allergy = st.selectbox(
    "Alergi",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

wheezing = st.selectbox(
    "Mengi",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

alcohol_consuming = st.selectbox(
    "Konsumsi Alkohol",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

coughing = st.selectbox(
    "Batuk",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

shortness_of_breath = st.selectbox(
    "Sesak Napas",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

swallowing_difficulty = st.selectbox(
    "Sulit Menelan",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

chest_pain = st.selectbox(
    "Nyeri Dada",
    options=[0, 1],
    format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

# ==========================================
# MEMBUAT DATAFRAME INPUT
# NAMA KOLOM HARUS SAMA DENGAN DATASET
# ==========================================

input_data = pd.DataFrame({
    "GENDER": [gender],
    "AGE": [age],
    "SMOKING": [smoking],
    "YELLOW_FINGERS": [yellow_fingers],
    "ANXIETY": [anxiety],
    "PEER_PRESSURE": [peer_pressure],
    "CHRONIC DISEASE": [chronic_disease],
    "FATIGUE ": [fatigue],
    "ALLERGY ": [allergy],
    "WHEEZING": [wheezing],
    "ALCOHOL CONSUMING": [alcohol_consuming],
    "COUGHING": [coughing],
    "SHORTNESS OF BREATH": [shortness_of_breath],
    "SWALLOWING DIFFICULTY": [swallowing_difficulty],
    "CHEST PAIN": [chest_pain]
})

# ==========================================
# PREPROCESSING INPUT
# ==========================================

encoded_cat = encoder.transform(input_data[cat_cols])

encoded_cat_df = pd.DataFrame(
    encoded_cat,
    columns=encoder.get_feature_names_out(cat_cols)
)

final_data = pd.concat(
    [
        input_data[num_cols].reset_index(drop=True),
        encoded_cat_df.reset_index(drop=True)
    ],
    axis=1
)

# ==========================================
# PREDIKSI
# ==========================================

if st.button("Prediksi"):

    prediction = model.predict(final_data)[0]

    if prediction == "YES":
        st.error("Pasien Terindikasi Kanker Paru")
    else:
        st.success("Pasien Tidak Terindikasi Kanker Paru")

st.write(prediction)
