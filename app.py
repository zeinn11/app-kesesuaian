import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_excel("tracer_studi.xlsx")
    return df

df = load_data()

# ======================
# PREPROCESS
# ======================
def convert_income(x):
    if pd.isna(x):
        return np.nan
    x = str(x).lower()
    if "1-3" in x:
        return 2000000
    if "3-5" in x:
        return 4000000
    if ">5" in x:
        return 6000000
    return np.nan

df["Pendapatan_num"] = df["Pendapatan_bersih"].apply(convert_income)

df_model = df[
    ["Pendapatan_num", "ruang_lingkup_kerja", "Jenis_pekerjaan",
     "Lama_tunggu_kerja", "Kesesuaian_bidang"]
].dropna()

X = df_model.drop("Kesesuaian_bidang", axis=1)
y = df_model["Kesesuaian_bidang"]

# ======================
# TRAIN MODEL
# ======================
num_features = ["Pendapatan_num"]
cat_features = ["ruang_lingkup_kerja", "Jenis_pekerjaan", "Lama_tunggu_kerja"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

model = ImbPipeline([
    ("prep", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

model.fit(X, y)

# ======================
# UI
# ======================
st.title("📊 Sistem Analisis Kesesuaian Bidang Alumni")

menu = st.sidebar.selectbox("Menu", ["Ringkasan Data", "Prediksi"])

# ======================
# PAGE 1
# ======================
if menu == "Ringkasan Data":
    st.subheader("Ringkasan Dataset")
    st.write("Jumlah data:", df_model.shape[0])
    st.write(df_model.head())

    st.subheader("Distribusi Kesesuaian Bidang")
    st.bar_chart(df_model["Kesesuaian_bidang"].value_counts())

# ======================
# PAGE 2
# ======================
if menu == "Prediksi":
    st.subheader("Prediksi Kesesuaian Bidang")

    pendapatan = st.number_input("Pendapatan (Rp)", 1000000, 20000000, step=500000)
    ruang = st.selectbox("Ruang Lingkup Kerja", df["ruang_lingkup_kerja"].dropna().unique())
    jenis = st.selectbox("Jenis Pekerjaan", df["Jenis_pekerjaan"].dropna().unique())
    tunggu = st.selectbox("Lama Tunggu Kerja", df["Lama_tunggu_kerja"].dropna().unique())

    if st.button("Prediksi"):
        input_df = pd.DataFrame([{
            "Pendapatan_num": pendapatan,
            "ruang_lingkup_kerja": ruang,
            "Jenis_pekerjaan": jenis,
            "Lama_tunggu_kerja": tunggu
        }])

        hasil = model.predict(input_df)[0]
        st.success(f"Hasil Prediksi: **{hasil}**")
