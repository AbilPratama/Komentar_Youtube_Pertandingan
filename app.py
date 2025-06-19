import streamlit as st
import joblib
import re
import string

# =====================
# Fungsi Preprocessing
# =====================
def preprocess(text):
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+", "", text)  # hapus URL
    text = re.sub(r"\d+", "", text)  # hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # hapus tanda baca
    text = text.strip()  # hapus spasi depan-belakang
    return text

# =====================
# Load Model dan Vectorizer
# =====================
model = joblib.load("model_random_forest.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# =====================
# UI Streamlit
# =====================
st.set_page_config(page_title="Klasifikasi Komentar YouTube", layout="centered")
st.title("📺 Klasifikasi Sentimen Komentar YouTube")
st.write("Masukkan komentar dari pengguna dan sistem akan memprediksi apakah sentimennya **positif** atau **negatif**.")

# =====================
# Input Pengguna
# =====================
user_input = st.text_area("Komentar pengguna:", height=150)

# =====================
# Tombol Prediksi
# =====================
if st.button("Klasifikasikan"):
    if user_input.strip():
        cleaned = preprocess(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        # Menampilkan hasil prediksi
        if prediction == 1 or prediction == "positif":
            st.success("💬 Klasifikasi: Sentimen **Positif** 😊")
        else:
            st.error("💬 Klasifikasi: Sentimen **Negatif** 😞")
    else:
        st.warning("⚠️ Silakan masukkan komentar terlebih dahulu.")
