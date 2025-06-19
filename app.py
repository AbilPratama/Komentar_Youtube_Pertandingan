import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load("model_random_forest.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# UI Streamlit
st.set_page_config(page_title="Klasifikasi Komentar Youtube", layout="centered")
st.title("📚 Klasifikasi Sentimen Komentar Youtube")
st.write("Masukkan komentar dari pengguna")

# Input teks pengguna
user_input = st.text_area("Komentar pengguna:", height=150)

# Tombol klasifikasi
if st.button("Klasifikasikan"):
    if not user_input.strip():
        st.warning("⚠️ Komentar tidak boleh kosong.")
    else:
        # Transform dan prediksi
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        # Tampilkan hasil klasifikasi
        st.success(f"💬 Hasil klasifikasi: **{prediction.upper()}**")
