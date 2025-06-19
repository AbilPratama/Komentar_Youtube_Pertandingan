import streamlit as st
import joblib
import os

# Load model dan vectorizer
model = joblib.load("model_random_forest.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# UI aplikasi Streamlit
st.set_page_config(page_title="Klasifikasi Komentar Webtoon", layout="centered")
st.title("📚 Klasifikasi Sentimen Komentar Webtoon")
st.write("Masukkan komentar dari pengguna Webtoon untuk diklasifikasikan sebagai **positif** atau **negatif**.")

# Input dari pengguna
user_input = st.text_area("Komentar pengguna:", height=150)

# Tombol klasifikasi
if st.button("Klasifikasikan"):
    if not user_input.strip():
        st.warning("⚠️ Komentar tidak boleh kosong.")
    else:
        # Cek berdasarkan keyword terlebih dahulu
        keyword_result = keyword_sentiment(user_input)
        if keyword_result:
            st.success(f"Hasil klasifikasi: {keyword_result}")
        else:
            # Klasifikasi menggunakan model machine learning
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            st.success(f"Hasil klasifikasi model: {prediction}")
