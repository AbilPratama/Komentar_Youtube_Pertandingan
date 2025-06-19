import streamlit as st
import joblib
import os

# Load model dan vectorizer
model = joblib.load("model_random_forest.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Kata kunci komentar positif
positif_keywords = [
    "bagus", "keren", "mantap", "lucu", "menarik", "suka",
    "seru", "terbaik", "hebat", "asyik", "menyenangkan",
    "rekomendasi", "grafik bagus", "cerita bagus", "bagus banget"
]

# Fungsi untuk deteksi keyword positif
def keyword_sentiment(text):
    text_lower = text.lower()
    for keyword in positif_keywords:
        if keyword in text_lower:
            return "positif (berdasarkan keyword)"
    return None

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
