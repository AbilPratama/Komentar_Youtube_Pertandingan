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
    if user_input:
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("Klasifikasi: Sentimen Positif 😊")
        else:
            st.error("Klasifikasi: Sentimen Negatif 😞")
    else:
        st.warning("Silakan masukkan komentar terlebih dahulu.")
