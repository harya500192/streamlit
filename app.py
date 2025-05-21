import streamlit as st
import pandas as pd
from joblib import load

# Load model dan vectorizer
model = load("model_sentimen.joblib")
vectorizer = load("vectorizer_sentimen.joblib")

st.title("Klasifikasi Sentimen Review Genshin Impact")
st.write("Masukkan review pengguna lalu klik tombol *Prediksi* untuk melihat sentimennya.")

# Input teks dari pengguna
user_input = st.text_area("Masukkan review di sini", "")

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Harap masukkan review terlebih dahulu.")
    else:
        # Preprocessing & transformasi teks
        X = vectorizer.transform([user_input])
        prediksi = model.predict(X)[0]

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        st.success(f"Sentimen review ini adalah: **{prediksi.upper()}**")

# Tambahan opsional: unggah file CSV dan klasifikasikan semua baris
st.markdown("---")
st.subheader("Atau unggah file CSV berisi review")

uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "content" not in df.columns:
        st.error("Kolom 'content' tidak ditemukan di file CSV.")
    else:
        df['content'] = df['content'].fillna("")
        X = vectorizer.transform(df['content'])
        df['prediksi_sentimen'] = model.predict(X)
        st.write("Hasil prediksi:")
        st.dataframe(df[['content', 'prediksi_sentimen']].head())

        # Unduh hasil
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download hasil sebagai CSV",
            data=csv,
            file_name='hasil_prediksi_review.csv',
            mime='text/csv',
        )
