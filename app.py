import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from bs4 import BeautifulSoup
import requests
from nltk.corpus import stopwords

# Download stopwords (hanya perlu dilakukan sekali)
nltk.download('stopwords')

# Stopwords
stop_words_id = set(stopwords.words('indonesian'))
stop_words_en = set(stopwords.words('english'))

# Fungsi untuk membersihkan teks
def clean_text(text, language='id'):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = stop_words_id if language == 'id' else stop_words_en
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Fungsi untuk mengambil teks dari link
def extract_text_from_link(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Cari elemen artikel atau konten utama
        main_content = soup.find('article') or soup.find('div', class_='main-content')
        if main_content:
            paragraphs = main_content.find_all('p')
        else:
            paragraphs = soup.find_all('p')
        
        # Gabungkan teks dari paragraf
        text = ' '.join([para.get_text() for para in paragraphs])
        return text.strip()
    except Exception as e:
        st.error(f"Error saat mengambil konten dari link: {e}")
        return ""

# Fungsi untuk memuat model
def load_model():
    model = joblib.load('./models/model.sav')
    vectorizer = joblib.load('./models/vectorizer.sav')
    return model, vectorizer

# Streamlit Antarmuka
st.title("Hoax News Detection")
st.write("Deteksi apakah sebuah berita adalah **Fake** atau **Real** berdasarkan teks atau link artikel.")

# Pemilihan Bahasa
st.subheader("Pilih Bahasa")
language = st.radio("Bahasa yang digunakan:", ["Bahasa Indonesia", "English"])
language_code = 'id' if language == "Bahasa Indonesia" else 'en'

# Memuat model dan vectorizer
model, vectorizer = load_model()

# Menu Input: Teks atau Link
st.subheader("Input Berita")
input_type = st.radio("Pilih metode input:", ["Teks", "Link"])

if input_type == "Teks":
    # Input teks berita
    news_text = st.text_area("Masukkan teks berita:")
    if st.button("Prediksi"):
        if news_text.strip():
            # Bersihkan teks sesuai bahasa
            cleaned_text = clean_text(news_text, language_code)
            text_tfidf = vectorizer.transform([cleaned_text])
            prediction = model.predict(text_tfidf)
            result = "Fake" if prediction[0] == 1 else "Real"
            st.subheader(f"Hasil Prediksi: {result}")
        else:
            st.error("Teks berita tidak boleh kosong.")

elif input_type == "Link":
    # Input link berita
    news_link = st.text_input("Masukkan link berita:")
    if st.button("Prediksi dari Link"):
        if news_link.strip():
            # Ambil dan bersihkan teks dari link
            news_content = extract_text_from_link(news_link)
            if news_content:
                cleaned_text = clean_text(news_content, language_code)
                text_tfidf = vectorizer.transform([cleaned_text])
                prediction = model.predict(text_tfidf)
                result = "Fake" if prediction[0] == 1 else "Real"
                st.subheader(f"Hasil Prediksi: {result}")
                st.write("**Konten yang diekstrak dari link:**")
                st.write(news_content[:500] + "...")
        else:
            st.error("Link berita tidak boleh kosong.")
