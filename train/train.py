# Import Library
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
from nltk.corpus import stopwords
import nltk

# Download stopwords (hanya perlu dilakukan sekali)
nltk.download('stopwords')

# Stopwords untuk Bahasa Indonesia dan Bahasa Inggris
stop_words_id = set(stopwords.words('indonesian'))
stop_words_en = set(stopwords.words('english'))

# Fungsi untuk membersihkan teks
def clean_text(text, language='en'):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = stop_words_id if language == 'id' else stop_words_en
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Fungsi untuk melatih model
def train_model():
    # Load dataset gabungan
    dataset_path = './train/data/Combined_Dataset.csv'  # Path ke dataset gabungan
    data = pd.read_csv(dataset_path)

    # Bersihkan data kosong
    data.dropna(subset=['text', 'label'], inplace=True)

    # Bersihkan teks berdasarkan bahasa (asumsi: Bahasa Indonesia atau Inggris)
    data['cleaned_text'] = data.apply(
        lambda row: clean_text(row['text'], 'id') if 'indonesian' in row['text'].lower() else clean_text(row['text'], 'en'),
        axis=1
    )

    # Split data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_text'], data['label'], test_size=0.25, random_state=42
    )

    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Model Passive Aggressive Classifier
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_tfidf, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test_tfidf)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

    # Simpan model dan vectorizer
    joblib.dump(model, './models/model.sav')
    joblib.dump(tfidf_vectorizer, './models/vectorizer.sav')
    print("Model dan vectorizer berhasil disimpan.")

# Main Function
if __name__ == "__main__":
    train_model()
