import streamlit as st
import re
import pickle
import nltk
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# ==============================
# Load Model & Tokenizer
# ==============================
MODEL_PATH = "fake_news_lstm.h5"
TOKENIZER_PATH = "tokenizer.pkl"

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

# ==============================
# Preprocessing Function
# ==============================
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

# ==============================
# Prediction Function
# ==============================
def predict_news(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0][0]
    label = "📰 Real News ✅" if prediction >= 0.5 else "📰 Fake News ❌"
    return label, prediction

# ==============================
# Fetch Recent News using NewsAPI
# ==============================
NEWS_API_KEY = "8ede8b70102f4b88b0d3346f31d23816"  # <-- Replace with your API key

def fetch_news(topic, max_articles=5):
    url = f"https://newsapi.org/v2/everything?q={topic}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    articles = data.get('articles', [])[:max_articles]
    return articles

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detection System")
st.write("Detect whether a news article is **Real** or **Fake**.")

# --- User input manually ---
user_input = st.text_area("Enter a news article/text:", height=200)
if st.button("🔍 Predict Text"):
    if user_input.strip() == "":
        st.warning("Please enter some news text to analyze.")
    else:
        label, confidence = predict_news(user_input)
        st.subheader("Prediction Result:")
        st.success(f"{label}")
        st.write(f"**Confidence Score:** {confidence:.4f}")

st.markdown("---")

