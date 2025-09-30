# 📰 Fake News Detection System

Detect Real vs Fake News in real-time using an LSTM-based NLP model and Streamlit! ⚡
This project combines deep learning, NLP preprocessing, and live news fetching to classify news articles. Users can either input text manually or fetch the latest news dynamically using NewsAPI.

## 🚀 Live Demo

Check out the **interactive demo** of the Fake News Detection System here:  

[Live Demo 📰](https://linda-lance-fake-news-detection-system-app-891hnt.streamlit.app/)

## ✨ Project Highlights

📝 Manual Text Prediction: Input any news text to check if it’s real or fake.

🌐 Dynamic News Fetching: Fetch the latest news articles on any topic from NewsAPI.

📊 Confidence Scores: Get probability scores for each prediction.

💻 Recruiter-Friendly UI: Clean and interactive Streamlit interface.

🔄 Persistent Model: Pretrained LSTM model with saved tokenizer for quick predictions.

## 🛠️ Tech Stack

• Python 3.x

• TensorFlow, Keras (LSTM)

• NLTK (stopwords, lemmatization)

• Streamlit

• Pandas, NumPy

• NewsAPI

## 📂 Project Structure
```
fake-news-detection/
│
├── app.py                  # Streamlit app
├── fake_news_lstm.h5       # Trained LSTM model
├── tokenizer.pkl           # Tokenizer for text sequences
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── Screenshots/            # Optional images/icons
├── fake_news.ipynb         # source code
└── training dataset        # Optional CSV for testing
```
## ⚡ Installation

Clone the repository
```
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

Install dependencies
```
pip install -r requirements.txt
```

Add NewsAPI key

Open app.py and replace:
```
NEWS_API_KEY = "YOUR_NEWSAPI_KEY"  # 🔑 Replace with your key
```

🚀 Running the App
```
streamlit run app.py
```

Open the URL shown in the terminal to access the app in your browser.

## 📰 How to Use

1️⃣ Manual Text Prediction

• Enter a news article in the text area.

• Click Predict Text to see the classification and confidence score.

## 💡 Model Architecture

- Embedding Layer: Converts words to dense vectors

- LSTM Layer: Sequential layer for capturing text dependencies

- Dense Layers: Fully connected layers with ReLU & Sigmoid activation

- Dropout Layers: Prevent overfitting

- Accuracy: Typically >90% on test data (dataset-dependent) ✅

## 📊 Result

- User Interface

<img width="853" height="494" alt="user_interface" src="https://github.com/user-attachments/assets/77310ef7-7270-4c5c-b7a4-7941f4837638" />

- Output

<img width="1001" height="594" alt="result" src="https://github.com/user-attachments/assets/0850b02e-9767-4d96-a5c7-9fa7acab1f23" />


## 🔮 Future Enhancements

🔹 Upgrade to Bidirectional LSTM + GloVe embeddings for better context understanding 🌐

🔹 Integrate Transformer-based models (BERT, RoBERTa) for state-of-the-art results 🚀

🔹 Multi-language support 🌍

🔹 Deploy on Heroku/AWS for global access ☁️

🔹 Periodic retraining with fresh news from NewsAPI 🔄


## 📫 Contact

[GitHub](https://github.com/Linda-Lance)
 
[LinkedIn](https://www.linkedin.com/in/linda--lance/)
 
[MailID](lindalance2210@gmail.com)
