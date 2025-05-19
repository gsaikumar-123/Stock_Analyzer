# 📈 Stock Analyzer

**Stock Analyzer** is an AI-powered web application that predicts Tesla’s stock closing price for the next trading day based on historical market data and sentiment analysis of user comments. It combines financial time-series modeling using an LSTM neural network with real-time sentiment insights derived from natural language processing. Built with **Streamlit**, this project offers an intuitive and interactive interface for financial enthusiasts, data scientists, and retail investors.

---

## 🚀 Features

* **📅 Historical Contextualization**: Uses preprocessed Tesla stock data including open, close, high, low, and technical indicators.
* **💬 Sentiment Analysis**: Analyzes user-submitted textual comments using `TextBlob` to determine sentiment polarity, influencing the stock price forecast.
* **🔁 Multi-Comment Aggregation**: Supports multiple user comments, calculates an average sentiment score, and combines it with historical scores for accurate predictions.
* **📉 LSTM Model Prediction**: Trained deep learning model (LSTM) built using PyTorch to capture long-term dependencies in financial time series.
* **📊 Scaled Prediction Output**: Normalizes and denormalizes features using MinMaxScaler to maintain prediction consistency.
* **📤 Real-Time Forecast**: Allows users to input any historical date (within the supported range) and get a next-day predicted closing price based on sentiment-influenced data.
* **🧠 Sentiment Fusion**: Smart averaging of daily historical sentiment score and input comment score to ensure realism and continuity in model input.

---

## 📁 Project Structure

```bash
Stock_Analyzer/
├── app.py                         # Main Streamlit interface for prediction
├── final_data.csv                 # Final stock dataset with technical indicators and sentiment column
├── tesla_lstm_model.pth           # Pretrained PyTorch model weights
├── cleaned-tsla-tweets.csv        # Cleaned Tesla tweet data used for sentiment aggregation
├── daily-avg-sentiment-scores.csv # Daily average sentiment scores computed from tweets
├── final_prediction.ipynb         # Prediction and forecasting notebook
├── sentimental_analysis.ipynb     # Notebook for analyzing and scoring tweet sentiment
├── Preprocessing.ipynb            # Data preprocessing and feature engineering
├── tsla-tweets.csv                # Raw Twitter data related to Tesla
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🧰 Technology Stack

| Layer            | Tools / Libraries              |
| ---------------- | ------------------------------ |
| Frontend         | Streamlit                      |
| Backend          | Python                         |
| Machine Learning | PyTorch (LSTM)                 |
| NLP              | TextBlob                       |
| Data Handling    | Pandas, NumPy                  |
| Visualization    | Streamlit Components           |
| Scaling          | MinMaxScaler from scikit-learn |

---

## 📈 How It Works

1. **Data Collection**: Collect historical Tesla stock data and tweets from reliable financial sources.
2. **Preprocessing**:

   * Normalize price and volume data.
   * Clean tweets (remove emojis, links, usernames).
   * Score sentiment using TextBlob.
3. **Model Input**:

   * Prepare sequences of the last 50 days of features.
   * Merge average daily sentiment from tweets with current user comment(s).
4. **Model Architecture**:

   * Input size: 29 features
   * Layers: 2-layer LSTM with hidden size 256
   * Output: 1-day ahead closing price (denormalized)
5. **User Interaction**:

   * User selects a historical date and enters multiple sentiment-based comments.
   * App calculates polarity scores and averages them with historical sentiment.
   * LSTM predicts the stock price for the following day.

---

## ⚙️ Installation Guide

To run this project locally, follow these steps:

1. **Clone the Repository**:

```bash
git clone https://github.com/gsaikumar-123/Stock_Analyzer.git
cd Stock_Analyzer
```

2. **Set Up a Virtual Environment** (Optional but recommended):

```bash
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. **Install Required Packages**:

```bash
pip install -r requirements.txt
```

4. **Launch the Streamlit Application**:

```bash
streamlit run app.py
```

---

## 🧪 Example Use Cases

| Scenario                             | Input                                            | Output                                      |
| ------------------------------------ | ------------------------------------------------ | ------------------------------------------- |
| Analyst simulating bullish sentiment | “Tesla is the future. I'm extremely optimistic.” | Predicted next-day price increases slightly |
| Bearish market mood                  | “Tesla is overvalued. Lots of competition now.”  | Predicted next-day price may decrease       |
| Neutral event                        | No comment or mixed neutral comments             | Model uses historical sentiment only        |

---

## 📝 How to Use

1. Launch the app.
2. Choose a date between Jan 1, 2020 and Jan 1, 2021.
3. Type one or multiple comments in the text area. Separate them with new lines.
4. Click "Predict Next Day Price."
5. The predicted stock closing price for the next day will be shown based on:

   * LSTM historical data
   * Combined sentiment from history and user input

---

## 🧠 Behind the Model

The LSTM model is built to forecast time-dependent financial data by remembering long sequences of trends. It is trained on Tesla's stock prices, technical indicators, and sentiment data over time. The sentiment score acts as a proxy for market psychology, which often affects short-term price movements.

---

## 🔮 Future Enhancements

* 🔗 Integrate real-time tweet sentiment via Twitter API (Tweepy)
* 📈 Add support for other stock tickers (Apple, Microsoft, etc.)
* 🧠 Upgrade sentiment scoring using transformer-based models like BERT or RoBERTa
* 🌐 Deploy on public cloud (e.g., Streamlit Cloud, Heroku, or AWS)
* 💹 Add historical price and volume visualizations with Plotly