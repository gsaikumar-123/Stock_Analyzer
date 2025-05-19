import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(input_size=29, hidden_size=256, num_layers=2)
model.load_state_dict(torch.load('tesla_lstm_model.pth', map_location=torch.device('cpu')))
model.eval()

def get_sentiment_score(comment):
    sentiment = TextBlob(comment).sentiment.polarity
    return sentiment

def predict_next_day_price(date, comment, model):
    df = pd.read_csv('final_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df = df[df['Date'] <= pd.to_datetime(date)]  

    if len(df) < 50:
        return "Not enough historical data before this date."

    feature_columns = df.columns[1:]
    data = df[feature_columns].values  

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    sequence_length = 50
    last_sequence = data_scaled[-sequence_length:]

    sentiment_score = get_sentiment_score(comment) 
    last_sequence[:, -1] = sentiment_score  

    last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0)  # Shape: (1, 50, 29)

    model.eval()
    with torch.no_grad():
        predicted_price = model(last_sequence).item()

    last_feature_set = np.zeros((1, len(feature_columns)))  
    last_feature_set[0, 0] = predicted_price  
    predicted_price_actual = scaler.inverse_transform(last_feature_set)[:, 0][0]

    return predicted_price_actual


st.title("Stock Analyzer")

date_input = st.date_input("Enter Date:", min_value=datetime(2020, 1, 1), max_value=datetime(2021, 1, 1))
comment_input = st.text_area("Enter Your Comment on Tesla Stock:")

if st.button("Predict Next Day Price"):
    if not (datetime(2020, 1, 1).date() <= date_input <= datetime(2021, 1, 1).date()):
        st.error("Please select a date between January 1, 2020, and January 1, 2021.")
    else:
        next_day_price = predict_next_day_price(date_input, comment_input, model)
        st.success(f"Predicted Stock Price for {date_input + timedelta(days=1)}: ${next_day_price:.2f}")

