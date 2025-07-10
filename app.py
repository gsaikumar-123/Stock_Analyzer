import streamlit as st
import pandas as pd
import numpy as np
import warnings
import pickle
import os
from datetime import datetime, timedelta

# Suppress all warnings
warnings.filterwarnings('ignore')

# Import PyTorch with error suppression
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError as e:
    st.error(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False

# Import other ML libraries
try:
    from sklearn.preprocessing import MinMaxScaler
    from textblob import TextBlob
    ML_AVAILABLE = True
except ImportError as e:
    st.error(f"ML libraries not available: {e}")
    ML_AVAILABLE = False

# Streamlit configuration will be handled by config.toml

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

# Load the trained model
@st.cache_resource
def load_model():
    model = LSTM(input_size=29, hidden_size=256, num_layers=2)
    model.load_state_dict(torch.load('tesla_lstm_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the scaler
@st.cache_data
def load_scaler():
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    return None

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('final_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Load sentiment data
    sentiment_df = pd.read_csv('daily-avg-sentiment-scores.csv')
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Merge sentiment data
    df = df.merge(sentiment_df[['date', 'sentiment_score_final']], 
                 left_on='Date', right_on='date', how='left')
    df = df.drop('date', axis=1)
    
    # Fill missing sentiment scores with 0
    df['sentiment_score_final'] = df['sentiment_score_final'].fillna(0)
    
    return df

def get_average_sentiment(comments_text):
    """Calculate average sentiment from user comments"""
    if not comments_text or comments_text.strip() == "":
        return 0.0
    
    comments = [c.strip() for c in comments_text.split('\n') if c.strip()]
    if not comments:
        return 0.0
    
    scores = [TextBlob(comment).sentiment.polarity for comment in comments]
    return sum(scores) / len(scores)

def prepare_features(df, target_date):
    """Prepare features for prediction"""
    feature_columns = ['Close', 'Open', 'High', 'Low', 'Volume', 'sentiment_score_final',
                      'daily_return', 'log_return', 'SMA_short', 'SMA_long', 'EMA_short', 'EMA_long',
                      'volatility', 'ATR', 'RSI', 'MACD', 'bollinger_h', 'bollinger_l',
                      'momentum', 'ROC', 'Day_of_week', 'Month', 'Quarter',
                      'lagged_return', 'lagged_close', 'volume_change', 'volume_sma',
                      'price_to_sma', 'high_low_range']
    
    # Convert target_date to pandas datetime if it's not already
    if not isinstance(target_date, pd.Timestamp):
        target_date = pd.to_datetime(target_date)
    
    # Check if target_date is within the model's valid date range
    model_start_date = pd.to_datetime('2020-01-01')
    model_end_date = pd.to_datetime('2021-01-01')
    
    if target_date < model_start_date or target_date >= model_end_date:
        return None, None, f"Date {target_date.strftime('%Y-%m-%d')} is outside the model's valid range (2020-01-01 to 2021-01-01)."
    
    # Filter data up to target date
    df_filtered = df[df['Date'] <= target_date].copy()
    
    if len(df_filtered) < 50:
        return None, None, "Not enough historical data before this date."
    
    # Get the last 50 days of data
    df_filtered = df_filtered.tail(50)
    
    # Prepare features
    features = df_filtered[feature_columns].values
    
    return features, feature_columns, None

def predict_next_day_price(date, comment, model, df, scaler):
    """Predict the next day's stock price"""
    
    # Prepare features
    features, feature_columns, error = prepare_features(df, date)
    if error:
        return error
    
    if features is None or feature_columns is None:
        return "Error preparing features for prediction."
    
    # Calculate sentiment from user comments
    user_sentiment = get_average_sentiment(comment)
    
    # Update the sentiment score in the last row
    features[-1, 5] = user_sentiment  # sentiment_score_final is at index 5
    
    # Scale the features
    if scaler is None:
        # If no saved scaler, create one (this should not happen in production)
        temp_scaler = MinMaxScaler()
        features_scaled = temp_scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)
    
    # Prepare input for model (add batch dimension)
    features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        predicted_scaled = model(features_tensor).item()
    
    # Inverse transform to get actual price
    if scaler is None:
        # Create dummy array for inverse transform
        dummy_array = np.zeros((1, len(feature_columns)))
        dummy_array[0, 0] = predicted_scaled
        predicted_price = temp_scaler.inverse_transform(dummy_array)[0, 0]
    else:
        # Create dummy array for inverse transform
        dummy_array = np.zeros((1, len(feature_columns)))
        dummy_array[0, 0] = predicted_scaled
        predicted_price = scaler.inverse_transform(dummy_array)[0, 0]
    
    return predicted_price

def create_scaler_and_save(df):
    """Create and save the scaler for consistent scaling"""
    feature_columns = ['Close', 'Open', 'High', 'Low', 'Volume', 'sentiment_score_final',
                      'daily_return', 'log_return', 'SMA_short', 'SMA_long', 'EMA_short', 'EMA_long',
                      'volatility', 'ATR', 'RSI', 'MACD', 'bollinger_h', 'bollinger_l',
                      'momentum', 'ROC', 'Day_of_week', 'Month', 'Quarter',
                      'lagged_return', 'lagged_close', 'volume_change', 'volume_sma',
                      'price_to_sma', 'high_low_range']
    
    try:
        scaler = MinMaxScaler()
        scaler.fit(df[feature_columns])
        
        # Save the scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        return scaler
    except Exception as e:
        st.error(f"Error creating scaler: {str(e)}")
        return None

# Main Streamlit app
def main():
    # Check if all required libraries are available
    if not TORCH_AVAILABLE or not ML_AVAILABLE:
        st.error("❌ Required libraries are not available. Please install all dependencies.")
        st.info("Run: pip install -r requirements.txt")
        return
    
    st.set_page_config(
        page_title="Stock Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Stock Price Analyzer")
    st.markdown("---")
    
    # Load data and model
    try:
        df = load_data()
        model = load_model()
        scaler = load_scaler()
        
        # Create scaler if it doesn't exist
        if scaler is None:
            scaler = create_scaler_and_save(df)
            if scaler is not None:
                st.success("Model scaler created successfully!")
            else:
                st.error("Failed to create scaler. Please check your data files.")
                return
        
    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        st.info("Please ensure all required files (final_data.csv, daily-avg-sentiment-scores.csv, tesla_lstm_model.pth) are in the current directory.")
        return
    
    # Sidebar for inputs
    st.sidebar.header("📊 Prediction Settings")
    
    # Define the restricted date range for the model
    model_start_date = datetime(2020, 1, 1).date()
    model_end_date = datetime(2021, 1, 1).date()
    
    # Show model date range information
    st.sidebar.info(f"📅 **Model Date Range**: {model_start_date.strftime('%Y-%m-%d')} to {model_end_date.strftime('%Y-%m-%d')}")
    st.sidebar.warning("⚠️ **Note**: This model only works for predictions within the specified date range.")
    
    # Date input with restricted range
    selected_date = st.sidebar.date_input(
        "Select Date for Prediction:",
        value=model_end_date - timedelta(days=1),
        min_value=model_start_date,
        max_value=model_end_date - timedelta(days=1)
    )
    
    # Comments input
    st.sidebar.subheader("💬 Market Sentiment")
    comments = st.sidebar.text_area(
        "Enter your market analysis or comments (one per line):",
        placeholder="Example:\nPositive news about the company\nStrong quarterly results\nMarket volatility expected",
        height=150
    )
    
    # Prediction button
    if st.sidebar.button("🚀 Predict Next Day Price", type="primary"):
        if selected_date:
            with st.spinner("Analyzing market data and generating prediction..."):
                try:
                    predicted_price = predict_next_day_price(
                        selected_date, comments, model, df, scaler
                    )
                    
                    if isinstance(predicted_price, str):
                        st.error(predicted_price)
                    else:
                        # Get current price for comparison
                        selected_date_pd = pd.to_datetime(selected_date)
                        current_data = df[df['Date'].dt.date == selected_date_pd.date()]
                        if not current_data.empty:
                            current_price = float(current_data['Close'].iloc[0])
                            price_change = predicted_price - current_price
                            price_change_pct = (price_change / current_price) * 100
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    label="Current Price",
                                    value=f"${current_price:.2f}",
                                    delta=None
                                )
                            
                            with col2:
                                st.metric(
                                    label="Predicted Price",
                                    value=f"${predicted_price:.2f}",
                                    delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                                )
                            
                            with col3:
                                next_day = selected_date + timedelta(days=1)
                                st.metric(
                                    label="Prediction Date",
                                    value=next_day.strftime("%Y-%m-%d"),
                                    delta=None
                                )
                            
                            # Additional insights
                            st.subheader("📊 Market Insights")
                            
                            if price_change > 0:
                                st.success(f"📈 **Bullish Prediction**: Expected increase of ${price_change:.2f} ({price_change_pct:.2f}%)")
                            else:
                                st.warning(f"📉 **Bearish Prediction**: Expected decrease of ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)")
                            
                            # Sentiment analysis
                            if comments.strip():
                                sentiment_score = get_average_sentiment(comments)
                                st.info(f"💭 **User Sentiment Score**: {sentiment_score:.3f} ({'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'})")
                            
                        else:
                            st.warning("Current price data not available for selected date.")
                            
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    # Main content area
    st.subheader("📈 Price History (Model Training Period)")
    
    # Filter data to show only the model's training period
    model_start_date = pd.to_datetime('2020-01-01')
    model_end_date = pd.to_datetime('2021-01-01')
    model_period_data = df[(df['Date'] >= model_start_date) & (df['Date'] < model_end_date)]
    
    # Show recent price chart within model period
    recent_data = model_period_data.tail(30)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.line_chart(
            recent_data.set_index('Date')['Close'],
            use_container_width=True
        )
    
    with col2:
        st.subheader("📊 Key Statistics")
        
        if not model_period_data.empty:
            latest_price = model_period_data['Close'].iloc[-1]
            price_24h_ago = model_period_data['Close'].iloc[-2] if len(model_period_data) > 1 else latest_price
            change_24h = latest_price - price_24h_ago
            change_24h_pct = (change_24h / price_24h_ago) * 100
            
            st.metric(
                label="Latest Price (Model Period)",
                value=f"${latest_price:.2f}",
                delta=f"{change_24h:+.2f} ({change_24h_pct:+.2f}%)"
            )
            
            # Additional stats
            st.write(f"**Volume**: {model_period_data['Volume'].iloc[-1]:,.0f}")
            st.write(f"**High**: ${model_period_data['High'].iloc[-1]:.2f}")
            st.write(f"**Low**: ${model_period_data['Low'].iloc[-1]:.2f}")
        else:
            st.warning("No data available for the model training period.")
    
    # Model information
    st.subheader("🤖 Model Information")
    st.info(f"""
    This prediction model uses:
    - **LSTM Neural Network** with 256 hidden units
    - **29 technical indicators** including price, volume, and sentiment data
    - **50-day historical sequence** for pattern recognition
    - **Real-time sentiment analysis** from user comments
    - **Training Period**: {model_start_date.strftime('%Y-%m-%d')} to {model_end_date.strftime('%Y-%m-%d')}
    
    ⚠️ **Disclaimer**: This is for educational purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions.
    """)

if __name__ == "__main__":
    main()

