# 📈 Stock Price Analyzer

A comprehensive stock price prediction application that uses LSTM neural networks and sentiment analysis to predict stock prices. This application combines technical indicators, historical price data, and real-time sentiment analysis to provide accurate stock price predictions.

## 🚀 Features

- **Advanced LSTM Model**: Uses a 256-unit LSTM neural network with 2 layers for pattern recognition
- **Technical Indicators**: Incorporates 29 technical indicators including RSI, MACD, Bollinger Bands, and more
- **Sentiment Analysis**: Real-time sentiment analysis from user comments using TextBlob
- **Interactive UI**: Beautiful Streamlit interface with real-time charts and metrics
- **Historical Data**: 50-day historical sequence analysis for pattern recognition
- **Price Predictions**: Next-day stock price predictions with confidence metrics

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.12+
- **Machine Learning**: PyTorch, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Sentiment Analysis**: TextBlob
- **Visualization**: Streamlit Charts

## 📊 Model Architecture

The prediction model uses:
- **Input Size**: 29 features (price, volume, technical indicators, sentiment)
- **Hidden Size**: 256 units
- **Layers**: 2 LSTM layers with dropout (0.2)
- **Sequence Length**: 50 days of historical data
- **Output**: Single price prediction for next day

## 🎯 Key Improvements Made

### 1. Fixed Prediction Issues
- **Consistent Scaling**: Implemented proper scaler management to prevent same predictions
- **Feature Engineering**: Proper handling of 29 technical indicators
- **Sentiment Integration**: Real-time sentiment analysis from user comments

### 2. Enhanced User Interface
- **Modern Design**: Clean, professional interface with emojis and better layout
- **Real-time Charts**: Interactive price history visualization
- **Comprehensive Metrics**: Current price, predicted price, and change percentages
- **Market Insights**: Bullish/bearish predictions with detailed analysis

### 3. Improved Functionality
- **Error Handling**: Robust error handling for data loading and predictions
- **Caching**: Efficient data and model caching for better performance
- **Date Validation**: Proper date range validation
- **Sentiment Analysis**: Real-time sentiment scoring from user input

## 📁 Project Structure

```
Stock_Analyzer/
├── app.py                          # Main Streamlit application
├── test_app.py                     # Test script for functionality
├── final_prediction.ipynb          # Model training notebook
├── final_data.csv                  # Processed stock data with technical indicators
├── daily-avg-sentiment-scores.csv  # Historical sentiment data
├── tesla_lstm_model.pth           # Trained LSTM model
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Stock_Analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application (Choose one option)**

   **Option A: Clean Launcher (Recommended - No Errors)**
   ```bash
   python launch_app.py
   ```

   **Option B: Direct Streamlit Run**
   ```bash
   streamlit run app.py
   ```

   **Option C: With Custom Configuration**
   ```bash
   streamlit run app.py --server.fileWatcherType none --logger.level error
   ```

4. **Access the application**
   - Open your browser and go to `http://localhost:8501`
   - The application will automatically load the model and data

## 📈 How to Use

### Making Predictions

1. **Select Date**: Choose a date from the sidebar (must have at least 50 days of historical data)
2. **Enter Comments**: Add your market analysis or comments (one per line)
3. **Generate Prediction**: Click "Predict Next Day Price" to get the forecast

### Understanding Results

- **Current Price**: The closing price for the selected date
- **Predicted Price**: The forecasted price for the next trading day
- **Price Change**: The expected dollar and percentage change
- **Market Sentiment**: Analysis of your input comments
- **Market Insights**: Bullish/bearish prediction with detailed explanation

### Example Comments
```
Positive news about the company
Strong quarterly results expected
Market volatility due to earnings
Technical indicators showing upward trend
```

## 🔧 Technical Details

### Features Used
1. **Price Data**: Open, High, Low, Close, Adjusted Close
2. **Volume**: Trading volume and volume indicators
3. **Returns**: Daily and log returns
4. **Moving Averages**: SMA and EMA (short and long term)
5. **Momentum**: RSI, MACD, Momentum, ROC
6. **Volatility**: ATR, Bollinger Bands
7. **Time Features**: Day of week, month, quarter
8. **Lag Features**: Previous day's return and close price
9. **Sentiment**: Historical and user-provided sentiment scores

### Model Performance
- **R² Score**: 0.9655 (96.55% accuracy)
- **RMSE**: 10.27
- **MAE**: 6.36

## ⚠️ Important Notes

### Disclaimer
This application is for **educational purposes only**. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial advisors and conduct thorough research before making investment decisions.

### Limitations
- Predictions are based on historical patterns and may not account for unforeseen events
- Market conditions can change rapidly, affecting prediction accuracy
- The model is trained on historical data and may not perform well in unprecedented market conditions

## 🐛 Troubleshooting

### Common Issues

1. **PyTorch File Watcher Errors**
   - **Solution**: Use the clean launcher: `python launch_app.py`
   - These errors are harmless but can be annoying
   - The clean launcher suppresses all PyTorch-related warnings

2. **Model Loading Error**
   - Ensure `tesla_lstm_model.pth` is in the project directory
   - Check that PyTorch is properly installed

3. **Data Loading Error**
   - Verify that `final_data.csv` and `daily-avg-sentiment-scores.csv` exist
   - Check file permissions and paths

4. **Prediction Errors**
   - Ensure selected date has at least 50 days of historical data
   - Check that all required features are present in the data

### Clean Running Options

To avoid any terminal errors, use one of these methods:

1. **Clean Launcher (Recommended)**:
   ```bash
   python launch_app.py
   ```

2. **Streamlit with Error Suppression**:
   ```bash
   streamlit run app.py --server.fileWatcherType none --logger.level error
   ```

3. **Environment Variables**:
   ```bash
   set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
   streamlit run app.py
   ```

### Performance Tips
- Use recent dates for better prediction accuracy
- Provide detailed, relevant comments for better sentiment analysis
- The model performs best with consistent market conditions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit for the web interface
- Uses PyTorch for deep learning capabilities
- TextBlob for sentiment analysis
- Technical indicators calculated using TA-Lib concepts

---

**Happy Trading! 📈💰**