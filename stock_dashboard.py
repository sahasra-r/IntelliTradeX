import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from textblob import TextBlob
import datetime

st.set_page_config(page_title="IntelliTradeX - AI Stock Dashboard", layout="wide")
st.title("📊 IntelliTradeX: AI-Powered Stock Prediction Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("Stock Settings")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY.NS)", "AAPL")
# Using 2 years of historical data for better LSTM predictions
start_date = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=730))
end_date = st.sidebar.date_input("End Date", datetime.date.today() + datetime.timedelta(days=1))

# --- Fetch Data ---
@st.cache_data
def get_stock_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

data = get_stock_data(stock_symbol, start_date, end_date)

if data.empty:
    st.error("❌ No data fetched. Please check the stock symbol or date range.")
    st.stop()

st.subheader(f"📈 {stock_symbol} Stock Data Overview")
st.dataframe(data.tail())

# --- Chart Visualization ---
st.subheader("📈 Price Chart")
if not data.empty:
    # Create a line chart for the closing prices
    st.line_chart(data['Close'], use_container_width=True)
    
    # Show additional price information
    col1, col2, col3 = st.columns(3)
    with col1:
        current_price = float(data['Close'].iloc[-1])
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        ma50 = float(data['Close'].rolling(window=50).mean().iloc[-1])
        st.metric("50-Day MA", f"${ma50:.2f}" if not pd.isna(ma50) else "N/A")
    with col3:
        ma200 = float(data['Close'].rolling(window=200).mean().iloc[-1])
        st.metric("200-Day MA", f"${ma200:.2f}" if not pd.isna(ma200) else "N/A")
else:
    st.warning("No data available to display the chart.")

# --- Plot Closing Prices ---
st.line_chart(data['Close'])

# --- LSTM Model Training ---
st.subheader("🤖 LSTM-based Future Price Prediction")

dataset = data['Close'].values
if len(dataset) < 60:
    st.warning("⚠️ Not enough data points for LSTM model. Try a wider date range.")
    st.stop()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]

X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# --- Build LSTM Model ---
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")

with st.spinner("Training AI model... ⏳"):
    model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)
st.success("✅ LSTM Model Trained Successfully!")

# --- Predict Next Closing Price ---
last_60_days = scaled_data[-60:]
X_test = np.reshape(last_60_days, (1, 60, 1))
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)[0][0]

curr_close = float(data['Close'].iloc[-1])
st.metric(
    label="Predicted Next Closing Price",
    value=f"${predicted_price:.2f}",
    delta=f"{float(predicted_price) - curr_close:.2f}"
)

# --- Sentiment Analysis Section ---
st.subheader("💬 Market Sentiment Analysis")

news_text = st.text_area("Paste Latest News or Tweet About the Stock", 
                         "Apple launches new AI-powered MacBooks.")
if st.button("Analyze Sentiment"):
    blob = TextBlob(news_text)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0.1:
        sentiment_label = "😊 Positive"
    elif sentiment_score < -0.1:
        sentiment_label = "😟 Negative"
    else:
        sentiment_label = "😐 Neutral"

    st.write(f"**Sentiment:** {sentiment_label} (Score: {sentiment_score:.2f})")

# --- Combined Insights ---
st.subheader("📊 AI Insights")
trend = "📈 Uptrend Expected" if predicted_price > curr_close else "📉 Possible Downtrend"
st.info(f"**AI Prediction:** {trend}\n\n**LSTM Predicted Close:** ${predicted_price:.2f}")

# --- Chart Visualization ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data.index, data['Close'], label="Actual Prices", color="blue")
ax.scatter(data.index[-1], curr_close, color="green", label="Current Price")
ax.scatter(data.index[-1] + pd.Timedelta(days=1), predicted_price, color="red", label="Predicted Next Price")
ax.legend()
ax.set_title(f"{stock_symbol} Price Prediction")
st.pyplot(fig)

st.success("✅ Dashboard executed successfully!")
