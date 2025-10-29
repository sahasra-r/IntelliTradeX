# enhanced_features.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Step 1: Fetch stock data
# -----------------------------
stock_symbol = "AAPL"  # You can change to TSLA, MSFT, etc.
data = yf.download(stock_symbol, period="2y", interval="1d")
data = data.dropna()  # Remove any missing values

print("📊 Raw stock data preview:")
print(data.head())

# -----------------------------
# Step 2: Add technical indicators
# -----------------------------

# Simple Moving Average (SMA)
data['SMA_20'] = data['Close'].rolling(window=20).mean()

# Exponential Moving Average (EMA)
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

# Relative Strength Index (RSI)
delta = data['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
ema_up = up.ewm(span=14, adjust=False).mean()
ema_down = down.ewm(span=14, adjust=False).mean()
rs = ema_up / ema_down
data['RSI'] = 100 - (100 / (1 + rs))

# Drop rows with NaN values created by rolling calculations
data = data.dropna()

print("\n📊 Stock data with indicators preview:")
print(data.head())

# -----------------------------
# Step 3: Scale features for LSTM
# -----------------------------
features = ['Close', 'SMA_20', 'EMA_20', 'RSI', 'Volume']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

print("\n📊 Scaled features preview:")
print(scaled_data[:5])

# Optionally, save preprocessed data for later use
data.to_csv(f"{stock_symbol}_preprocessed.csv", index=True)
print(f"\n✅ Preprocessed data saved as {stock_symbol}_preprocessed.csv")
