import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Stock symbol to match your trained model
stock_symbol = "AAPL"

# Fetch 1 year of data
data = yf.download(stock_symbol, period="1y", interval="1d")

# Get the closing prices
dataset = data['Close'].values.reshape(-1, 1)

# Initialize and fit the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(dataset)

# Save the scaler
joblib.dump(scaler, f"scaler_{stock_symbol}.pkl")

print(f"✅ Scaler saved successfully as scaler_{stock_symbol}.pkl in the current folder!")
