import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Step 1: Choose a stock
stock_symbol = "AAPL"

# Step 2: Fetch 1 year of historical data
data = yf.download(stock_symbol, period="1y", interval="1d")

# Step 3: Handle missing values (if any)
data = data.dropna()

# Step 4: Use only 'Close' price for predictions
close_prices = data[['Close']].values  # shape: (num_days, 1)

# Step 5: Normalize prices between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices)

# Step 6: Create sequences for AI (e.g., last 60 days -> predict next day)
sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_prices)):
    X.append(scaled_prices[i-sequence_length:i, 0])  # past 60 days
    y.append(scaled_prices[i, 0])  # next day's price

# Convert to NumPy arrays
X, y = np.array(X), np.array(y)

# Step 7: Reshape X for LSTM (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print("✅ Preprocessing Done!")
print("X shape:", X.shape)
print("y shape:", y.shape)
