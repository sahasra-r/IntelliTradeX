import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

# Step 1: Fetch 1 year of stock data
stock_symbol = "AAPL"
df = yf.download(stock_symbol, period="1y", interval="1d")

# Step 2: Prepare data
df = df[['Close']]
dataset = df.values

# Step 3: Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Step 4: Create sequences
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Step 5: Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Train
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# Step 7: Save model + scaler
os.makedirs("backend/models", exist_ok=True)
model.save("backend/models/hybrid_model_AAPL.h5")
joblib.dump(scaler, "backend/models/scaler_AAPL.pkl")

print("✅ Model and Scaler saved successfully!")
