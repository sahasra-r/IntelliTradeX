# train_lstm.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Step 0: Enable eager execution
# -----------------------------
tf.config.run_functions_eagerly(True)

# -----------------------------
# Step 1: Load preprocessed data
# -----------------------------
stock_symbol = "AAPL"
data = pd.read_csv(f"{stock_symbol}_preprocessed.csv", index_col=0)
features = ['Close', 'SMA_20', 'EMA_20', 'RSI', 'Volume']

# Scale features (again, just to ensure LSTM input is correct)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# -----------------------------
# Step 2: Create sequences for LSTM
# -----------------------------
sequence_length = 60  # last 60 days to predict next day
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])  # predict 'Close' price

X, y = np.array(X), np.array(y)
print(f"✅ Input shape: {X.shape}, Output shape: {y.shape}")

# -----------------------------
# Step 3: Build LSTM model
# -----------------------------
model = Sequential([
    LSTM(60, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(60, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# -----------------------------
# Step 4: Train model
# -----------------------------
epochs = 20
batch_size = 32

print("\n⏳ Training model...")
model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

# -----------------------------
# Step 5: Save model
# -----------------------------
model_filename = f"{stock_symbol}_lstm_model.h5"
model.save(model_filename)
print(f"✅ Model saved as {model_filename}")
    