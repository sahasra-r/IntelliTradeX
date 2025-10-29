import yfinance as yf
import matplotlib.pyplot as plt

# Choose a stock symbol
stock_symbol = "AAPL"

# Fetch 1 year of historical data
data = yf.download(stock_symbol, period="1y", interval="1d")

# Preview data
print(data.head())

# Plot closing price
plt.figure(figsize=(10,5))
plt.plot(data["Close"], label="Close Price")
plt.title(f"{stock_symbol} Stock Closing Price (Last 1 Year)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()
