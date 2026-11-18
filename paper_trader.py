import os
from datetime import datetime, timedelta, timezone

import lightgbm as lgb
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# ---------------------------------
# Load Alpaca credentials
# ---------------------------------

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY")
API_SECRET = os.getenv("APCA_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("Alpaca API credentials were not found in .env")


# ---------------------------------
# Load trained LightGBM model
# ---------------------------------

model = lgb.Booster(
    model_file="lightgbm_model.txt"
)

print("LightGBM model loaded.")


# ---------------------------------
# Connect to Alpaca market data
# ---------------------------------

client = StockHistoricalDataClient(
    API_KEY,
    API_SECRET
)

symbol = "AAPL"

end = datetime.now(timezone.utc)
start = end - timedelta(hours=2)

request = StockBarsRequest(
    symbol_or_symbols=symbol,
    timeframe=TimeFrame(
        5,
        TimeFrameUnit.Minute
    ),
    start=start,
    end=end,
    feed="iex"
)

bars = client.get_stock_bars(request)

df = bars.df.reset_index()

if df.empty:
    raise ValueError("No market data was returned.")


# ---------------------------------
# Prepare features
# ---------------------------------

df["price_range"] = df["high"] - df["low"]

df["recent_return"] = df["close"].pct_change()

df = df.dropna()

latest = df.iloc[-1]


features = pd.DataFrame([{
    "close": latest["close"],
    "volume": latest["volume"],
    "price_range": latest["price_range"],
    "recent_return": latest["recent_return"]
}])


# ---------------------------------
# Make prediction
# ---------------------------------

prediction = model.predict(features)[0]


# ---------------------------------
# Make paper-trading decision
# ---------------------------------

if prediction > 0.002:
    decision = "BUY"

elif prediction < -0.002:
    decision = "SELL"

else:
    decision = "HOLD"


# ---------------------------------
# Display results
# ---------------------------------

timestamp = datetime.now(timezone.utc).isoformat()

print()
print("======================================")
print("PAPER TRADING MODEL")
print("======================================")

print(f"Time: {timestamp}")
print(f"Symbol: {symbol}")
print(f"Current price: ${latest['close']:.2f}")
print(f"Predicted future return: {prediction:.6f}")
print(f"Model decision: {decision}")


# ---------------------------------
# Save decision to CSV
# ---------------------------------

log_file = "paper_trading_log.csv"

new_row = pd.DataFrame([{
    "timestamp": timestamp,
    "symbol": symbol,
    "price": latest["close"],
    "prediction": prediction,
    "decision": decision
}])

if os.path.exists(log_file):
    new_row.to_csv(
        log_file,
        mode="a",
        header=False,
        index=False
    )
else:
    new_row.to_csv(
        log_file,
        index=False
    )

print()
print(f"Decision saved to {log_file}")
print("No real-money trade was placed.")