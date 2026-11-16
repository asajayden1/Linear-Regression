"""IB Computer Science — Create a stock database using Alpaca data."""

import os
import sqlite3
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# Load API credentials from .env
load_dotenv()

API_KEY = os.getenv("APCA_API_KEY")
API_SECRET = os.getenv("APCA_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("Alpaca API credentials were not found in .env")


# Your own database
OUTPUT_DB = "my_stock_model.sqlite"

# Stocks for the model
STOCKS = ["AAPL", "MSFT", "NVDA"]

# Six 5-minute bars = approximately 30 minutes
HORIZON_BARS = 6


def main():

    print("Connecting to Alpaca...")

    client = StockHistoricalDataClient(
        API_KEY,
        API_SECRET
    )

    print("Connected to Alpaca!")

    # Delete old database so we create a fresh one
    if os.path.exists(OUTPUT_DB):
        os.remove(OUTPUT_DB)

    conn = sqlite3.connect(OUTPUT_DB)

    conn.execute("""
        CREATE TABLE training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            ts_utc TEXT NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            price_range REAL NOT NULL,
            recent_return REAL NOT NULL,
            future_return REAL NOT NULL
        )
    """)

    # Get approximately one year of data
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365)

    print()
    print("Downloading Alpaca 5-minute stock data...")
    print(f"Start: {start_date}")
    print(f"End:   {end_date}")

    for stock in STOCKS:

        print()
        print(f"Processing {stock}...")

        request = StockBarsRequest(
        symbol_or_symbols=stock,
        timeframe=TimeFrame(
        5,
        TimeFrameUnit.Minute
        ),
        start=start_date,
        end=end_date,
        feed="iex"
        )

        bars = client.get_stock_bars(request)

        stock_bars = bars[stock]

        print(f"Found {len(stock_bars)} bars.")

        # Convert Alpaca bars into a list
        rows = []

        for bar in stock_bars:
            rows.append({
                "timestamp": bar.timestamp.isoformat(),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "high": float(bar.high),
                "low": float(bar.low)
            })

        # Create training examples
        for i in range(1, len(rows) - HORIZON_BARS):

            current = rows[i]
            previous = rows[i - 1]
            future = rows[i + HORIZON_BARS]

            close = current["close"]
            previous_close = previous["close"]
            future_close = future["close"]

            if close <= 0 or previous_close <= 0:
                continue

            price_range = current["high"] - current["low"]

            recent_return = (
                (close - previous_close)
                / previous_close
            )

            future_return = (
                (future_close - close)
                / close
            )

            conn.execute("""
                INSERT INTO training_data (
                    symbol,
                    ts_utc,
                    close,
                    volume,
                    price_range,
                    recent_return,
                    future_return
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                stock,
                current["timestamp"],
                close,
                current["volume"],
                price_range,
                recent_return,
                future_return
            ))

    conn.commit()

    count = conn.execute("""
        SELECT COUNT(*)
        FROM training_data
    """).fetchone()[0]

    print()
    print("======================================")
    print("Your Alpaca database was created!")
    print("======================================")
    print(f"File: {OUTPUT_DB}")
    print(f"Training examples: {count}")

    print()
    print("Rows by stock:")

    stock_counts = conn.execute("""
        SELECT symbol, COUNT(*)
        FROM training_data
        GROUP BY symbol
        ORDER BY symbol
    """).fetchall()

    for stock, count in stock_counts:
        print(f"{stock}: {count}")

    print()
    print("Sample data:")

    sample = conn.execute("""
        SELECT *
        FROM training_data
        LIMIT 10
    """).fetchall()

    for row in sample:
        print(row)

    conn.close()


if __name__ == "__main__":
    main()