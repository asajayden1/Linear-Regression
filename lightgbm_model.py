import sqlite3
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


DB_PATH = "my_stock_model.sqlite"


def main():

    # -----------------------------
    # Load data from SQLite
    # -----------------------------

    conn = sqlite3.connect(DB_PATH)

    data = pd.read_sql_query("""
        SELECT
            symbol,
            ts_utc,
            close,
            volume,
            price_range,
            recent_return,
            future_return
        FROM training_data
        ORDER BY ts_utc ASC
    """, conn)

    conn.close()

    print("Loaded rows:", len(data))

    # -----------------------------
    # Select features and target
    # -----------------------------

    features = [
        "close",
        "volume",
        "price_range",
        "recent_return"
    ]

    X = data[features]
    y = data["future_return"]

    # -----------------------------
    # Time-based train/test split
    # -----------------------------

    split_index = int(len(data) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # -----------------------------
    # Create LightGBM model
    # -----------------------------

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
        verbosity=-1
    )

    # -----------------------------
    # Train model
    # -----------------------------

    print()
    print("Training LightGBM model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # -----------------------------
    # Make predictions
    # -----------------------------

    predictions = model.predict(X_test)

    # -----------------------------
    # Calculate metrics
    # -----------------------------

    r2 = r2_score(y_test, predictions)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    # -----------------------------
    # Display results
    # -----------------------------

    print()
    print("======================================")
    print("LIGHTGBM RESULTS")
    print("======================================")

    print(f"R²:   {r2:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    # -----------------------------
    # Feature importance
    # -----------------------------

    print()
    print("======================================")
    print("FEATURE IMPORTANCE")
    print("======================================")

    importance = model.feature_importances_

    for feature, value in zip(features, importance):
        print(f"{feature}: {value}")

    # -----------------------------
    # Sample predictions
    # -----------------------------

    print()
    print("======================================")
    print("SAMPLE PREDICTIONS")
    print("======================================")

    for actual, predicted in zip(
        y_test.iloc[:10],
        predictions[:10]
    ):
        print(
            f"Actual: {actual:.6f} | "
            f"Predicted: {predicted:.6f}"
        )


if __name__ == "__main__":
    main()