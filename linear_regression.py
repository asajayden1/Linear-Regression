import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Your database
DB_PATH = "my_stock_model.sqlite"


def main():

    # Connect to your database
    conn = sqlite3.connect(DB_PATH)

    # Load the data
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

    # -----------------------------------------
    # Features and target
    # -----------------------------------------

    X = data[
        [
            "close",
            "volume",
            "price_range",
            "recent_return"
        ]
    ]

    y = data["future_return"]

    # -----------------------------------------
    # Chronological train/test split
    # -----------------------------------------
    # First 80% = training
    # Last 20% = testing

    split_index = int(len(data) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # -----------------------------------------
    # Create the linear regression model
    # -----------------------------------------

    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # -----------------------------------------
    # Make predictions
    # -----------------------------------------

    predictions = model.predict(X_test)

    # -----------------------------------------
    # Evaluate the model
    # -----------------------------------------

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print()
    print("========== LINEAR REGRESSION RESULTS ==========")

    print("R²:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)

    # -----------------------------------------
    # Model coefficients
    # -----------------------------------------

    print()
    print("========== MODEL COEFFICIENTS ==========")

    features = [
        "close",
        "volume",
        "price_range",
        "recent_return"
    ]

    for feature, coefficient in zip(features, model.coef_):
        print(f"{feature}: {coefficient}")

    print("Intercept:", model.intercept_)

    # -----------------------------------------
    # Show predictions
    # -----------------------------------------

    print()
    print("========== SAMPLE PREDICTIONS ==========")

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