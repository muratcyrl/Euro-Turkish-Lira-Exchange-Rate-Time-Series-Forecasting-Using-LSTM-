from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "EUR-TUR.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

WINDOW_SIZE = 30
STEP_SIZE = 1
FORECAST_HORIZON = 30
LOOKBACK_PLOT_POINTS = 180
RANDOM_SEED = 42

CSV_COLUMNS = ["Date", "Close", "Open", "High", "Low"]
FEATURE_COLUMNS = CSV_COLUMNS[1:]


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_sliding_windows(
    data: np.ndarray, window_size: int, step_size: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    features, targets = [], []

    for index in range(0, len(data) - window_size, step_size):
        features.append(data[index : index + window_size])
        targets.append(data[index + window_size])

    return np.array(features), np.array(targets)


def load_exchange_rate_data(csv_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path, encoding="utf-8-sig")
    dataframe = dataframe.iloc[:, : len(CSV_COLUMNS)].copy()
    dataframe.columns = CSV_COLUMNS
    dataframe["Date"] = pd.to_datetime(
        dataframe["Date"], dayfirst=True, errors="coerce"
    )
    dataframe = dataframe.dropna(subset=["Date"]).sort_values("Date").reset_index(
        drop=True
    )

    for column in FEATURE_COLUMNS:
        series = dataframe[column].astype(str)
        series = series.str.replace(".", "", regex=False)
        series = series.str.replace(",", ".", regex=False)
        dataframe[column] = series.astype(float)

    return dataframe


def build_model(window_size: int, feature_count: int) -> Sequential:
    model = Sequential(
        [
            tf.keras.Input(shape=(window_size, feature_count)),
            LSTM(128, return_sequences=False),
            Dropout(0.2),
            Dense(feature_count),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def split_indices(sample_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_start_pct = 0.3
    train_middle_pct = 0.3
    train_end_pct = 0.1

    n_start = int(train_start_pct * sample_count)
    n_middle = int(train_middle_pct * sample_count)
    n_end = int(train_end_pct * sample_count)

    start_indices = np.arange(0, n_start)
    middle_indices = np.arange(
        sample_count // 2 - n_middle // 2, sample_count // 2 + n_middle // 2
    )
    end_indices = np.arange(sample_count - n_end, sample_count)

    train_indices = np.unique(
        np.concatenate([start_indices, middle_indices, end_indices])
    )
    remaining_indices = np.setdiff1d(np.arange(sample_count), train_indices)

    n_validation = len(remaining_indices) // 2
    validation_indices = remaining_indices[:n_validation]
    test_indices = remaining_indices[n_validation:]

    return train_indices, validation_indices, test_indices


def save_training_plot(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    figure = plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=200)
    plt.close(figure)


def save_test_prediction_plot(
    test_dates: pd.Series,
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
    output_dir: Path,
) -> None:
    figure = plt.figure(figsize=(12, 5))
    plt.plot(test_dates, actual_values[:, 0], label="Actual close", color="steelblue")
    plt.plot(
        test_dates,
        predicted_values[:, 0],
        label="Predicted close",
        color="tomato",
    )
    plt.xlabel("Date")
    plt.ylabel("Exchange rate")
    plt.title("Test Set Close Price Predictions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figure.savefig(output_dir / "test_close_predictions.png", dpi=200)
    plt.close(figure)


def save_future_forecast_plot(
    historical_dates: pd.Series,
    historical_values: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    output_dir: Path,
) -> None:
    figure = plt.figure(figsize=(12, 5))
    plt.plot(
        historical_dates,
        historical_values[:, 0],
        color="steelblue",
        label="Historical close",
    )
    plt.plot(
        forecast_dates,
        forecast_values[:, 0],
        color="tomato",
        label="Forecasted close",
    )
    plt.xlabel("Date")
    plt.ylabel("Exchange rate")
    plt.title("Next 30 Business Days Forecast")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figure.savefig(output_dir / "future_30_day_forecast.png", dpi=200)
    plt.close(figure)


def main() -> None:
    set_reproducibility(RANDOM_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    exchange_rates = load_exchange_rate_data(DATA_PATH)
    exchange_rates.to_csv(OUTPUT_DIR / "cleaned_exchange_rates.csv", index=False)

    original_feature_values = exchange_rates[FEATURE_COLUMNS].to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_feature_values = scaler.fit_transform(original_feature_values)

    with (OUTPUT_DIR / "minmax_scaler.pkl").open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    X, y = create_sliding_windows(
        scaled_feature_values, window_size=WINDOW_SIZE, step_size=STEP_SIZE
    )
    train_indices, validation_indices, test_indices = split_indices(len(X))

    X_train, y_train = X[train_indices], y[train_indices]
    X_validation, y_validation = X[validation_indices], y[validation_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print(
        f"Train: {X_train.shape} Validation: {X_validation.shape} Test: {X_test.shape}"
    )

    model = build_model(WINDOW_SIZE, X_train.shape[2])
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_validation, y_validation),
        verbose=1,
        shuffle=False,
        callbacks=[early_stopping],
    )

    model.save(OUTPUT_DIR / "lstm_exchange_rate_model.keras")

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(OUTPUT_DIR / "training_history.csv", index_label="epoch")
    save_training_plot(history, OUTPUT_DIR)

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    test_predictions_scaled = model.predict(X_test, verbose=0)
    test_predictions = scaler.inverse_transform(test_predictions_scaled)
    test_targets = scaler.inverse_transform(y_test)

    target_dates = exchange_rates["Date"].iloc[WINDOW_SIZE:].reset_index(drop=True)
    test_dates = target_dates.iloc[test_indices].reset_index(drop=True)

    test_results = pd.DataFrame(
        {
            "Date": test_dates,
            "Actual_Close": test_targets[:, 0],
            "Predicted_Close": test_predictions[:, 0],
            "Actual_Open": test_targets[:, 1],
            "Predicted_Open": test_predictions[:, 1],
            "Actual_High": test_targets[:, 2],
            "Predicted_High": test_predictions[:, 2],
            "Actual_Low": test_targets[:, 3],
            "Predicted_Low": test_predictions[:, 3],
        }
    )
    test_results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    save_test_prediction_plot(test_dates, test_targets, test_predictions, OUTPUT_DIR)

    rolling_window = scaled_feature_values[-WINDOW_SIZE:].copy()
    future_predictions_scaled = []

    for _ in range(FORECAST_HORIZON):
        predicted_step = model.predict(rolling_window[np.newaxis, :, :], verbose=0)[0]
        future_predictions_scaled.append(predicted_step)
        rolling_window = np.vstack([rolling_window[1:], predicted_step])

    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled))
    last_date = exchange_rates["Date"].iloc[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON
    )

    future_forecast = pd.DataFrame(
        future_predictions, columns=[f"Predicted_{column}" for column in FEATURE_COLUMNS]
    )
    future_forecast.insert(0, "Date", future_dates)
    future_forecast.to_csv(OUTPUT_DIR / "future_30_day_forecast.csv", index=False)

    history_dates = exchange_rates["Date"].tail(LOOKBACK_PLOT_POINTS)
    history_values = original_feature_values[-LOOKBACK_PLOT_POINTS:]
    save_future_forecast_plot(
        history_dates, history_values, future_dates, future_predictions, OUTPUT_DIR
    )

    metadata = {
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_validation)),
        "test_samples": int(len(X_test)),
        "test_loss": float(test_loss),
        "test_mae": float(test_mae),
        "output_directory": str(OUTPUT_DIR),
    }
    with (OUTPUT_DIR / "run_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")


if __name__ == "__main__":
    main()
