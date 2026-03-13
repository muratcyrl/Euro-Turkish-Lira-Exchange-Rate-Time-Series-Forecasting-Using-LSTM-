# Euro - Turkish Lira Exchange Rate Forecasting with LSTM

This project trains an LSTM model on historical EUR/TRY data and produces:

- Test-set predictions against known values
- A recursive 30-business-day forecast
- Saved charts, CSV outputs, scaler, model, and run metadata

The goal is educational and experimental. Forecasts can be noisy and should not be used as financial advice.

## Project Files

- `LSTM-FORECASTING.py`: Main training + forecasting pipeline
- `EUR-TUR.csv`: Input time-series dataset
- `requirements.txt`: Python dependencies
- `outputs/`: Generated artifacts after a run (ignored by git)

## How to Run

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the training and forecasting script

```bash
python LSTM-FORECASTING.py
```

The script trains the model, evaluates on a held-out test set, creates a 30-day forecast, and writes all outputs into the `outputs/` directory.

## What Gets Saved

After execution, the following files are created in `outputs/`:

- `cleaned_exchange_rates.csv`: Parsed and cleaned input data in English column names
- `training_history.csv`: Epoch-by-epoch training/validation metrics
- `training_history.png`: Loss curves
- `test_predictions.csv`: Actual vs predicted values on the test set
- `test_close_predictions.png`: Test-set close price chart
- `future_30_day_forecast.csv`: 30 business-day forecast values
- `future_30_day_forecast.png`: Historical close + future close forecast plot
- `minmax_scaler.pkl`: Fitted scaler for inverse transforms/reuse
- `lstm_exchange_rate_model.keras`: Trained Keras model
- `run_metadata.json`: Run parameters and summary metrics

## Code Walkthrough

`LSTM-FORECASTING.py` runs this pipeline:

1. Load `EUR-TUR.csv`, convert date format, and normalize number formatting.
2. Rename columns to English (`Date`, `Close`, `Open`, `High`, `Low`).
3. Scale features to `[0, 1]` using `MinMaxScaler`.
4. Build sliding windows of length `30` to predict the next step.
5. Split windows into train/validation/test with a custom index strategy.
6. Train an LSTM model (`LSTM(128) -> Dropout(0.2) -> Dense(4)`).
7. Evaluate test performance and inverse-transform predictions.
8. Perform recursive multi-step forecasting for the next `30` business days.
9. Save all artifacts to `outputs/`.

## Configuration

You can edit these constants in `LSTM-FORECASTING.py`:

- `WINDOW_SIZE`: Lookback window length
- `FORECAST_HORIZON`: Number of future business days to forecast
- `LOOKBACK_PLOT_POINTS`: Number of recent historical points shown in final forecast chart
- `RANDOM_SEED`: Reproducibility seed for NumPy and TensorFlow

## Notes and Limitations

- Recursive forecasting compounds error over time, so long-horizon forecasts are less reliable.
- Results depend on dataset quality and preprocessing.
- Current split strategy is custom and not a strict chronological train-then-test split.
