from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


@dataclass
class TrainResult:
    model: Sequential
    scaler: MinMaxScaler
    last_window: np.ndarray
    last_price: float


def fetch_close_prices(symbol: str, period: str, interval: str) -> np.ndarray:
    data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if data.empty or "Close" not in data:
        raise ValueError(f"No close-price data found for {symbol} ({period}, {interval}).")
    closes = data["Close"].dropna().to_numpy(dtype=np.float32)
    if closes.size < 2:
        raise ValueError("Not enough price points returned to train a model.")
    return closes


def create_sequences(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(window_size, len(values)):
        x.append(values[i - window_size : i, 0])
        y.append(values[i, 0])
    if not x:
        raise ValueError("Not enough points to create training sequences. Lower --window or request more data.")
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def build_model(window_size: int) -> Sequential:
    model = Sequential(
        [
            LSTM(50, input_shape=(window_size, 1)),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_predictor(
    symbol: str,
    period: str,
    interval: str,
    window_size: int,
    epochs: int,
    batch_size: int,
) -> TrainResult:
    closes = fetch_close_prices(symbol, period, interval)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(closes.reshape(-1, 1))

    x_train, y_train = create_sequences(scaled, window_size)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    model = build_model(window_size)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    return TrainResult(
        model=model,
        scaler=scaler,
        last_window=scaled[-window_size:].reshape(1, window_size, 1),
        last_price=float(closes[-1]),
    )


def predict_next_close(result: TrainResult) -> float:
    scaled_prediction = result.model.predict(result.last_window, verbose=0)
    predicted = result.scaler.inverse_transform(scaled_prediction)
    return float(predicted[0][0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple LSTM model and predict the next stock closing price.")
    parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--period", default="2y", help="Data period supported by yfinance (default: 2y)")
    parser.add_argument("--interval", default="1d", help="Data interval supported by yfinance (default: 1d)")
    parser.add_argument("--window", type=int, default=60, help="Lookback window size (default: 60)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_predictor(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
        window_size=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    next_close = predict_next_close(result)

    print(f"Symbol: {args.symbol}")
    print(f"Latest close: {result.last_price:.2f}")
    print(f"Predicted next close: {next_close:.2f}")


if __name__ == "__main__":
    main()
