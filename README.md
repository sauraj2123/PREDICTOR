# PREDICTOR

A small side project for learning LSTMs by training a stock price predictor for swing-trading experiments.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python stock_lstm_predictor.py AAPL --period 2y --interval 1d --window 60 --epochs 10
```

The script downloads historical close prices with `yfinance`, trains a simple LSTM model, and prints a predicted next closing price.
