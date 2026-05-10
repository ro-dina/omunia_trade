import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.db.supabase_client import supabase

EXCHANGE = os.getenv("TRADE_EXCHANGE", "bybit")
SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
MARKET_TYPE = os.getenv("TRADE_MARKET_TYPE", "linear")
TIMEFRAME = os.getenv("TRADE_TIMEFRAME", "5m")
LIMIT = int(os.getenv("ML_DATASET_LIMIT", "5000"))

OUTPUT_DIR = Path("data/ml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_market_id() -> str:
    result = (
        supabase.table("markets")
        .select("id")
        .eq("exchange", EXCHANGE)
        .eq("symbol", SYMBOL)
        .eq("market_type", MARKET_TYPE)
        .eq("timeframe", TIMEFRAME)
        .limit(1)
        .execute()
    )

    if not result.data:
        raise RuntimeError(f"Market not found: {SYMBOL} {TIMEFRAME}")

    return result.data[0]["id"]


def fetch_candles(market_id: str) -> pd.DataFrame:
    result = (
        supabase.table("candles")
        .select("*")
        .eq("market_id", market_id)
        .order("open_time", desc=True)
        .limit(LIMIT)
        .execute()
    )

    rows = result.data or []

    if not rows:
        raise RuntimeError("No candles found.")

    df = pd.DataFrame(rows)
    df = df.sort_values("open_time").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"])

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1"] = df["close"].pct_change()
    df["log_return_1"] = np.log(df["close"]).diff()

    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_30"] = df["close"].rolling(30).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    df["sma_5_gap"] = (df["close"] - df["sma_5"]) / df["close"]
    df["sma_10_gap"] = (df["close"] - df["sma_10"]) / df["close"]
    df["sma_30_gap"] = (df["close"] - df["sma_30"]) / df["close"]

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["open_close_range"] = (df["close"] - df["open"]) / df["open"]

    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    
    FUTURE_STEPS = int(os.getenv("ML_FUTURE_STEPS", "3"))
    THRESHOLD = float(os.getenv("ML_THRESHOLD", "0.002"))

    future_steps = FUTURE_STEPS
    threshold = THRESHOLD

    df["future_close"] = df["close"].shift(-future_steps)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

    df["label"] = 0
    df.loc[df["future_return"] > threshold, "label"] = 1

    return df


def main() -> None:
    market_id = get_market_id()
    df = fetch_candles(market_id)

    df = add_indicators(df)
    df = add_labels(df)

    df = df.dropna().reset_index(drop=True)

    output_path = OUTPUT_DIR / f"{SYMBOL}_{TIMEFRAME}_dataset.csv"
    df.to_csv(output_path, index=False)

    print("===================================")
    print("ML dataset created")
    print(f"symbol: {SYMBOL}")
    print(f"timeframe: {TIMEFRAME}")
    print(f"rows: {len(df)}")
    print(f"output: {output_path}")
    print("label counts:")
    print(df["label"].value_counts().sort_index())
    print("===================================")


if __name__ == "__main__":
    main()