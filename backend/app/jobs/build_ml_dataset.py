import os
from pathlib import Path

import pandas as pd

from app.db.supabase_client import supabase
from app.services.ml_features import add_ml_indicators, normalize_candles_frame

EXCHANGE = os.getenv("TRADE_EXCHANGE", "bybit")
SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
MARKET_TYPE = os.getenv("TRADE_MARKET_TYPE", "linear")
TIMEFRAME = os.getenv("TRADE_TIMEFRAME", "5m")
BATCH_SIZE = int(os.getenv("ML_DATASET_BATCH_SIZE", "1000"))
MAX_ROWS = int(os.getenv("ML_DATASET_MAX_ROWS", "0"))

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
    all_rows: list[dict] = []
    offset = 0

    while True:
        end = offset + BATCH_SIZE - 1

        result = (
            supabase.table("candles")
            .select("*")
            .eq("market_id", market_id)
            .order("open_time", desc=False)
            .range(offset, end)
            .execute()
        )

        rows = result.data or []

        if not rows:
            break

        all_rows.extend(rows)
        print(f"loaded candles: {len(all_rows)}")

        if MAX_ROWS > 0 and len(all_rows) >= MAX_ROWS:
            all_rows = all_rows[:MAX_ROWS]
            break

        if len(rows) < BATCH_SIZE:
            break

        offset += BATCH_SIZE

    if not all_rows:
        raise RuntimeError("No candles found.")

    return normalize_candles_frame(pd.DataFrame(all_rows))


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
    print(f"raw candle rows: {len(df)}")

    df = add_ml_indicators(df)
    df = add_labels(df)

    df = df.dropna().reset_index(drop=True)

    output_path = OUTPUT_DIR / f"{SYMBOL}_{TIMEFRAME}_dataset.csv"
    df.to_csv(output_path, index=False)

    print("===================================")
    print("ML dataset created")
    print(f"symbol: {SYMBOL}")
    print(f"timeframe: {TIMEFRAME}")
    print(f"future_steps: {os.getenv('ML_FUTURE_STEPS', '3')}")
    print(f"threshold: {os.getenv('ML_THRESHOLD', '0.002')}")
    print(f"rows: {len(df)}")
    print(f"output: {output_path}")
    print("label counts:")
    print(df["label"].value_counts().sort_index())
    print("===================================")


if __name__ == "__main__":
    main()
