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
LABEL_MODE = os.getenv("ML_LABEL_MODE", "future_return")
FUTURE_STEPS = int(os.getenv("ML_FUTURE_STEPS", "3"))
THRESHOLD = float(os.getenv("ML_THRESHOLD", "0.002"))
TP_SL_LOOKAHEAD_STEPS = int(os.getenv("ML_TP_SL_LOOKAHEAD_STEPS", "12"))
TP_RATE = float(os.getenv("ML_TP_RATE", os.getenv("TAKE_PROFIT_RATE", "0.010")))
SL_RATE = float(os.getenv("ML_SL_RATE", os.getenv("STOP_LOSS_RATE", "0.008")))
TP_SL_NEUTRAL_ACTION = os.getenv("ML_TP_SL_NEUTRAL_ACTION", "drop")
TP_SL_TIE_BREAKER = os.getenv("ML_TP_SL_TIE_BREAKER", "stop_loss")

OUTPUT_DIR = Path("data/ml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if LABEL_MODE not in {"future_return", "tp_sl_first"}:
    raise ValueError("ML_LABEL_MODE must be future_return or tp_sl_first.")

if TP_SL_NEUTRAL_ACTION not in {"drop", "loss"}:
    raise ValueError("ML_TP_SL_NEUTRAL_ACTION must be drop or loss.")

if TP_SL_TIE_BREAKER not in {"stop_loss", "take_profit"}:
    raise ValueError("ML_TP_SL_TIE_BREAKER must be stop_loss or take_profit.")


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

    df["label_mode"] = LABEL_MODE
    df["future_close"] = df["close"].shift(-FUTURE_STEPS)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

    if LABEL_MODE == "future_return":
        df["label"] = 0
        df.loc[df["future_return"] > THRESHOLD, "label"] = 1
        return df

    labels: list[int | None] = []
    outcomes: list[str | None] = []
    exit_steps: list[int | None] = []

    for i, row in df.iterrows():
        entry_price = float(row["close"])
        tp_price = entry_price * (1 + TP_RATE)
        sl_price = entry_price * (1 - SL_RATE)

        label = None
        outcome = None
        exit_step = None

        end_i = min(i + TP_SL_LOOKAHEAD_STEPS, len(df) - 1)

        for future_i in range(i + 1, end_i + 1):
            future_row = df.iloc[future_i]
            hit_tp = float(future_row["high"]) >= tp_price
            hit_sl = float(future_row["low"]) <= sl_price

            if hit_tp and hit_sl:
                exit_step = future_i - i

                if TP_SL_TIE_BREAKER == "take_profit":
                    label = 1
                    outcome = "take_profit"
                else:
                    label = 0
                    outcome = "stop_loss"

                break

            if hit_tp:
                label = 1
                outcome = "take_profit"
                exit_step = future_i - i
                break

            if hit_sl:
                label = 0
                outcome = "stop_loss"
                exit_step = future_i - i
                break

        if outcome is None and TP_SL_NEUTRAL_ACTION == "loss":
            label = 0
            outcome = "neutral_as_loss"
            exit_step = TP_SL_LOOKAHEAD_STEPS

        labels.append(label)
        outcomes.append(outcome)
        exit_steps.append(exit_step)

    df["label"] = labels
    df["label_outcome"] = outcomes
    df["label_exit_step"] = exit_steps
    df["label_tp_rate"] = TP_RATE
    df["label_sl_rate"] = SL_RATE
    df["label_lookahead_steps"] = TP_SL_LOOKAHEAD_STEPS

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
    print(f"label_mode: {LABEL_MODE}")
    print(f"future_steps: {FUTURE_STEPS}")
    print(f"threshold: {THRESHOLD}")
    if LABEL_MODE == "tp_sl_first":
        print(f"tp_rate: {TP_RATE}")
        print(f"sl_rate: {SL_RATE}")
        print(f"tp_sl_lookahead_steps: {TP_SL_LOOKAHEAD_STEPS}")
        print(f"tp_sl_neutral_action: {TP_SL_NEUTRAL_ACTION}")
        print(f"tp_sl_tie_breaker: {TP_SL_TIE_BREAKER}")
    print(f"rows: {len(df)}")
    print(f"output: {output_path}")
    print("label counts:")
    print(df["label"].value_counts().sort_index())
    if LABEL_MODE == "tp_sl_first":
        print("label outcomes:")
        print(df["label_outcome"].value_counts(dropna=False))
    print("===================================")


if __name__ == "__main__":
    main()
