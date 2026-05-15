import os
from pathlib import Path

import joblib
import pandas as pd

SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("TRADE_TIMEFRAME", "5m")
ML_MODEL_NAME = os.getenv("ML_MODEL_NAME", "RandomForest")

DATASET_PATH = Path(
    os.getenv("ML_DATASET_PATH", f"data/ml/{SYMBOL}_{TIMEFRAME}_dataset.csv")
)
MODEL_PATH = Path(
    os.getenv("ML_MODEL_PATH", f"data/models/{SYMBOL}_{TIMEFRAME}_{ML_MODEL_NAME}.joblib")
)

ML_PROBA_THRESHOLD = float(os.getenv("ML_PROBA_THRESHOLD", "0.50"))
BACKTEST_EVAL_START_RATIO = float(os.getenv("BACKTEST_EVAL_START_RATIO", "0.8"))

INITIAL_CASH = 10_000.0
TRADE_NOTIONAL = 1_000.0
FEE_RATE = 0.0006

SHORT_PERIOD = 5
LONG_PERIOD = 30
RSI_BUY_THRESHOLD = 60.0
RSI_SELL_THRESHOLD = 35.0
TAKE_PROFIT_RATE = float(os.getenv("TAKE_PROFIT_RATE", "0.015"))
STOP_LOSS_RATE = float(os.getenv("STOP_LOSS_RATE", "0.003"))


def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def load_dataset(feature_columns: list[str]) -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=feature_columns + ["label"]).reset_index(drop=True)

    return df


def get_positive_class_index(model, positive_label: int = 1) -> int:
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif hasattr(model, "named_steps") and "model" in model.named_steps:
        classes = list(model.named_steps["model"].classes_)
    else:
        raise RuntimeError("Could not read model classes.")

    if positive_label not in classes:
        raise RuntimeError(f"Positive label {positive_label} not found in classes: {classes}")

    return classes.index(positive_label)


def add_sma_cross_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sma_short"] = df["close"].rolling(SHORT_PERIOD).mean()
    df["sma_long"] = df["close"].rolling(LONG_PERIOD).mean()
    df["prev_sma_short"] = df["sma_short"].shift(1)
    df["prev_sma_long"] = df["sma_long"].shift(1)

    df["prev_diff"] = df["prev_sma_short"] - df["prev_sma_long"]
    df["curr_diff"] = df["sma_short"] - df["sma_long"]

    return df


def add_ml_proba(df: pd.DataFrame, model, feature_columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    positive_index = get_positive_class_index(model)
    proba = model.predict_proba(df[feature_columns])[:, positive_index]
    df["ml_proba_1"] = proba

    return df


def filter_evaluation_period(df: pd.DataFrame) -> pd.DataFrame:
    if not 0 <= BACKTEST_EVAL_START_RATIO < 1:
        raise ValueError(
            "BACKTEST_EVAL_START_RATIO must be greater than or equal to 0 "
            "and less than 1."
        )

    start_index = int(len(df) * BACKTEST_EVAL_START_RATIO)
    return df.iloc[start_index:].copy().reset_index(drop=True)


def run_backtest(
    df: pd.DataFrame,
    use_ml_filter: bool,
    ml_proba_threshold: float = ML_PROBA_THRESHOLD,
) -> dict:
    cash = INITIAL_CASH
    qty = 0.0
    entry_price = 0.0

    buy_count = 0
    sell_count = 0
    take_profit_count = 0
    stop_loss_count = 0
    sma_exit_count = 0
    ml_blocked_buy_count = 0

    realized_pnl = 0.0
    wins = 0
    losses = 0

    trades = []

    for _, row in df.iterrows():
        close = float(row["close"])

        if qty > 0:
            change_rate = (close - entry_price) / entry_price

            sell_reason = None

            if change_rate >= TAKE_PROFIT_RATE:
                sell_reason = "TAKE_PROFIT"
                take_profit_count += 1
            elif change_rate <= -STOP_LOSS_RATE:
                sell_reason = "STOP_LOSS"
                stop_loss_count += 1
            elif (
                row["prev_diff"] >= 0
                and row["curr_diff"] < 0
            ) or row["rsi_14"] < RSI_SELL_THRESHOLD or (
                row["macd"] < row["macd_signal"] and row["macd_hist"] < 0
            ):
                sell_reason = "SMA_RSI_MACD_SELL"
                sma_exit_count += 1

            if sell_reason:
                gross = qty * close
                fee = gross * FEE_RATE
                pnl = (close - entry_price) * qty - fee

                cash += gross - fee
                realized_pnl += pnl
                sell_count += 1

                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                trades.append(
                    {
                        "side": "SELL",
                        "time": row.get("open_time"),
                        "price": close,
                        "pnl": pnl,
                        "reason": sell_reason,
                    }
                )

                qty = 0.0
                entry_price = 0.0

                continue

        if qty == 0:
            buy_signal = (
                row["prev_diff"] <= 0
                and row["curr_diff"] > 0
                and row["rsi_14"] > RSI_BUY_THRESHOLD
                and row["macd"] > row["macd_signal"]
                and row["macd_hist"] > 0
            )

            if buy_signal:
                if use_ml_filter and row["ml_proba_1"] < ml_proba_threshold:
                    ml_blocked_buy_count += 1
                    continue

                notional = min(TRADE_NOTIONAL, cash)
                fee = notional * FEE_RATE
                usable = notional - fee
                qty = usable / close
                entry_price = close
                cash -= notional

                buy_count += 1

                trades.append(
                    {
                        "side": "BUY",
                        "time": row.get("open_time"),
                        "price": close,
                        "pnl": 0.0,
                        "reason": (
                            "SMA_RSI_MACD_BUY_ML"
                            if use_ml_filter
                            else "SMA_RSI_MACD_BUY"
                        ),
                    }
                )

    if qty > 0:
        close = float(df.iloc[-1]["close"])
        gross = qty * close
        fee = gross * FEE_RATE
        pnl = (close - entry_price) * qty - fee

        cash += gross - fee
        realized_pnl += pnl
        sell_count += 1

        if pnl > 0:
            wins += 1
        else:
            losses += 1

        trades.append(
            {
                "side": "SELL",
                "time": df.iloc[-1].get("open_time"),
                "price": close,
                "pnl": pnl,
                "reason": "FORCED_CLOSE",
            }
        )

        qty = 0.0

    final_equity = cash
    total_return = ((final_equity - INITIAL_CASH) / INITIAL_CASH) * 100
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0

    return {
        "use_ml_filter": use_ml_filter,
        "final_equity": final_equity,
        "total_return": total_return,
        "realized_pnl": realized_pnl,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "take_profit_count": take_profit_count,
        "stop_loss_count": stop_loss_count,
        "sma_exit_count": sma_exit_count,
        "ml_blocked_buy_count": ml_blocked_buy_count,
        "win_rate": win_rate,
        "trades": trades,
    }


def print_result(result: dict) -> None:
    title = "SMA/RSI/MACD + ML Filter" if result["use_ml_filter"] else "SMA/RSI/MACD Only"

    print("\n===================================")
    print(title)
    print("===================================")
    print(f"Final equity: {result['final_equity']:.2f}")
    print(f"Total return: {result['total_return']:.2f}%")
    print(f"Realized PnL: {result['realized_pnl']:.2f}")
    print(f"BUY/SELL: {result['buy_count']}/{result['sell_count']}")
    print(f"TP/SL/SMA exits: {result['take_profit_count']}/{result['stop_loss_count']}/{result['sma_exit_count']}")
    print(f"ML blocked buys: {result['ml_blocked_buy_count']}")
    print(f"Win rate: {result['win_rate']:.2f}%")
    print(f"Total trades: {len(result['trades'])}")

    print("Recent trades:")
    for trade in result["trades"][-10:]:
        print(
            f"{trade['time']} {trade['side']} "
            f"price={trade['price']:.2f} "
            f"pnl={trade['pnl']:.2f} "
            f"reason={trade['reason']}"
        )


def main() -> None:
    bundle = load_model_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    df = load_dataset(feature_columns)
    df = add_sma_cross_columns(df)
    df = add_ml_proba(df, model, feature_columns)
    df = df.dropna().reset_index(drop=True)
    eval_df = filter_evaluation_period(df)

    if eval_df.empty:
        raise RuntimeError("Evaluation dataset is empty.")

    print("===================================")
    print("Backtest ML Filter")
    print("===================================")
    print(f"dataset: {DATASET_PATH}")
    print(f"model: {MODEL_PATH}")
    print(f"model_name: {ML_MODEL_NAME}")
    print(f"rows: {len(df)}")
    print(f"eval_start_ratio: {BACKTEST_EVAL_START_RATIO}")
    print(f"eval_rows: {len(eval_df)}")
    if "open_time" in eval_df.columns:
        print(f"eval_range: {eval_df['open_time'].iloc[0]} -> {eval_df['open_time'].iloc[-1]}")
    print(f"ml_proba_threshold: {ML_PROBA_THRESHOLD}")
    print(f"take_profit_rate: {TAKE_PROFIT_RATE}")
    print(f"stop_loss_rate: {STOP_LOSS_RATE}")
    print("===================================")

    base_result = run_backtest(eval_df, use_ml_filter=False)
    ml_result = run_backtest(eval_df, use_ml_filter=True)

    print_result(base_result)
    print_result(ml_result)


if __name__ == "__main__":
    main()
