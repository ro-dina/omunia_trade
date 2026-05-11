import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1",
    "log_return_1",
    "sma_5_gap",
    "sma_10_gap",
    "sma_30_gap",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "high_low_range",
    "open_close_range",
    "volume",
]


def normalize_candles_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["market_id", "open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"])
    return df


def add_ml_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def build_latest_feature_row(candles: list[dict], feature_columns: list[str]) -> pd.DataFrame | None:
    if len(candles) < 50:
        return None

    df = normalize_candles_frame(pd.DataFrame(candles))
    df = add_ml_indicators(df)
    latest = df.iloc[-1]

    if latest[feature_columns].isna().any():
        return None

    return pd.DataFrame([{column: latest[column] for column in feature_columns}])
