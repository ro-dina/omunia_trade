from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import math
import os
import pandas as pd
import requests
from dotenv import load_dotenv

from app.db.supabase_client import supabase

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
# =========================================
# Paper strategy settings
# =========================================
EXCHANGE = os.getenv("TRADE_EXCHANGE", "bybit")
SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
MARKET_TYPE = os.getenv("TRADE_MARKET_TYPE", "linear")
TIMEFRAME = os.getenv("TRADE_TIMEFRAME", "5m")

DEFAULT_SOURCE = "bybit-mainnet-public" if TIMEFRAME == "1m" else f"bybit-mainnet-public-{TIMEFRAME}"
SOURCE = os.getenv("TRADE_SOURCE", DEFAULT_SOURCE)

SHORT_PERIOD = 5
LONG_PERIOD = 30
TIMEFRAME_SECONDS = 60 if TIMEFRAME == "1m" else 300
FETCH_CANDLE_LIMIT = max(LONG_PERIOD + 20, 80)


RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 60.0
RSI_SELL_THRESHOLD = 35.0

MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
USE_MACD_FILTER = True


TAKE_PROFIT_RATE = float(os.getenv("TAKE_PROFIT_RATE", "0.010"))  # +1.0%
STOP_LOSS_RATE = float(os.getenv("STOP_LOSS_RATE", "0.008"))      # -0.8%

USE_ML_FILTER = os.getenv("USE_ML_FILTER", "true").lower() == "true"
ML_MODEL_NAME = os.getenv("ML_MODEL_NAME", "RandomForest")
ML_MODEL_PATH = Path(
    os.getenv(
        "ML_MODEL_PATH",
        f"data/models/{SYMBOL}_{TIMEFRAME}_{ML_MODEL_NAME}.joblib",
    )
)
ML_PROBA_THRESHOLD = float(os.getenv("ML_PROBA_THRESHOLD", "0.60"))
_ml_model_bundle: Optional[dict] = None

STRATEGY_NAME = (
    f"{SYMBOL.lower()}_"
    f"sma{SHORT_PERIOD}_sma{LONG_PERIOD}_cross_"
    f"rsi{int(RSI_BUY_THRESHOLD)}_{int(RSI_SELL_THRESHOLD)}_"
    f"{TIMEFRAME}_"
    f"ml{int(ML_PROBA_THRESHOLD * 100) if USE_ML_FILTER else 0}"
)

INITIAL_CASH = 10_000.0

# 資産の何%を1回の取引に使うか
RISK_PER_TRADE = 0.1

FEE_RATE = 0.0006  # 0.06%想定
# FEE_RATE = 0


def to_float(value) -> float:
    return float(value)

def send_discord_message(message: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return

    try:
        requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message},
            timeout=10,
        )
    except Exception as e:
        print("Discord notification failed:", e)


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
        raise RuntimeError(
            f"markets に {SYMBOL} {TIMEFRAME} が見つかりません。"
        )

    return result.data[0]["id"]


def fetch_recent_candles(market_id: str, limit: int = FETCH_CANDLE_LIMIT) -> list[dict]:
    result = (
        supabase.table("candles")
        .select("*")
        .eq("market_id", market_id)
        .eq("source", SOURCE)
        .order("open_time", desc=True)
        .limit(limit)
        .execute()
    )

    return list(reversed(result.data or []))


def calculate_sma(candles: list[dict], period: int) -> list[Optional[float]]:
    values: list[Optional[float]] = []

    for i in range(len(candles)):
        if i < period - 1:
            values.append(None)
            continue

        window = candles[i - period + 1 : i + 1]
        avg = sum(to_float(c["close"]) for c in window) / period
        values.append(avg)

    return values


def calculate_rsi(candles: list[dict], period: int = RSI_PERIOD) -> list[Optional[float]]:
    values: list[Optional[float]] = [None] * len(candles)

    if len(candles) <= period:
        return values

    gains: list[float] = []
    losses: list[float] = []

    for i in range(1, period + 1):
        change = to_float(candles[i]["close"]) - to_float(candles[i - 1]["close"])
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        values[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        values[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, len(candles)):
        change = to_float(candles[i]["close"]) - to_float(candles[i - 1]["close"])
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

        if avg_loss == 0:
            values[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            values[i] = 100.0 - (100.0 / (1.0 + rs))

    return values


# ===========================
# MACD Calculation Functions
# ===========================

def calculate_ema(values: list[float], period: int) -> list[Optional[float]]:
    result: list[Optional[float]] = [None] * len(values)

    if len(values) < period:
        return result

    multiplier = 2 / (period + 1)
    sma = sum(values[:period]) / period
    result[period - 1] = sma

    prev_ema = sma

    for i in range(period, len(values)):
        ema = (values[i] - prev_ema) * multiplier + prev_ema
        result[i] = ema
        prev_ema = ema

    return result


def calculate_macd(
    candles: list[dict],
    fast_period: int = MACD_FAST_PERIOD,
    slow_period: int = MACD_SLOW_PERIOD,
    signal_period: int = MACD_SIGNAL_PERIOD,
) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    closes = [to_float(candle["close"]) for candle in candles]

    fast_ema = calculate_ema(closes, fast_period)
    slow_ema = calculate_ema(closes, slow_period)

    macd_line: list[Optional[float]] = [None] * len(candles)

    for i in range(len(candles)):
        if fast_ema[i] is None or slow_ema[i] is None:
            continue

        macd_line[i] = fast_ema[i] - slow_ema[i]

    compact_macd = [value for value in macd_line if value is not None]
    compact_signal = calculate_ema(compact_macd, signal_period)

    signal_line: list[Optional[float]] = [None] * len(candles)

    compact_index = 0

    for i in range(len(candles)):
        if macd_line[i] is None:
            continue

        signal_line[i] = compact_signal[compact_index]
        compact_index += 1

    histogram: list[Optional[float]] = [None] * len(candles)

    for i in range(len(candles)):
        if macd_line[i] is None or signal_line[i] is None:
            continue

        histogram[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, histogram


# ===================== ML Model Integration =======================

def load_ml_model_bundle() -> Optional[dict]:
    global _ml_model_bundle

    if not USE_ML_FILTER:
        return None

    if _ml_model_bundle is not None:
        return _ml_model_bundle

    if not ML_MODEL_PATH.exists():
        print(f"ML model not found: {ML_MODEL_PATH}")
        return None

    _ml_model_bundle = joblib.load(ML_MODEL_PATH)
    return _ml_model_bundle


def get_positive_class_index(model, positive_label: int = 1) -> int:
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif hasattr(model, "named_steps") and "model" in model.named_steps:
        classes = list(model.named_steps["model"].classes_)
    else:
        raise RuntimeError("Could not read model classes.")

    if positive_label not in classes:
        raise RuntimeError(
            f"Positive label {positive_label} not found in classes: {classes}"
        )

    return classes.index(positive_label)


def calculate_latest_ml_features(candles: list[dict]) -> Optional[dict]:
    if len(candles) < 50:
        return None

    closes = [to_float(candle["close"]) for candle in candles]
    latest = candles[-1]
    prev = candles[-2]

    close = to_float(latest["close"])
    prev_close = to_float(prev["close"])
    open_price = to_float(latest["open"])
    high = to_float(latest["high"])
    low = to_float(latest["low"])
    volume = to_float(latest["volume"])

    if close <= 0 or prev_close <= 0 or open_price <= 0:
        return None

    sma_5 = sum(closes[-5:]) / 5
    sma_10 = sum(closes[-10:]) / 10
    sma_30 = sum(closes[-30:]) / 30

    rsi_values = calculate_rsi(candles)
    macd_line, macd_signal, macd_histogram = calculate_macd(candles)

    rsi_14 = rsi_values[-1]
    macd = macd_line[-1]
    macd_sig = macd_signal[-1]
    macd_hist = macd_histogram[-1]

    if None in (rsi_14, macd, macd_sig, macd_hist):
        return None

    return {
        "return_1": (close - prev_close) / prev_close,
        "log_return_1": math.log(close) - math.log(prev_close),
        "sma_5_gap": (close - sma_5) / close,
        "sma_10_gap": (close - sma_10) / close,
        "sma_30_gap": (close - sma_30) / close,
        "rsi_14": rsi_14,
        "macd": macd,
        "macd_signal": macd_sig,
        "macd_hist": macd_hist,
        "high_low_range": (high - low) / close,
        "open_close_range": (close - open_price) / open_price,
        "volume": volume,
    }


def predict_ml_buy_probability(candles: list[dict]) -> Optional[float]:
    bundle = load_ml_model_bundle()

    if bundle is None:
        return None

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    features = calculate_latest_ml_features(candles)

    if features is None:
        return None

    row = pd.DataFrame([{column: features[column] for column in feature_columns}])
    positive_index = get_positive_class_index(model, positive_label=1)
    proba = model.predict_proba(row)[0][positive_index]

    return float(proba)


def detect_latest_sma_cross(candles: list[dict]) -> Optional[str]:
    if len(candles) < LONG_PERIOD + 1:
        return None

    short_sma = calculate_sma(candles, SHORT_PERIOD)
    long_sma = calculate_sma(candles, LONG_PERIOD)
    rsi_values = calculate_rsi(candles)
    macd_line, macd_signal, macd_histogram = calculate_macd(candles)

    i = len(candles) - 1
    prev_i = i - 1

    prev_time = datetime.fromisoformat(candles[prev_i]["open_time"].replace("Z", "+00:00"))
    curr_time = datetime.fromisoformat(candles[i]["open_time"].replace("Z", "+00:00"))

    # 欠損があるところでは判定しない
    if (curr_time - prev_time).total_seconds() != TIMEFRAME_SECONDS:
        return None

    prev_short = short_sma[prev_i]
    prev_long = long_sma[prev_i]
    curr_short = short_sma[i]
    curr_long = long_sma[i]

    if None in (prev_short, prev_long, curr_short, curr_long):
        return None

    curr_rsi = rsi_values[i]

    if curr_rsi is None:
        return None

    curr_macd = macd_line[i]
    curr_macd_signal = macd_signal[i]
    curr_macd_histogram = macd_histogram[i]

    if USE_MACD_FILTER and (
        curr_macd is None
        or curr_macd_signal is None
        or curr_macd_histogram is None
    ):
        return None

    prev_diff = prev_short - prev_long
    curr_diff = curr_short - curr_long

    macd_buy_ok = (
        not USE_MACD_FILTER
        or (
            curr_macd is not None
            and curr_macd_signal is not None
            and curr_macd_histogram is not None
            and curr_macd > curr_macd_signal
            and curr_macd_histogram > 0
        )
    )

    macd_sell_ok = (
        USE_MACD_FILTER
        and curr_macd is not None
        and curr_macd_signal is not None
        and curr_macd_histogram is not None
        and curr_macd < curr_macd_signal
        and curr_macd_histogram < 0
    )

    if (
        prev_diff <= 0
        and curr_diff > 0
        and curr_rsi > RSI_BUY_THRESHOLD
        and macd_buy_ok
    ):
        return "BUY"

    if (
        (prev_diff >= 0 and curr_diff < 0)
        or curr_rsi < RSI_SELL_THRESHOLD
        or macd_sell_ok
    ):
        return "SELL"

    return None

def save_signal(
    market_id: str,
    candle: dict,
    signal_type: str,
    reason: str,
) -> None:
    signal_time = candle["open_time"]
    price = to_float(candle["close"])

    supabase.table("signals").upsert(
        {
            "market_id": market_id,
            "strategy_name": STRATEGY_NAME,
            "signal_time": signal_time,
            "signal_type": signal_type,
            "price": price,
            "reason": reason,
            "meta": {
                "short_period": SHORT_PERIOD,
                "long_period": LONG_PERIOD,
                "rsi_period": RSI_PERIOD,
                "rsi_buy_threshold": RSI_BUY_THRESHOLD,
                "rsi_sell_threshold": RSI_SELL_THRESHOLD,
                "macd_fast_period": MACD_FAST_PERIOD,
                "macd_slow_period": MACD_SLOW_PERIOD,
                "macd_signal_period": MACD_SIGNAL_PERIOD,
                "use_macd_filter": USE_MACD_FILTER,
                "timeframe": TIMEFRAME,
                "source": SOURCE,
                "take_profit_rate": TAKE_PROFIT_RATE,
                "stop_loss_rate": STOP_LOSS_RATE,
                "use_ml_filter": USE_ML_FILTER,
                "ml_model_name": ML_MODEL_NAME,
                "ml_model_path": str(ML_MODEL_PATH),
                "ml_proba_threshold": ML_PROBA_THRESHOLD,
            },
        },
        on_conflict="market_id,strategy_name,signal_time",
    ).execute()

def get_latest_portfolio() -> dict:
    market_id = get_market_id()

    result = (
        supabase.table("portfolio_snapshots")
        .select("*")
        .eq("market_id", market_id)
        .order("snapshot_time", desc=True)
        .limit(1)
        .execute()
    )

    if not result.data:
        return {
            "cash_balance": INITIAL_CASH,
            "asset_value": 0.0,
            "total_equity": INITIAL_CASH,
            "free_balance": INITIAL_CASH,
            "used_margin": 0.0,
        }

    return result.data[0]


def get_open_position(market_id: str) -> Optional[dict]:
    result = (
        supabase.table("positions")
        .select("*")
        .eq("market_id", market_id)
        .eq("status", "open")
        .limit(1)
        .execute()
    )

    if not result.data:
        return None

    return result.data[0]


def order_already_exists(order_key: str) -> bool:
    result = (
        supabase.table("orders")
        .select("id")
        .eq("exchange_order_id", order_key)
        .limit(1)
        .execute()
    )

    return bool(result.data)


def get_tp_sl_exit_signal(position: dict, candle: dict) -> Optional[str]:
    price = to_float(candle["close"])
    entry_price = to_float(position["entry_price"])

    take_profit_price = entry_price * (1 + TAKE_PROFIT_RATE)
    stop_loss_price = entry_price * (1 - STOP_LOSS_RATE)

    if price >= take_profit_price:
        return "TAKE_PROFIT"

    if price <= stop_loss_price:
        return "STOP_LOSS"

    return None


def create_portfolio_snapshot(
    market_id: str,
    cash_balance: float,
    price: Optional[float] = None,
    asset_value: Optional[float] = None,
    used_margin: float = 0.0,
) -> None:
    """
    Create a portfolio snapshot.

    If a long position is open, asset_value is recalculated from the open
    position and the latest price. This prevents abnormal snapshots where
    cash is reduced after BUY but asset_value is accidentally recorded as 0.
    """
    position = get_open_position(market_id)

    if position:
        qty = to_float(position["qty"])
        current_price = price

        if current_price is None:
            current_price = to_float(position.get("current_price") or position["entry_price"])

        asset_value = qty * current_price
    elif asset_value is None:
        asset_value = 0.0

    total_equity = cash_balance + asset_value

    supabase.table("portfolio_snapshots").insert(
        {
            "market_id": market_id,
            "cash_balance": cash_balance,
            "asset_value": asset_value,
            "total_equity": total_equity,
            "used_margin": used_margin,
            "free_balance": cash_balance,
            "snapshot_time": datetime.now(timezone.utc).isoformat(),
        }
    ).execute()


def execute_buy(market_id: str, candle: dict) -> None:
    price = to_float(candle["close"])
    open_time = candle["open_time"]
    order_key = f"paper-{STRATEGY_NAME}-buy-{open_time}"

    if order_already_exists(order_key):
        print("BUY already processed.")
        return

    if get_open_position(market_id):
        print("Open position already exists. BUY skipped.")
        return

    portfolio = get_latest_portfolio()
    cash_balance = to_float(portfolio["cash_balance"])

    trade_notional = cash_balance * RISK_PER_TRADE

    if trade_notional <= 0:
        print("Trade notional is zero. BUY skipped.")
        return

    fee = trade_notional * FEE_RATE
    total_cost = trade_notional + fee

    if cash_balance < total_cost:
        print("Not enough paper cash. BUY skipped.")
        return

    qty = trade_notional / price
    new_cash = cash_balance - total_cost
    asset_value = qty * price

    order = (
        supabase.table("orders")
        .insert(
            {
                "market_id": market_id,
                "side": "buy",
                "order_type": "market",
                "qty": qty,
                "requested_price": price,
                "filled_price": price,
                "status": "filled",
                "is_paper": True,
                "exchange_order_id": order_key,
                "fee": fee,
            }
        )
        .execute()
    )

    supabase.table("positions").insert(
        {
            "market_id": market_id,
            "side": "long",
            "qty": qty,
            "entry_price": price,
            "current_price": price,
            "unrealized_pnl": 0,
            "realized_pnl": 0,
            "status": "open",
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
    ).execute()

    create_portfolio_snapshot(
        market_id=market_id,
        cash_balance=new_cash,
        price=price,
        asset_value=asset_value,
        used_margin=0.0,
    )

    print(
        f"BUY executed: price={price}, qty={qty}, "
        f"notional={trade_notional}, fee={fee}"
    )

    send_discord_message(
        f"🟢 PAPER BUY\n"
        f"Price: {price:.2f}\n"
        f"Qty: {qty:.6f}\n"
        f"Fee: {fee:.4f}\n"
        f"TP: {TAKE_PROFIT_RATE * 100:.1f}% / SL: {STOP_LOSS_RATE * 100:.1f}%\n"
        f"ML: {USE_ML_FILTER} threshold={ML_PROBA_THRESHOLD:.2f}\n"
        f"Strategy: {STRATEGY_NAME}"
    )


def execute_sell(market_id: str, candle: dict, reason: str = "SMA_CROSS_SELL") -> None:
    price = to_float(candle["close"])
    open_time = candle["open_time"]
    order_key = f"paper-{STRATEGY_NAME}-sell-{reason.lower()}-{open_time}"

    if order_already_exists(order_key):
        print("SELL already processed.")
        return

    position = get_open_position(market_id)

    if not position:
        print("No open position. SELL skipped.")
        return

    qty = to_float(position["qty"])
    entry_price = to_float(position["entry_price"])

    gross_value = qty * price
    fee = gross_value * FEE_RATE
    realized_pnl = (price - entry_price) * qty - fee

    portfolio = get_latest_portfolio()
    cash_balance = to_float(portfolio["cash_balance"])
    new_cash = cash_balance + gross_value - fee

    supabase.table("orders").insert(
        {
            "market_id": market_id,
            "side": "sell",
            "order_type": "market",
            "qty": qty,
            "requested_price": price,
            "filled_price": price,
            "status": "filled",
            "is_paper": True,
            "exchange_order_id": order_key,
            "fee": fee,
        }
    ).execute()

    supabase.table("positions").update(
        {
            "current_price": price,
            "unrealized_pnl": 0,
            "realized_pnl": realized_pnl,
            "status": "closed",
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ).eq("id", position["id"]).execute()

    create_portfolio_snapshot(
        market_id=market_id,
        cash_balance=new_cash,
        price=price,
        asset_value=0.0,
        used_margin=0.0,
    )

    print(
        f"SELL executed: price={price}, qty={qty}, "
        f"pnl={realized_pnl}, fee={fee}, reason={reason}"
    )

    send_discord_message(
        f"🔴 PAPER SELL ({reason})\n"
        f"Price: {price:.2f}\n"
        f"Qty: {qty:.6f}\n"
        f"PnL: {realized_pnl:.2f}\n"
        f"Fee: {fee:.4f}\n"
        f"Strategy: {STRATEGY_NAME}"
    )


def update_mark_to_market(market_id: str, candle: dict) -> None:
    position = get_open_position(market_id)
    portfolio = get_latest_portfolio()

    cash_balance = to_float(portfolio["cash_balance"])
    price = to_float(candle["close"])

    # If the latest snapshot was abnormal, e.g. cash after BUY but asset_value=0,
    # rebuild asset_value from the actual open position below.

    if not position:
        create_portfolio_snapshot(
            market_id=market_id,
            cash_balance=cash_balance,
            price=price,
            asset_value=0.0,
            used_margin=0.0,
        )
        return

    qty = to_float(position["qty"])
    entry_price = to_float(position["entry_price"])

    asset_value = qty * price
    unrealized_pnl = (price - entry_price) * qty

    supabase.table("positions").update(
        {
            "current_price": price,
            "unrealized_pnl": unrealized_pnl,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ).eq("id", position["id"]).execute()

    create_portfolio_snapshot(
        market_id=market_id,
        cash_balance=cash_balance,
        price=price,
        asset_value=asset_value,
        used_margin=0.0,
    )


def run_paper_strategy() -> None:
    market_id = get_market_id()
    candles = fetch_recent_candles(market_id)

    if len(candles) < LONG_PERIOD + 1:
        print(f"Not enough candles. required={LONG_PERIOD + 1}, actual={len(candles)}")
        return

    latest_candle = candles[-1]
    signal = detect_latest_sma_cross(candles)
    ml_buy_probability = predict_ml_buy_probability(candles)

    print("latest candle:", latest_candle["open_time"], latest_candle["close"])
    print("signal:", signal)
    print("ml_buy_probability:", ml_buy_probability)

    open_position = get_open_position(market_id)

    if open_position:
        exit_signal = get_tp_sl_exit_signal(open_position, latest_candle)

        if exit_signal:
            save_signal(
                market_id=market_id,
                candle=latest_candle,
                signal_type="SELL",
                reason=exit_signal,
            )
            execute_sell(market_id, latest_candle, reason=exit_signal)
            return

    if signal == "BUY":
        if USE_ML_FILTER:
            if ml_buy_probability is None:
                save_signal(
                    market_id=market_id,
                    candle=latest_candle,
                    signal_type="HOLD",
                    reason="BUY blocked because ML probability is unavailable",
                )
                update_mark_to_market(market_id, latest_candle)
                print("BUY blocked: ML probability is unavailable.")
                return

            if ml_buy_probability < ML_PROBA_THRESHOLD:
                save_signal(
                    market_id=market_id,
                    candle=latest_candle,
                    signal_type="HOLD",
                    reason=(
                        f"BUY blocked by ML filter: "
                        f"proba={ml_buy_probability:.4f} < {ML_PROBA_THRESHOLD:.4f}"
                    ),
                )
                update_mark_to_market(market_id, latest_candle)
                print(
                    f"BUY blocked by ML filter: "
                    f"proba={ml_buy_probability:.4f} < {ML_PROBA_THRESHOLD:.4f}"
                )
                return

        save_signal(
            market_id=market_id,
            candle=latest_candle,
            signal_type="BUY",
            reason=(
                f"SMA{SHORT_PERIOD} crossed above SMA{LONG_PERIOD} "
                f"and RSI > {RSI_BUY_THRESHOLD} "
                f"and ML proba={ml_buy_probability}"
            ),
        )
        execute_buy(market_id, latest_candle)
        return

    if signal == "SELL":
        save_signal(
            market_id=market_id,
            candle=latest_candle,
            signal_type="SELL",
            reason=(
                f"SMA{SHORT_PERIOD} crossed below SMA{LONG_PERIOD} "
                f"or RSI < {RSI_SELL_THRESHOLD} or MACD bearish"
            ),
        )
        execute_sell(market_id, latest_candle, reason="SMA_CROSS_SELL")
        return

    save_signal(
        market_id=market_id,
        candle=latest_candle,
        signal_type="HOLD",
        reason="No SMA/RSI/MACD signal",
    )

    update_mark_to_market(market_id, latest_candle)
    print("No trade. Portfolio snapshot updated.")