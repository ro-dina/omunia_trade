from datetime import datetime, timezone
from typing import Optional

from app.db.supabase_client import supabase

# =========================================
# Paper strategy settings
# =========================================
EXCHANGE = "bybit"
SYMBOL = "BTCUSDT"
MARKET_TYPE = "linear"
TIMEFRAME = "5m"
SOURCE = "bybit-mainnet-public-5m"

SHORT_PERIOD = 30
LONG_PERIOD = 150
TIMEFRAME_SECONDS = 300
FETCH_CANDLE_LIMIT = LONG_PERIOD + 20

STRATEGY_NAME = f"sma{SHORT_PERIOD}_sma{LONG_PERIOD}_cross_{TIMEFRAME}"

TAKE_PROFIT_RATE = 0.01   # +1.0%
STOP_LOSS_RATE = 0.003    # -0.3%

INITIAL_CASH = 10_000.0

# 資産の何%を1回の取引に使うか
RISK_PER_TRADE = 0.1

FEE_RATE = 0.0006  # 0.06%想定
# FEE_RATE = 0


def to_float(value) -> float:
    return float(value)


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
        raise RuntimeError("markets に BTCUSDT が見つかりません。")

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


def detect_latest_sma_cross(candles: list[dict]) -> Optional[str]:
    if len(candles) < LONG_PERIOD + 1:
        return None

    short_sma = calculate_sma(candles, SHORT_PERIOD)
    long_sma = calculate_sma(candles, LONG_PERIOD)

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

    prev_diff = prev_short - prev_long
    curr_diff = curr_short - curr_long

    if prev_diff <= 0 and curr_diff > 0:
        return "BUY"

    if prev_diff >= 0 and curr_diff < 0:
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
                "timeframe": TIMEFRAME,
                "source": SOURCE,
                "take_profit_rate": TAKE_PROFIT_RATE,
                "stop_loss_rate": STOP_LOSS_RATE,
            },
        },
        on_conflict="market_id,strategy_name,signal_time",
    ).execute()

def get_latest_portfolio() -> dict:
    result = (
        supabase.table("portfolio_snapshots")
        .select("*")
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
    cash_balance: float,
    asset_value: float,
    used_margin: float = 0.0,
) -> None:
    total_equity = cash_balance + asset_value

    supabase.table("portfolio_snapshots").insert(
        {
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
        cash_balance=new_cash,
        asset_value=asset_value,
        used_margin=0.0,
    )

    print(
        f"BUY executed: price={price}, qty={qty}, "
        f"notional={trade_notional}, fee={fee}"
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
        cash_balance=new_cash,
        asset_value=0.0,
        used_margin=0.0,
    )

    print(
        f"SELL executed: price={price}, qty={qty}, "
        f"pnl={realized_pnl}, fee={fee}, reason={reason}"
    )


def update_mark_to_market(market_id: str, candle: dict) -> None:
    position = get_open_position(market_id)
    portfolio = get_latest_portfolio()

    cash_balance = to_float(portfolio["cash_balance"])
    price = to_float(candle["close"])

    if not position:
        create_portfolio_snapshot(
            cash_balance=cash_balance,
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
        cash_balance=cash_balance,
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

    print("latest candle:", latest_candle["open_time"], latest_candle["close"])
    print("signal:", signal)

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
        save_signal(
            market_id=market_id,
            candle=latest_candle,
            signal_type="BUY",
            reason=f"SMA{SHORT_PERIOD} crossed above SMA{LONG_PERIOD}",
        )
        execute_buy(market_id, latest_candle)
        return

    if signal == "SELL":
        save_signal(
            market_id=market_id,
            candle=latest_candle,
            signal_type="SELL",
            reason=f"SMA{SHORT_PERIOD} crossed below SMA{LONG_PERIOD}",
        )
        execute_sell(market_id, latest_candle, reason="SMA_CROSS_SELL")
        return

    save_signal(
        market_id=market_id,
        candle=latest_candle,
        signal_type="HOLD",
        reason="No SMA cross",
    )

    update_mark_to_market(market_id, latest_candle)
    print("No trade. Portfolio snapshot updated.")