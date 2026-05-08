from datetime import datetime
from typing import Optional

from app.db.supabase_client import supabase
from app.services.paper_trade_service import (
    EXCHANGE,
    SYMBOL,
    MARKET_TYPE,
    TIMEFRAME,
    SOURCE,
    TRADE_NOTIONAL,
    FEE_RATE,
    INITIAL_CASH,
    get_market_id,
    to_float,
)


def fetch_backtest_candles(market_id: str, limit: int = 1000) -> list[dict]:
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


def is_continuous(prev_candle: dict, curr_candle: dict) -> bool:
    prev_time = datetime.fromisoformat(prev_candle["open_time"].replace("Z", "+00:00"))
    curr_time = datetime.fromisoformat(curr_candle["open_time"].replace("Z", "+00:00"))

    return (curr_time - prev_time).total_seconds() == 60


def run_sma_cross_backtest(
    short_period: int = 20,
    long_period: int = 50,
    limit: int = 1000,
) -> None:
    market_id = get_market_id()
    candles = fetch_backtest_candles(market_id, limit=limit)

    if len(candles) < 60:
        print("Not enough candles for backtest.")
        return

    short_sma = calculate_sma(candles, short_period)
    long_sma = calculate_sma(candles, long_period)

    cash = INITIAL_CASH
    position_qty = 0.0
    entry_price: Optional[float] = None

    trades: list[dict] = []
    equity_curve: list[dict] = []

    for i in range(1, len(candles)):
        candle = candles[i]
        prev_candle = candles[i - 1]

        price = to_float(candle["close"])

        asset_value = position_qty * price
        total_equity = cash + asset_value

        equity_curve.append(
            {
                "time": candle["open_time"],
                "equity": total_equity,
            }
        )

        if not is_continuous(prev_candle, candle):
            continue

        prev_short = short_sma[i - 1]
        prev_long = long_sma[i - 1]
        curr_short = short_sma[i]
        curr_long = long_sma[i]

        if None in (prev_short, prev_long, curr_short, curr_long):
            continue

        prev_diff = prev_short - prev_long
        curr_diff = curr_short - curr_long

        buy_signal = prev_diff <= 0 and curr_diff > 0
        sell_signal = prev_diff >= 0 and curr_diff < 0

        if buy_signal and position_qty == 0:
            fee = TRADE_NOTIONAL * FEE_RATE
            total_cost = TRADE_NOTIONAL + fee

            if cash >= total_cost:
                qty = TRADE_NOTIONAL / price
                cash -= total_cost
                position_qty = qty
                entry_price = price

                trades.append(
                    {
                        "time": candle["open_time"],
                        "side": "BUY",
                        "price": price,
                        "qty": qty,
                        "fee": fee,
                        "pnl": 0.0,
                    }
                )

        elif sell_signal and position_qty > 0 and entry_price is not None:
            gross_value = position_qty * price
            fee = gross_value * FEE_RATE
            pnl = (price - entry_price) * position_qty - fee

            cash += gross_value - fee

            trades.append(
                {
                    "time": candle["open_time"],
                    "side": "SELL",
                    "price": price,
                    "qty": position_qty,
                    "fee": fee,
                    "pnl": pnl,
                }
            )

            position_qty = 0.0
            entry_price = None

    final_price = to_float(candles[-1]["close"])
    final_asset_value = position_qty * final_price
    final_equity = cash + final_asset_value

    realized_pnl = sum(t["pnl"] for t in trades if t["side"] == "SELL")
    total_return = ((final_equity - INITIAL_CASH) / INITIAL_CASH) * 100

    buy_count = len([t for t in trades if t["side"] == "BUY"])
    sell_count = len([t for t in trades if t["side"] == "SELL"])

    wins = len([t for t in trades if t["side"] == "SELL" and t["pnl"] > 0])
    losses = len([t for t in trades if t["side"] == "SELL" and t["pnl"] <= 0])
    closed_trades = wins + losses
    win_rate = (wins / closed_trades * 100) if closed_trades > 0 else 0

    print("===================================")
    print(f"SMA{short_period} / SMA{long_period} Backtest Result")
    print("===================================")
    print(f"Symbol: {EXCHANGE} {SYMBOL} {MARKET_TYPE} {TIMEFRAME}")
    print(f"Candles: {len(candles)}")
    print(f"Initial cash: {INITIAL_CASH:.2f} USDT")
    print(f"Final equity: {final_equity:.2f} USDT")
    print(f"Total return: {total_return:.2f}%")
    print(f"Realized PnL: {realized_pnl:.2f} USDT")
    print(f"Open position value: {final_asset_value:.2f} USDT")
    print(f"BUY count: {buy_count}")
    print(f"SELL count: {sell_count}")
    print(f"Win rate: {win_rate:.2f}%")
    print("===================================")

    print("\nRecent trades:")
    for trade in trades[-10:]:
        print(
            trade["time"],
            trade["side"],
            f"price={trade['price']:.2f}",
            f"qty={trade['qty']:.6f}",
            f"fee={trade['fee']:.4f}",
            f"pnl={trade['pnl']:.2f}",
        )

def evaluate_sma_cross_backtest(
    short_period: int,
    long_period: int,
    limit: int = 1000,
) -> dict:
    market_id = get_market_id()
    candles = fetch_backtest_candles(market_id, limit=limit)

    if len(candles) < long_period + 10:
        return {
            "short_period": short_period,
            "long_period": long_period,
            "candles": len(candles),
            "final_equity": INITIAL_CASH,
            "total_return": 0.0,
            "realized_pnl": 0.0,
            "buy_count": 0,
            "sell_count": 0,
            "win_rate": 0.0,
        }

    short_sma = calculate_sma(candles, short_period)
    long_sma = calculate_sma(candles, long_period)

    cash = INITIAL_CASH
    position_qty = 0.0
    entry_price: Optional[float] = None

    trades: list[dict] = []

    for i in range(1, len(candles)):
        candle = candles[i]
        prev_candle = candles[i - 1]
        price = to_float(candle["close"])

        if not is_continuous(prev_candle, candle):
            continue

        prev_short = short_sma[i - 1]
        prev_long = long_sma[i - 1]
        curr_short = short_sma[i]
        curr_long = long_sma[i]

        if None in (prev_short, prev_long, curr_short, curr_long):
            continue

        prev_diff = prev_short - prev_long
        curr_diff = curr_short - curr_long

        buy_signal = prev_diff <= 0 and curr_diff > 0
        sell_signal = prev_diff >= 0 and curr_diff < 0

        if buy_signal and position_qty == 0:
            fee = TRADE_NOTIONAL * FEE_RATE
            total_cost = TRADE_NOTIONAL + fee

            if cash >= total_cost:
                qty = TRADE_NOTIONAL / price
                cash -= total_cost
                position_qty = qty
                entry_price = price

                trades.append(
                    {
                        "side": "BUY",
                        "price": price,
                        "qty": qty,
                        "fee": fee,
                        "pnl": 0.0,
                    }
                )

        elif sell_signal and position_qty > 0 and entry_price is not None:
            gross_value = position_qty * price
            fee = gross_value * FEE_RATE
            pnl = (price - entry_price) * position_qty - fee

            cash += gross_value - fee

            trades.append(
                {
                    "side": "SELL",
                    "price": price,
                    "qty": position_qty,
                    "fee": fee,
                    "pnl": pnl,
                }
            )

            position_qty = 0.0
            entry_price = None

    final_price = to_float(candles[-1]["close"])
    final_asset_value = position_qty * final_price
    final_equity = cash + final_asset_value

    realized_pnl = sum(t["pnl"] for t in trades if t["side"] == "SELL")
    total_return = ((final_equity - INITIAL_CASH) / INITIAL_CASH) * 100

    buy_count = len([t for t in trades if t["side"] == "BUY"])
    sell_count = len([t for t in trades if t["side"] == "SELL"])

    wins = len([t for t in trades if t["side"] == "SELL" and t["pnl"] > 0])
    losses = len([t for t in trades if t["side"] == "SELL" and t["pnl"] <= 0])
    closed_trades = wins + losses
    win_rate = (wins / closed_trades * 100) if closed_trades > 0 else 0.0

    return {
        "short_period": short_period,
        "long_period": long_period,
        "candles": len(candles),
        "final_equity": final_equity,
        "total_return": total_return,
        "realized_pnl": realized_pnl,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "win_rate": win_rate,
        "open_position_value": final_asset_value,
    }


def optimize_sma_parameters(
    short_periods: list[int],
    long_periods: list[int],
    limit: int = 1000,
) -> list[dict]:
    results: list[dict] = []

    for short_period in short_periods:
        for long_period in long_periods:
            if short_period >= long_period:
                continue

            result = evaluate_sma_cross_backtest(
                short_period=short_period,
                long_period=long_period,
                limit=limit,
            )
            results.append(result)

    results.sort(key=lambda x: x["final_equity"], reverse=True)

    return results


def print_optimization_results(results: list[dict], top_n: int = 10) -> None:
    print("===================================")
    print("SMA Parameter Optimization Result")
    print("===================================")

    for i, result in enumerate(results[:top_n], start=1):
        print(
            f"{i:02d}. "
            f"SMA{result['short_period']}/SMA{result['long_period']} | "
            f"equity={result['final_equity']:.2f} | "
            f"return={result['total_return']:.2f}% | "
            f"pnl={result['realized_pnl']:.2f} | "
            f"trades={result['buy_count']}/{result['sell_count']} | "
            f"win_rate={result['win_rate']:.2f}%"
        )