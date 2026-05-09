from datetime import datetime
from typing import Optional

from app.db.supabase_client import supabase
from app.services.paper_trade_service import (
    EXCHANGE,
    SYMBOL,
    MARKET_TYPE,
    FEE_RATE,
    INITIAL_CASH,
    to_float,
)

TAKE_PROFIT_RATE = 0.01  # +1.0%
STOP_LOSS_RATE = 0.005  # -0.5%

TRADE_NOTIONAL = 1_000.0


def source_for_timeframe(timeframe: str) -> str:
    if timeframe == "1m":
        return "bybit-mainnet-public"

    if timeframe == "5m":
        return "bybit-mainnet-public-5m"

    raise ValueError(f"Unsupported timeframe: {timeframe}")


def seconds_for_timeframe(timeframe: str) -> int:
    if timeframe == "1m":
        return 60

    if timeframe == "5m":
        return 300

    raise ValueError(f"Unsupported timeframe: {timeframe}")


def get_market_id_by_timeframe(timeframe: str) -> str:
    result = (
        supabase.table("markets")
        .select("id")
        .eq("exchange", EXCHANGE)
        .eq("symbol", SYMBOL)
        .eq("market_type", MARKET_TYPE)
        .eq("timeframe", timeframe)
        .limit(1)
        .execute()
    )

    if not result.data:
        raise RuntimeError(f"markets に {SYMBOL} {timeframe} が見つかりません。")

    return result.data[0]["id"]


def fetch_backtest_candles(
    market_id: str,
    timeframe: str,
    limit: int = 1000,
) -> list[dict]:
    all_rows: list[dict] = []
    page_size = 1000

    while len(all_rows) < limit:
        start = len(all_rows)
        end = min(start + page_size - 1, limit - 1)

        result = (
            supabase.table("candles")
            .select("*")
            .eq("market_id", market_id)
            .eq("source", source_for_timeframe(timeframe))
            .order("open_time", desc=True)
            .range(start, end)
            .execute()
        )

        rows = result.data or []

        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < page_size:
            break

    return list(reversed(all_rows))


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


def is_continuous(prev_candle: dict, curr_candle: dict, timeframe: str) -> bool:
    prev_time = datetime.fromisoformat(prev_candle["open_time"].replace("Z", "+00:00"))
    curr_time = datetime.fromisoformat(curr_candle["open_time"].replace("Z", "+00:00"))

    return (curr_time - prev_time).total_seconds() == seconds_for_timeframe(timeframe)


def check_tp_sl_exit(
    position_qty: float,
    entry_price: Optional[float],
    price: float,
    take_profit_rate: float,
    stop_loss_rate: float,
) -> Optional[str]:
    if position_qty <= 0 or entry_price is None:
        return None

    take_profit_price = entry_price * (1 + take_profit_rate)
    stop_loss_price = entry_price * (1 - stop_loss_rate)

    if price >= take_profit_price:
        return "TAKE_PROFIT"

    if price <= stop_loss_price:
        return "STOP_LOSS"

    return None


def run_sma_cross_backtest(
    short_period: int = 20,
    long_period: int = 50,
    timeframe: str = "1m",
    limit: int = 1000,
    take_profit_rate: float = TAKE_PROFIT_RATE,
    stop_loss_rate: float = STOP_LOSS_RATE,
) -> None:
    market_id = get_market_id_by_timeframe(timeframe)
    candles = fetch_backtest_candles(
        market_id=market_id,
        timeframe=timeframe,
        limit=limit,
    )

    if len(candles) < long_period + 10:
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

        exit_reason = check_tp_sl_exit(
            position_qty=position_qty,
            entry_price=entry_price,
            price=price,
            take_profit_rate=take_profit_rate,
            stop_loss_rate=stop_loss_rate,
        )

        if exit_reason is not None and entry_price is not None:
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
                    "reason": exit_reason,
                }
            )

            position_qty = 0.0
            entry_price = None
            continue

        if not is_continuous(prev_candle, candle, timeframe):
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
                        "reason": "SMA_CROSS_BUY",
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
                    "reason": "SMA_CROSS_SELL",
                }
            )

            position_qty = 0.0
            entry_price = None

    final_price = to_float(candles[-1]["close"])

    if position_qty > 0 and entry_price is not None:
        gross_value = position_qty * final_price
        fee = gross_value * FEE_RATE
        pnl = (final_price - entry_price) * position_qty - fee

        cash += gross_value - fee

        trades.append(
            {
                "time": candles[-1]["open_time"],
                "side": "SELL",
                "price": final_price,
                "qty": position_qty,
                "fee": fee,
                "pnl": pnl,
                "reason": "FORCED_CLOSE",
            }
        )

        position_qty = 0.0
        entry_price = None

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

    take_profit_count = len([t for t in trades if t.get("reason") == "TAKE_PROFIT"])
    stop_loss_count = len([t for t in trades if t.get("reason") == "STOP_LOSS"])
    sma_exit_count = len([t for t in trades if t.get("reason") == "SMA_CROSS_SELL"])
    forced_close_count = len([t for t in trades if t.get("reason") == "FORCED_CLOSE"])

    print("===================================")
    print(f"SMA{short_period} / SMA{long_period} Backtest Result")
    print("===================================")
    print(f"Symbol: {EXCHANGE} {SYMBOL} {MARKET_TYPE} {timeframe}")
    print(f"Candles: {len(candles)}")
    print(f"TP/SL: +{take_profit_rate * 100:.2f}% / -{stop_loss_rate * 100:.2f}%")
    print(f"Initial cash: {INITIAL_CASH:.2f} USDT")
    print(f"Final equity: {final_equity:.2f} USDT")
    print(f"Total return: {total_return:.2f}%")
    print(f"Realized PnL: {realized_pnl:.2f} USDT")
    print(f"Open position value: {final_asset_value:.2f} USDT")
    print(f"BUY count: {buy_count}")
    print(f"SELL count: {sell_count}")
    print(f"Take profit exits: {take_profit_count}")
    print(f"Stop loss exits: {stop_loss_count}")
    print(f"SMA exits: {sma_exit_count}")
    print(f"Forced close exits: {forced_close_count}")
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
            f"reason={trade.get('reason', '-')}",
        )


def evaluate_sma_cross_backtest(
    short_period: int,
    long_period: int,
    timeframe: str = "1m",
    limit: int = 1000,
    take_profit_rate: float = TAKE_PROFIT_RATE,
    stop_loss_rate: float = STOP_LOSS_RATE,
) -> dict:
    market_id = get_market_id_by_timeframe(timeframe)
    candles = fetch_backtest_candles(
        market_id=market_id,
        timeframe=timeframe,
        limit=limit,
    )

    if len(candles) < long_period + 10:
        return {
            "short_period": short_period,
            "long_period": long_period,
            "timeframe": timeframe,
            "candles": len(candles),
            "final_equity": INITIAL_CASH,
            "total_return": 0.0,
            "realized_pnl": 0.0,
            "buy_count": 0,
            "sell_count": 0,
            "win_rate": 0.0,
            "open_position_value": 0.0,
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

        exit_reason = check_tp_sl_exit(
            position_qty=position_qty,
            entry_price=entry_price,
            price=price,
            take_profit_rate=take_profit_rate,
            stop_loss_rate=stop_loss_rate,
        )

        if exit_reason is not None and entry_price is not None:
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
                    "reason": exit_reason,
                }
            )

            position_qty = 0.0
            entry_price = None
            continue

        if not is_continuous(prev_candle, candle, timeframe):
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
                        "reason": "SMA_CROSS_BUY",
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
                    "reason": "SMA_CROSS_SELL",
                }
            )

            position_qty = 0.0
            entry_price = None

    final_price = to_float(candles[-1]["close"])

    if position_qty > 0 and entry_price is not None:
        gross_value = position_qty * final_price
        fee = gross_value * FEE_RATE
        pnl = (final_price - entry_price) * position_qty - fee

        cash += gross_value - fee

        trades.append(
            {
                "side": "SELL",
                "price": final_price,
                "qty": position_qty,
                "fee": fee,
                "pnl": pnl,
                "reason": "FORCED_CLOSE",
            }
        )

        position_qty = 0.0
        entry_price = None

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

    take_profit_count = len([t for t in trades if t.get("reason") == "TAKE_PROFIT"])
    stop_loss_count = len([t for t in trades if t.get("reason") == "STOP_LOSS"])

    return {
        "short_period": short_period,
        "long_period": long_period,
        "timeframe": timeframe,
        "candles": len(candles),
        "final_equity": final_equity,
        "total_return": total_return,
        "realized_pnl": realized_pnl,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "win_rate": win_rate,
        "open_position_value": final_asset_value,
        "take_profit_count": take_profit_count,
        "stop_loss_count": stop_loss_count,
        "take_profit_rate": take_profit_rate,
        "stop_loss_rate": stop_loss_rate,
    }


def optimize_sma_parameters(
    short_periods: list[int],
    long_periods: list[int],
    timeframe: str = "1m",
    limit: int = 1000,
    take_profit_rates: Optional[list[float]] = None,
    stop_loss_rates: Optional[list[float]] = None,
) -> list[dict]:
    results: list[dict] = []

    if take_profit_rates is None:
        take_profit_rates = [0.005, 0.01, 0.015, 0.02]

    if stop_loss_rates is None:
        stop_loss_rates = [0.003, 0.005, 0.008, 0.01]

    for short_period in short_periods:
        for long_period in long_periods:
            if short_period >= long_period:
                continue

            for take_profit_rate in take_profit_rates:
                for stop_loss_rate in stop_loss_rates:
                    result = evaluate_sma_cross_backtest(
                        short_period=short_period,
                        long_period=long_period,
                        timeframe=timeframe,
                        limit=limit,
                        take_profit_rate=take_profit_rate,
                        stop_loss_rate=stop_loss_rate,
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
            f"{result['timeframe']} "
            f"SMA{result['short_period']}/SMA{result['long_period']} | "
            f"equity={result['final_equity']:.2f} | "
            f"return={result['total_return']:.2f}% | "
            f"pnl={result['realized_pnl']:.2f} | "
            f"trades={result['buy_count']}/{result['sell_count']} | "
            f"win_rate={result['win_rate']:.2f}% | "
            f"TP/SL exits={result.get('take_profit_count', 0)}/{result.get('stop_loss_count', 0)} | "
            f"tp={result.get('take_profit_rate', 0) * 100:.1f}% | "
            f"sl={result.get('stop_loss_rate', 0) * 100:.1f}%"
        )

def save_backtest_results(results: list[dict], top_n: int = 20) -> None:
    rows = []

    for result in results[:top_n]:
        strategy_name = (
            f"sma{result['short_period']}_"
            f"sma{result['long_period']}_"
            f"{result['timeframe']}_"
            f"tp{result['take_profit_rate']}_"
            f"sl{result['stop_loss_rate']}"
        )

        rows.append(
            {
                "strategy_name": strategy_name,
                "exchange": EXCHANGE,
                "symbol": SYMBOL,
                "market_type": MARKET_TYPE,
                "timeframe": result["timeframe"],
                "candle_count": result["candles"],
                "short_period": result["short_period"],
                "long_period": result["long_period"],
                "take_profit_rate": result["take_profit_rate"],
                "stop_loss_rate": result["stop_loss_rate"],
                "initial_cash": INITIAL_CASH,
                "final_equity": result["final_equity"],
                "total_return": result["total_return"],
                "realized_pnl": result["realized_pnl"],
                "buy_count": result["buy_count"],
                "sell_count": result["sell_count"],
                "win_rate": result["win_rate"],
                "take_profit_count": result.get("take_profit_count", 0),
                "stop_loss_count": result.get("stop_loss_count", 0),
                "open_position_value": result["open_position_value"],
                "meta": {
                    "rank_source": "optimization",
                    "fee_rate": FEE_RATE,
                },
            }
        )

    if not rows:
        return

    supabase.table("backtest_results").insert(rows).execute()

    print(f"saved backtest_results: {len(rows)}")