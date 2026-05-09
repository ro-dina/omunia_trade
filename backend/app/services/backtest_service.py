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

RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 55.0
RSI_SELL_THRESHOLD = 45.0

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
    rsi_buy_threshold: float = RSI_BUY_THRESHOLD,
    rsi_sell_threshold: float = RSI_SELL_THRESHOLD,
) -> None:
    market_id = get_market_id_by_timeframe(timeframe)
    candles = fetch_backtest_candles(
        market_id=market_id,
        timeframe=timeframe,
        limit=limit,
    )

    result = evaluate_sma_cross_backtest_with_candles(
        candles=candles,
        short_period=short_period,
        long_period=long_period,
        timeframe=timeframe,
        take_profit_rate=take_profit_rate,
        stop_loss_rate=stop_loss_rate,
        rsi_buy_threshold=rsi_buy_threshold,
        rsi_sell_threshold=rsi_sell_threshold,
    )

    print("===================================")
    print(f"SMA{short_period} / SMA{long_period} Backtest Result")
    print("===================================")
    print(f"Symbol: {EXCHANGE} {SYMBOL} {MARKET_TYPE} {timeframe}")
    print(f"Candles: {result['candles']}")
    print(f"TP/SL: +{take_profit_rate * 100:.2f}% / -{stop_loss_rate * 100:.2f}%")
    print(f"RSI: period={RSI_PERIOD}, buy>{rsi_buy_threshold}, sell<{rsi_sell_threshold}")
    print(f"Initial cash: {INITIAL_CASH:.2f} USDT")
    print(f"Final equity: {result['final_equity']:.2f} USDT")
    print(f"Total return: {result['total_return']:.2f}%")
    print(f"Realized PnL: {result['realized_pnl']:.2f} USDT")
    print(f"Open position value: {result['open_position_value']:.2f} USDT")
    print(f"BUY count: {result['buy_count']}")
    print(f"SELL count: {result['sell_count']}")
    print(f"Take profit exits: {result['take_profit_count']}")
    print(f"Stop loss exits: {result['stop_loss_count']}")
    print(f"Win rate: {result['win_rate']:.2f}%")
    print("===================================")


def evaluate_sma_cross_backtest_with_candles(
    candles: list[dict],
    short_period: int,
    long_period: int,
    timeframe: str = "1m",
    take_profit_rate: float = TAKE_PROFIT_RATE,
    stop_loss_rate: float = STOP_LOSS_RATE,
    rsi_buy_threshold: float = RSI_BUY_THRESHOLD,
    rsi_sell_threshold: float = RSI_SELL_THRESHOLD,
) -> dict:
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
            "take_profit_count": 0,
            "stop_loss_count": 0,
            "take_profit_rate": take_profit_rate,
            "stop_loss_rate": stop_loss_rate,
            "rsi_period": RSI_PERIOD,
            "rsi_buy_threshold": rsi_buy_threshold,
            "rsi_sell_threshold": rsi_sell_threshold,
        }

    short_sma = calculate_sma(candles, short_period)
    long_sma = calculate_sma(candles, long_period)
    rsi_values = calculate_rsi(candles)

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

        curr_rsi = rsi_values[i]

        if curr_rsi is None:
            continue

        prev_diff = prev_short - prev_long
        curr_diff = curr_short - curr_long

        buy_signal = (
            prev_diff <= 0
            and curr_diff > 0
            and curr_rsi > rsi_buy_threshold
        )
        sell_signal = (
            (prev_diff >= 0 and curr_diff < 0)
            or curr_rsi < rsi_sell_threshold
        )

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
        "rsi_period": RSI_PERIOD,
        "rsi_buy_threshold": rsi_buy_threshold,
        "rsi_sell_threshold": rsi_sell_threshold,
    }

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
    rsi_values = calculate_rsi(candles)

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

        curr_rsi = rsi_values[i]

        if curr_rsi is None:
            continue

        prev_diff = prev_short - prev_long
        curr_diff = curr_short - curr_long

        buy_signal = (
            prev_diff <= 0
            and curr_diff > 0
            and curr_rsi > RSI_BUY_THRESHOLD
        )
        sell_signal = (
            (prev_diff >= 0 and curr_diff < 0)
            or curr_rsi < RSI_SELL_THRESHOLD
        )

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
        "rsi_period": RSI_PERIOD,
        "rsi_buy_threshold": RSI_BUY_THRESHOLD,
        "rsi_sell_threshold": RSI_SELL_THRESHOLD,
    }


def optimize_sma_parameters(
    short_periods: list[int],
    long_periods: list[int],
    timeframe: str = "1m",
    limit: int = 1000,
    take_profit_rates: Optional[list[float]] = None,
    stop_loss_rates: Optional[list[float]] = None,
    rsi_buy_thresholds: Optional[list[float]] = None,
    rsi_sell_thresholds: Optional[list[float]] = None,
) -> list[dict]:
    results: list[dict] = []

    if take_profit_rates is None:
        take_profit_rates = [0.005, 0.01, 0.015, 0.02]

    if stop_loss_rates is None:
        stop_loss_rates = [0.003, 0.005, 0.008, 0.01]

    if rsi_buy_thresholds is None:
        rsi_buy_thresholds = [45.0, 50.0, 55.0, 60.0]

    if rsi_sell_thresholds is None:
        rsi_sell_thresholds = [35.0, 40.0, 45.0, 50.0]

    market_id = get_market_id_by_timeframe(timeframe)
    candles = fetch_backtest_candles(
        market_id=market_id,
        timeframe=timeframe,
        limit=limit,
    )
    print(f"Loaded candles once: {len(candles)}")

    for short_period in short_periods:
        for long_period in long_periods:
            if short_period >= long_period:
                continue

            for take_profit_rate in take_profit_rates:
                for stop_loss_rate in stop_loss_rates:
                    for rsi_buy_threshold in rsi_buy_thresholds:
                        for rsi_sell_threshold in rsi_sell_thresholds:
                            if rsi_sell_threshold >= rsi_buy_threshold:
                                continue

                            result = evaluate_sma_cross_backtest_with_candles(
                                candles=candles,
                                short_period=short_period,
                                long_period=long_period,
                                timeframe=timeframe,
                                take_profit_rate=take_profit_rate,
                                stop_loss_rate=stop_loss_rate,
                                rsi_buy_threshold=rsi_buy_threshold,
                                rsi_sell_threshold=rsi_sell_threshold,
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
            f"sl={result.get('stop_loss_rate', 0) * 100:.1f}% | "
            f"RSI>{result.get('rsi_buy_threshold', RSI_BUY_THRESHOLD):.0f}/<{result.get('rsi_sell_threshold', RSI_SELL_THRESHOLD):.0f}"
        )

def save_backtest_results(results: list[dict], top_n: int = 20) -> None:
    rows = []

    for result in results[:top_n]:
        strategy_name = (
            f"sma{result['short_period']}_"
            f"sma{result['long_period']}_"
            f"{result['timeframe']}_"
            f"tp{result['take_profit_rate']}_"
            f"sl{result['stop_loss_rate']}_"
            f"rsi{result.get('rsi_buy_threshold')}_{result.get('rsi_sell_threshold')}"
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
                    "rsi_period": RSI_PERIOD,
                    "rsi_buy_threshold": result.get("rsi_buy_threshold"),
                    "rsi_sell_threshold": result.get("rsi_sell_threshold"),
                },
            }
        )

    if not rows:
        return

    supabase.table("backtest_results").insert(rows).execute()

    print(f"saved backtest_results: {len(rows)}")