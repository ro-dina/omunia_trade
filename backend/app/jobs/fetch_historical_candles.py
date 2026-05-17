import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from app.db.supabase_client import supabase


BYBIT_URL = "https://api.bybit.com/v5/market/kline"

EXCHANGE = os.getenv("TRADE_EXCHANGE", "bybit")
MARKET_TYPE = os.getenv("TRADE_MARKET_TYPE", "linear")
CREATE_MARKETS = os.getenv("HISTORICAL_CREATE_MARKETS", "1") != "0"
LIMIT_PER_REQUEST = int(os.getenv("HISTORICAL_LIMIT_PER_REQUEST", "1000"))
TOTAL_CANDLES_TARGET = int(os.getenv("HISTORICAL_TOTAL_CANDLES", "50000"))
SLEEP_SECONDS = float(os.getenv("HISTORICAL_SLEEP_SECONDS", "0.25"))
MAX_RETRIES = int(os.getenv("HISTORICAL_MAX_RETRIES", "5"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("HISTORICAL_REQUEST_TIMEOUT", "20"))
START_TIME = os.getenv("HISTORICAL_START_TIME", "").strip()
END_TIME = os.getenv("HISTORICAL_END_TIME", "").strip()


@dataclass(frozen=True)
class HistoricalTarget:
    symbol: str
    timeframe: str


def parse_targets() -> list[HistoricalTarget]:
    raw_targets = (
        os.getenv("HISTORICAL_TARGETS")
        or os.getenv("REPORT_TARGETS")
        or os.getenv("TRADE_TARGETS")
    )

    if not raw_targets:
        symbol = os.getenv("TRADE_SYMBOL", "BTCUSDT")
        timeframe = os.getenv("TRADE_TIMEFRAME", "5m")
        raw_targets = f"{symbol}:{timeframe}"

    targets: list[HistoricalTarget] = []

    for raw_item in raw_targets.replace(";", ",").split(","):
        item = raw_item.strip()

        if not item:
            continue

        if ":" in item:
            symbol, timeframe = item.split(":", 1)
        elif "/" in item:
            symbol, timeframe = item.split("/", 1)
        else:
            raise ValueError(f"Invalid target '{item}'. Use SYMBOL:TIMEFRAME.")

        targets.append(
            HistoricalTarget(
                symbol=symbol.strip().upper(),
                timeframe=timeframe.strip().lower(),
            )
        )

    if not targets:
        raise ValueError("No historical targets were provided.")

    return targets


def timeframe_to_bybit_interval(timeframe: str) -> str:
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
        "1mo": "M",
    }

    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe for Bybit kline: {timeframe}")

    return mapping[timeframe]


def timeframe_to_ms(timeframe: str) -> int:
    if timeframe.endswith("m"):
        return int(timeframe[:-1]) * 60 * 1000

    if timeframe.endswith("h"):
        return int(timeframe[:-1]) * 60 * 60 * 1000

    if timeframe == "1d":
        return 24 * 60 * 60 * 1000

    if timeframe == "1w":
        return 7 * 24 * 60 * 60 * 1000

    if timeframe == "1mo":
        return 31 * 24 * 60 * 60 * 1000

    raise ValueError(f"Unsupported timeframe for milliseconds: {timeframe}")


def parse_time_to_ms(value: str) -> int | None:
    if not value:
        return None

    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return int(parsed.timestamp() * 1000)


def default_end_ms(timeframe: str) -> int:
    interval_ms = timeframe_to_ms(timeframe)
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    return now_ms - interval_ms


def source_for(timeframe: str) -> str:
    return os.getenv("TRADE_SOURCE", f"bybit-mainnet-public-{timeframe}")


def get_or_create_market_id(target: HistoricalTarget) -> str:
    result = (
        supabase.table("markets")
        .select("id")
        .eq("exchange", EXCHANGE)
        .eq("symbol", target.symbol)
        .eq("market_type", MARKET_TYPE)
        .eq("timeframe", target.timeframe)
        .limit(1)
        .execute()
    )

    if result.data:
        return result.data[0]["id"]

    if not CREATE_MARKETS:
        raise RuntimeError(f"markets に {target.symbol} {target.timeframe} が見つかりません。")

    insert_result = (
        supabase.table("markets")
        .insert(
            {
                "exchange": EXCHANGE,
                "symbol": target.symbol,
                "market_type": MARKET_TYPE,
                "timeframe": target.timeframe,
                "is_active": True,
            }
        )
        .execute()
    )

    if not insert_result.data:
        raise RuntimeError(f"Failed to create market: {target.symbol} {target.timeframe}")

    print(f"created market: {target.symbol} {target.timeframe}")
    return insert_result.data[0]["id"]


def fetch_klines(target: HistoricalTarget, end_ms: int | None = None) -> list[list[str]]:
    params = {
        "category": MARKET_TYPE,
        "symbol": target.symbol,
        "interval": timeframe_to_bybit_interval(target.timeframe),
        "limit": LIMIT_PER_REQUEST,
    }

    if end_ms is not None:
        params["end"] = end_ms

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                BYBIT_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit API error: {data}")

            return data["result"]["list"]
        except Exception as error:
            last_error = error
            wait_seconds = min(2**attempt, 30)
            print(
                f"{target.symbol} {target.timeframe}: fetch failed "
                f"attempt={attempt}/{MAX_RETRIES}: {error}. retrying in {wait_seconds}s"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"fetch failed after retries: {last_error}")


def convert_to_rows(
    target: HistoricalTarget,
    market_id: str,
    klines: list[list[str]],
    start_ms: int | None,
    end_ms: int | None,
) -> list[dict]:
    rows = []
    source = source_for(target.timeframe)

    for item in klines:
        open_ms = int(item[0])

        if start_ms is not None and open_ms < start_ms:
            continue

        if end_ms is not None and open_ms > end_ms:
            continue

        rows.append(
            {
                "market_id": market_id,
                "open_time": datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc).isoformat(),
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
                "source": source,
            }
        )

    return rows


def save_rows(rows: list[dict]) -> int:
    if not rows:
        return 0

    result = (
        supabase.table("candles")
        .upsert(rows, on_conflict="market_id,open_time")
        .execute()
    )

    return len(result.data or [])


def fetch_target(target: HistoricalTarget) -> dict:
    market_id = get_or_create_market_id(target)
    requested_start_ms = parse_time_to_ms(START_TIME)
    requested_end_ms = parse_time_to_ms(END_TIME)
    end_ms = requested_end_ms if requested_end_ms is not None else default_end_ms(target.timeframe)
    previous_oldest_ms: int | None = None
    total_fetched = 0
    total_saved = 0

    print("-----------------------------------")
    print(f"target: {target.symbol}:{target.timeframe}")
    print(f"source: {source_for(target.timeframe)}")
    print(f"target candles: {TOTAL_CANDLES_TARGET}")
    print(f"start: {START_TIME or '(auto by total candles)'}")
    print(f"end: {END_TIME or '(latest confirmed-ish)'}")
    print("-----------------------------------")

    while total_fetched < TOTAL_CANDLES_TARGET:
        klines = fetch_klines(target, end_ms=end_ms)

        if not klines:
            print(f"{target.symbol} {target.timeframe}: no more klines.")
            break

        timestamps = [int(item[0]) for item in klines]
        oldest_ms = min(timestamps)
        newest_ms = max(timestamps)
        rows = convert_to_rows(target, market_id, klines, requested_start_ms, requested_end_ms)
        saved_count = save_rows(rows)

        total_fetched += len(klines)
        total_saved += saved_count

        oldest = datetime.fromtimestamp(oldest_ms / 1000, tz=timezone.utc)
        newest = datetime.fromtimestamp(newest_ms / 1000, tz=timezone.utc)
        print(
            f"{target.symbol} {target.timeframe}: fetched={len(klines)} saved={saved_count} "
            f"range={oldest.isoformat()} -> {newest.isoformat()} "
            f"total_fetched={total_fetched} total_saved={total_saved}"
        )

        if requested_start_ms is not None and oldest_ms <= requested_start_ms:
            print(f"{target.symbol} {target.timeframe}: reached requested start time.")
            break

        if previous_oldest_ms is not None and oldest_ms >= previous_oldest_ms:
            print(
                f"{target.symbol} {target.timeframe}: pagination stopped because oldest "
                f"timestamp did not move backward. oldest_ms={oldest_ms}, "
                f"previous_oldest_ms={previous_oldest_ms}"
            )
            break

        previous_oldest_ms = oldest_ms
        end_ms = oldest_ms - 1
        time.sleep(SLEEP_SECONDS)

    return {
        "symbol": target.symbol,
        "timeframe": target.timeframe,
        "total_fetched": total_fetched,
        "total_saved": total_saved,
    }


def main() -> None:
    targets = parse_targets()
    summaries = []

    print("===================================")
    print("Historical candle fetch started")
    print("===================================")
    print(f"targets: {', '.join(f'{target.symbol}:{target.timeframe}' for target in targets)}")
    print(f"exchange: {EXCHANGE}")
    print(f"market_type: {MARKET_TYPE}")
    print(f"limit/request: {LIMIT_PER_REQUEST}")
    print(f"sleep_seconds: {SLEEP_SECONDS}")
    print(f"create_markets: {CREATE_MARKETS}")
    print("===================================")

    for target in targets:
        summaries.append(fetch_target(target))

    print("\n===================================")
    print("Historical fetch completed")
    print("===================================")

    for summary in summaries:
        print(
            f"{summary['symbol']} {summary['timeframe']}: "
            f"total_fetched={summary['total_fetched']} total_saved={summary['total_saved']}"
        )

    print("===================================")


if __name__ == "__main__":
    main()
