import os
import time
from datetime import datetime, timezone

import requests

from app.db.supabase_client import supabase

BYBIT_URL = "https://api.bybit.com/v5/market/kline"

EXCHANGE = os.getenv("TRADE_EXCHANGE", "bybit")
SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
MARKET_TYPE = os.getenv("TRADE_MARKET_TYPE", "linear")
TIMEFRAME = os.getenv("TRADE_TIMEFRAME", "5m")
SOURCE = os.getenv("TRADE_SOURCE", f"bybit-mainnet-public-{TIMEFRAME}")

INTERVAL = "1" if TIMEFRAME == "1m" else "5"
LIMIT_PER_REQUEST = int(os.getenv("HISTORICAL_LIMIT_PER_REQUEST", "1000"))
TOTAL_CANDLES_TARGET = int(os.getenv("HISTORICAL_TOTAL_CANDLES", "50000"))
SLEEP_SECONDS = float(os.getenv("HISTORICAL_SLEEP_SECONDS", "0.25"))
MAX_RETRIES = int(os.getenv("HISTORICAL_MAX_RETRIES", "5"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("HISTORICAL_REQUEST_TIMEOUT", "20"))


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


def fetch_klines(end_ms: int | None = None) -> list[list[str]]:
    params = {
        "category": MARKET_TYPE,
        "symbol": SYMBOL,
        "interval": INTERVAL,
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
            wait_seconds = min(2 ** attempt, 30)
            print(
                f"fetch failed attempt={attempt}/{MAX_RETRIES}: {error}. "
                f"retrying in {wait_seconds}s"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"fetch failed after retries: {last_error}")


def convert_to_rows(market_id: str, klines: list[list[str]]) -> list[dict]:
    rows = []

    for item in klines:
        rows.append(
            {
                "market_id": market_id,
                "open_time": datetime.fromtimestamp(
                    int(item[0]) / 1000,
                    tz=timezone.utc,
                ).isoformat(),
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
                "source": SOURCE,
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


def main() -> None:
    market_id = get_market_id()

    total_saved = 0
    total_fetched = 0
    end_ms: int | None = None
    previous_oldest_ms: int | None = None

    print("===================================")
    print("Historical candle fetch started")
    print(f"symbol: {SYMBOL}")
    print(f"timeframe: {TIMEFRAME}")
    print(f"source: {SOURCE}")
    print(f"target: {TOTAL_CANDLES_TARGET}")
    print(f"limit/request: {LIMIT_PER_REQUEST}")
    print("===================================")

    while total_fetched < TOTAL_CANDLES_TARGET:
        klines = fetch_klines(end_ms=end_ms)

        if not klines:
            print("No more klines.")
            break

        rows = convert_to_rows(market_id, klines)
        saved_count = save_rows(rows)

        total_fetched += len(klines)
        total_saved += saved_count

        timestamps = [int(item[0]) for item in klines]
        oldest_ms = min(timestamps)
        newest_ms = max(timestamps)

        if previous_oldest_ms is not None and oldest_ms >= previous_oldest_ms:
            print(
                "pagination stopped because oldest timestamp did not move backward. "
                f"oldest_ms={oldest_ms}, previous_oldest_ms={previous_oldest_ms}"
            )
            break

        previous_oldest_ms = oldest_ms

        oldest = datetime.fromtimestamp(oldest_ms / 1000, tz=timezone.utc)
        newest = datetime.fromtimestamp(newest_ms / 1000, tz=timezone.utc)

        print(
            f"fetched={len(klines)} saved={saved_count} "
            f"range={oldest.isoformat()} -> {newest.isoformat()} "
            f"total_fetched={total_fetched} total_saved={total_saved}"
        )

        # 次は今回取得した最古足よりさらに前を取る
        end_ms = oldest_ms - 1

        time.sleep(SLEEP_SECONDS)

    print("===================================")
    print("Historical fetch completed")
    print(f"symbol: {SYMBOL}")
    print(f"timeframe: {TIMEFRAME}")
    print(f"total_fetched: {total_fetched}")
    print(f"total_saved: {total_saved}")
    print("===================================")


if __name__ == "__main__":
    main()