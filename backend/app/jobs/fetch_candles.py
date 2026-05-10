# backend/app/jobs/fetch_candles.py

import os
from datetime import datetime, timezone

import requests

from app.db.supabase_client import supabase

#BYBIT_URL = "https://api-testnet.bybit.com/v5/market/kline"
BYBIT_URL = "https://api.bybit.com/v5/market/kline"

EXCHANGE = os.getenv("TRADE_EXCHANGE", "bybit")
SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
MARKET_TYPE = os.getenv("TRADE_MARKET_TYPE", "linear")
TIMEFRAME = os.getenv("TRADE_TIMEFRAME", "1m")
DEFAULT_SOURCE = "bybit-mainnet-public" if TIMEFRAME == "1m" else f"bybit-mainnet-public-{TIMEFRAME}"
SOURCE = os.getenv("TRADE_SOURCE", DEFAULT_SOURCE)


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
            f"marketsテーブルに {SYMBOL} {TIMEFRAME} の設定が見つかりません。"
        )

    return result.data[0]["id"]


def fetch_bybit_klines(limit: int = 5) -> list[list[str]]:
    interval = "1" if TIMEFRAME == "1m" else "5"

    params = {
        "category": MARKET_TYPE,
        "symbol": SYMBOL,
        "interval": interval,
        "limit": limit,
    }

    response = requests.get(BYBIT_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {data}")

    return data["result"]["list"]


def convert_to_candle_rows(market_id: str, klines: list[list[str]]) -> list[dict]:
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
                # "source": "bybit-testnet",
                "source": SOURCE,
            }
        )

    return rows


def save_candles(rows: list[dict]) -> list[dict]:
    result = (
        supabase.table("candles")
        .upsert(rows, on_conflict="market_id,open_time")
        .execute()
    )

    return result.data


def main() -> None:
    market_id = get_market_id()

    # 最新足は形成中の可能性があるので2本取得する
    klines = fetch_bybit_klines(limit=2)

    # 1つ前の確定済み足だけ保存する
    confirmed_klines = [klines[1]]

    rows = convert_to_candle_rows(market_id, confirmed_klines)
    saved = save_candles(rows)

    print(f"saved {TIMEFRAME} candles ({SYMBOL}): {len(saved)}")

    for row in saved:
        print(row["open_time"], row["close"], "volume:", row["volume"])


if __name__ == "__main__":
    main()