import requests
from datetime import datetime, timezone

from app.db.supabase_client import supabase

BYBIT_URL = "https://api-testnet.bybit.com/v5/market/kline"

EXCHANGE = "bybit"
SYMBOL = "BTCUSDT"
MARKET_TYPE = "linear"
TIMEFRAME = "1m"


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
        raise RuntimeError("markets に BTCUSDT がありません。先にINSERTしてください。")

    return result.data[0]["id"]


def fetch_klines():
    params = {
        "category": "linear",
        "symbol": SYMBOL,
        "interval": "1",
        "limit": 5,
    }

    res = requests.get(BYBIT_URL, params=params, timeout=10)
    res.raise_for_status()

    data = res.json()

    if data.get("retCode") != 0:
        raise RuntimeError(data)

    return data["result"]["list"]


def to_candle_rows(market_id: str, klines: list[list[str]]):
    rows = []

    for item in klines:
        open_time = datetime.fromtimestamp(
            int(item[0]) / 1000,
            tz=timezone.utc,
        ).isoformat()

        rows.append(
            {
                "market_id": market_id,
                "open_time": open_time,
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
                "source": "bybit-testnet",
            }
        )

    return rows


def save_candles(rows):
    result = (
        supabase.table("candles")
        .upsert(rows, on_conflict="market_id,open_time")
        .execute()
    )
    return result.data


def main():
    market_id = get_market_id()
    klines = fetch_klines()
    rows = to_candle_rows(market_id, klines)
    saved = save_candles(rows)

    print(f"saved candles: {len(saved)}")
    for row in saved:
        print(row["open_time"], row["close"])


if __name__ == "__main__":
    main()