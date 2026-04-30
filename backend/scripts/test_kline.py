# backend/test_kline.py

import requests

url = "https://api-testnet.bybit.com/v5/market/kline"

params = {
    "category": "linear",
    "symbol": "BTCUSDT",
    "interval": "1",
    "limit": 5
}

res = requests.get(url, params=params)
data = res.json()

print(data)