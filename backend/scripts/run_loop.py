import time
from datetime import datetime
from app.jobs.fetch_candles import main

while True:
    now = datetime.utcnow()
    sleep_seconds = 60 - now.second

    print(f"Sleeping {sleep_seconds}s to align...")
    time.sleep(sleep_seconds)

    try:
        print(f"[{datetime.utcnow()}] Fetching candles...")
        main()
    except Exception as e:
        print("ERROR:", e)