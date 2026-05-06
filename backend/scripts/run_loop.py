import time
from datetime import datetime, timezone

from app.jobs.fetch_candles import main as fetch_candles
from app.jobs.run_paper_strategy import main as run_paper_strategy


while True:
    now = datetime.now(timezone.utc)

    # 毎分05秒に実行
    sleep_seconds = 65 - now.second - (now.microsecond / 1_000_000)

    if sleep_seconds <= 0:
        sleep_seconds += 60

    print(f"Sleeping {sleep_seconds:.1f}s to align...")
    time.sleep(sleep_seconds)

    try:
        print(f"[{datetime.now(timezone.utc)}] Fetching candles...")
        fetch_candles()

        print("Running paper strategy...")
        run_paper_strategy()

        print("Cycle done.\n")

    except Exception as e:
        print("ERROR:", e)