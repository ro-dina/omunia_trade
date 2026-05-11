import time
from datetime import datetime, timedelta, timezone

from app.jobs.fetch_candles_5m import main as fetch_candles_5m
from app.jobs.run_paper_strategy import main as run_paper_strategy


while True:
    now = datetime.now(timezone.utc)

    # 次の5分区切り + 5秒に実行
    minutes_to_add = 5 - (now.minute % 5)
    target = (now + timedelta(minutes=minutes_to_add)).replace(second=5, microsecond=0)

    sleep_seconds = (target - now).total_seconds()

    if sleep_seconds <= 0:
        sleep_seconds += 300

    print(f"Sleeping {sleep_seconds:.1f}s to align 5m...")
    time.sleep(sleep_seconds)

    try:
        print(f"[{datetime.now(timezone.utc)}] Fetching 5m candles...")
        fetch_candles_5m()
        print("Running paper strategy...")
        run_paper_strategy()
        print("5m cycle done.\n")

    except Exception as e:
        print("ERROR:", e)
