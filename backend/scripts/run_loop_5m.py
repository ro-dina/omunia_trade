import time
from datetime import datetime, timezone

from app.jobs.fetch_candles_5m import main as fetch_candles_5m


while True:
    now = datetime.now(timezone.utc)

    # 次の5分区切り + 5秒に実行
    next_minute = ((now.minute // 5) + 1) * 5

    if next_minute >= 60:
        target_minute = 0
        extra_hour = 1
    else:
        target_minute = next_minute
        extra_hour = 0

    target = now.replace(
        minute=target_minute,
        second=5,
        microsecond=0,
    )

    if extra_hour:
        target = target.replace(hour=(target.hour + 1) % 24)

    sleep_seconds = (target - now).total_seconds()

    if sleep_seconds <= 0:
        sleep_seconds += 300

    print(f"Sleeping {sleep_seconds:.1f}s to align 5m...")
    time.sleep(sleep_seconds)

    try:
        print(f"[{datetime.now(timezone.utc)}] Fetching 5m candles...")
        fetch_candles_5m()
        print("5m cycle done.\n")

    except Exception as e:
        print("ERROR:", e)