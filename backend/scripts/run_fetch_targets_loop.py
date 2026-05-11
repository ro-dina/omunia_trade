from datetime import datetime, timezone

from target_loop_common import parse_targets, run_module, sleep_until_next_minute, target_is_due


def fetch_module_for_timeframe(timeframe: str) -> str:
    if timeframe == "5m":
        return "app.jobs.fetch_candles_5m"

    return "app.jobs.fetch_candles"


def main() -> None:
    targets = parse_targets("FETCH_TARGETS")
    target_text = ", ".join(f"{target.symbol}:{target.timeframe}" for target in targets)
    print(f"Fetch targets: {target_text}")

    while True:
        sleep_until_next_minute(second=5)
        now = datetime.now(timezone.utc)
        due_targets = [target for target in targets if target_is_due(target, now)]

        if not due_targets:
            print(f"No fetch targets due at {now.isoformat()}.")
            continue

        for target in due_targets:
            run_module(fetch_module_for_timeframe(target.timeframe), target)


if __name__ == "__main__":
    main()
