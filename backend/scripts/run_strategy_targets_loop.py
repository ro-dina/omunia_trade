import os
from datetime import datetime, timezone

from target_loop_common import parse_targets, run_module, sleep_until_next_minute, target_is_due


def main() -> None:
    targets = parse_targets("STRATEGY_TARGETS")
    run_second = int(os.getenv("STRATEGY_RUN_SECOND", "15"))
    target_text = ", ".join(f"{target.symbol}:{target.timeframe}" for target in targets)
    print(f"Strategy targets: {target_text}")
    print(f"Strategy run second: {run_second}")

    while True:
        sleep_until_next_minute(second=run_second)
        now = datetime.now(timezone.utc)
        due_targets = [target for target in targets if target_is_due(target, now)]

        if not due_targets:
            print(f"No strategy targets due at {now.isoformat()}.")
            continue

        for target in due_targets:
            run_module("app.jobs.run_paper_strategy", target)


if __name__ == "__main__":
    main()
