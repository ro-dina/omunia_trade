import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class TradeTarget:
    symbol: str
    timeframe: str


def parse_targets(env_name: str) -> list[TradeTarget]:
    raw_targets = os.getenv(env_name) or os.getenv("TRADE_TARGETS")

    if not raw_targets:
        symbol = os.getenv("TRADE_SYMBOL", "BTCUSDT")
        timeframe = os.getenv("TRADE_TIMEFRAME", "5m")
        raw_targets = f"{symbol}:{timeframe}"

    targets: list[TradeTarget] = []

    for raw_item in raw_targets.replace(";", ",").split(","):
        item = raw_item.strip()

        if not item:
            continue

        if ":" in item:
            symbol, timeframe = item.split(":", 1)
        elif "/" in item:
            symbol, timeframe = item.split("/", 1)
        else:
            raise ValueError(
                f"Invalid target '{item}'. Use SYMBOL:TIMEFRAME, e.g. BTCUSDT:5m."
            )

        timeframe = timeframe.strip().lower()

        if timeframe not in {"1m", "5m"}:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: 1m, 5m.")

        targets.append(TradeTarget(symbol=symbol.strip().upper(), timeframe=timeframe))

    if not targets:
        raise ValueError(f"{env_name} is empty.")

    return targets


def target_is_due(target: TradeTarget, now: datetime) -> bool:
    if target.timeframe == "1m":
        return True

    if target.timeframe == "5m":
        return now.minute % 5 == 0

    return False


def sleep_until_next_minute(second: int) -> None:
    now = datetime.now(timezone.utc)
    target = (now + timedelta(minutes=1)).replace(second=second, microsecond=0)
    sleep_seconds = (target - now).total_seconds()

    if sleep_seconds <= 0:
        sleep_seconds += 60

    print(f"Sleeping {sleep_seconds:.1f}s until {target.isoformat()}...")
    time.sleep(sleep_seconds)


def run_module(module_name: str, target: TradeTarget) -> int:
    env = os.environ.copy()
    env["TRADE_SYMBOL"] = target.symbol
    env["TRADE_TIMEFRAME"] = target.timeframe

    if "TRADE_SOURCE" in env:
        del env["TRADE_SOURCE"]

    command = [sys.executable, "-m", module_name]
    print(f"[{datetime.now(timezone.utc).isoformat()}] {target.symbol} {target.timeframe}: {module_name}")

    result = subprocess.run(command, env=env, check=False)

    if result.returncode != 0:
        print(
            f"ERROR: {module_name} failed for {target.symbol} {target.timeframe} "
            f"with exit code {result.returncode}"
        )

    return result.returncode
