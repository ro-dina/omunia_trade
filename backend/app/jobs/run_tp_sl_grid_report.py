import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReportTarget:
    symbol: str
    timeframe: str


TP_RATES = [
    float(value.strip())
    for value in os.getenv("TP_SL_GRID_TP_RATES", "0.004,0.006,0.008,0.010").split(",")
    if value.strip()
]
SL_RATES = [
    float(value.strip())
    for value in os.getenv("TP_SL_GRID_SL_RATES", "0.003,0.004,0.006,0.008").split(",")
    if value.strip()
]
LOOKAHEAD_STEPS_LIST = [
    int(value.strip())
    for value in os.getenv("TP_SL_GRID_LOOKAHEAD_STEPS", "6,12,24").split(",")
    if value.strip()
]
TIE_BREAKER = os.getenv("TP_SL_GRID_TIE_BREAKER", "stop_loss")
INITIAL_CASH = float(os.getenv("TP_SL_GRID_INITIAL_CASH", "10000"))
TRADE_NOTIONAL = float(os.getenv("TP_SL_GRID_TRADE_NOTIONAL", "1000"))
FEE_RATE = float(os.getenv("TP_SL_GRID_FEE_RATE", "0.0006"))

DATA_DIR = Path(os.getenv("ML_DATA_DIR", "data/ml"))
REPORT_DIR = Path(os.getenv("TP_SL_GRID_REPORT_DIR", "data/reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if TIE_BREAKER not in {"stop_loss", "take_profit"}:
    raise ValueError("TP_SL_GRID_TIE_BREAKER must be stop_loss or take_profit.")


def parse_targets() -> list[ReportTarget]:
    raw_targets = os.getenv("REPORT_TARGETS") or os.getenv("TRADE_TARGETS")

    if not raw_targets:
        symbol = os.getenv("TRADE_SYMBOL", "BTCUSDT")
        timeframe = os.getenv("TRADE_TIMEFRAME", "5m")
        raw_targets = f"{symbol}:{timeframe}"

    targets: list[ReportTarget] = []

    for raw_item in raw_targets.replace(";", ",").split(","):
        item = raw_item.strip()

        if not item:
            continue

        if ":" in item:
            symbol, timeframe = item.split(":", 1)
        elif "/" in item:
            symbol, timeframe = item.split("/", 1)
        else:
            raise ValueError(f"Invalid target '{item}'. Use SYMBOL:TIMEFRAME.")

        targets.append(ReportTarget(symbol=symbol.strip().upper(), timeframe=timeframe.strip().lower()))

    if not targets:
        raise ValueError("No report targets were provided.")

    return targets


def load_dataset(target: ReportTarget) -> pd.DataFrame:
    path = DATA_DIR / f"{target.symbol}_{target.timeframe}_dataset.csv"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing_columns = [
        column
        for column in ["open_time", "high", "low", "close"]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")

    df = df.dropna(subset=["open_time", "high", "low", "close"]).reset_index(drop=True)

    for column in ["high", "low", "close"]:
        df[column] = df[column].astype(float)

    return df


def resolve_exit(
    df: pd.DataFrame,
    entry_index: int,
    tp_rate: float,
    sl_rate: float,
    lookahead_steps: int,
) -> tuple[int, float, str, float]:
    entry_price = float(df.iloc[entry_index]["close"])
    tp_price = entry_price * (1 + tp_rate)
    sl_price = entry_price * (1 - sl_rate)
    last_index = min(entry_index + lookahead_steps, len(df) - 1)

    for i in range(entry_index + 1, last_index + 1):
        row = df.iloc[i]
        hit_tp = float(row["high"]) >= tp_price
        hit_sl = float(row["low"]) <= sl_price

        if hit_tp and hit_sl:
            if TIE_BREAKER == "take_profit":
                return i, tp_price, "take_profit", tp_rate

            return i, sl_price, "stop_loss", -sl_rate

        if hit_tp:
            return i, tp_price, "take_profit", tp_rate

        if hit_sl:
            return i, sl_price, "stop_loss", -sl_rate

    exit_price = float(df.iloc[last_index]["close"])
    return last_index, exit_price, "time_exit", (exit_price - entry_price) / entry_price


def scan_all_entries(
    df: pd.DataFrame,
    tp_rate: float,
    sl_rate: float,
    lookahead_steps: int,
) -> dict:
    outcomes = []
    returns = []
    max_entry_index = len(df) - 1

    for entry_index in range(max_entry_index):
        _, _, outcome, gross_return = resolve_exit(
            df=df,
            entry_index=entry_index,
            tp_rate=tp_rate,
            sl_rate=sl_rate,
            lookahead_steps=lookahead_steps,
        )
        outcomes.append(outcome)
        returns.append(gross_return)

    entry_count = len(outcomes)
    tp_count = sum(1 for outcome in outcomes if outcome == "take_profit")
    sl_count = sum(1 for outcome in outcomes if outcome == "stop_loss")
    time_exit_count = sum(1 for outcome in outcomes if outcome == "time_exit")
    avg_gross_return = sum(returns) / entry_count if entry_count else 0.0
    avg_net_return = avg_gross_return - (FEE_RATE * 2)

    return {
        "entry_count": entry_count,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "time_exit_count": time_exit_count,
        "tp_rate_pct": tp_count / entry_count * 100 if entry_count else 0.0,
        "sl_rate_pct": sl_count / entry_count * 100 if entry_count else 0.0,
        "time_exit_rate_pct": time_exit_count / entry_count * 100 if entry_count else 0.0,
        "avg_gross_return": avg_gross_return,
        "avg_net_return": avg_net_return,
    }


def simulate_non_overlapping_entries(
    df: pd.DataFrame,
    tp_rate: float,
    sl_rate: float,
    lookahead_steps: int,
) -> dict:
    cash = INITIAL_CASH
    trades = []
    entry_index = 0

    while entry_index < len(df) - 1:
        entry_price = float(df.iloc[entry_index]["close"])
        notional = min(TRADE_NOTIONAL, cash)

        if notional <= 0:
            break

        exit_index, exit_price, outcome, gross_return = resolve_exit(
            df=df,
            entry_index=entry_index,
            tp_rate=tp_rate,
            sl_rate=sl_rate,
            lookahead_steps=lookahead_steps,
        )
        entry_fee = notional * FEE_RATE
        qty = (notional - entry_fee) / entry_price
        gross = qty * exit_price
        exit_fee = gross * FEE_RATE
        pnl = gross - exit_fee - notional
        cash += pnl

        trades.append(
            {
                "entry_time": df.iloc[entry_index]["open_time"],
                "exit_time": df.iloc[exit_index]["open_time"],
                "outcome": outcome,
                "gross_return": gross_return,
                "pnl": pnl,
            }
        )
        entry_index = exit_index + 1

    trade_count = len(trades)
    wins = sum(1 for trade in trades if trade["pnl"] > 0)
    gross_profit = sum(trade["pnl"] for trade in trades if trade["pnl"] > 0)
    gross_loss = -sum(trade["pnl"] for trade in trades if trade["pnl"] < 0)
    total_pnl = sum(trade["pnl"] for trade in trades)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    return {
        "trade_count": trade_count,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / trade_count if trade_count else 0.0,
        "win_rate": wins / trade_count * 100 if trade_count else 0.0,
        "profit_factor": profit_factor,
        "final_equity": cash,
        "total_return": (cash - INITIAL_CASH) / INITIAL_CASH * 100,
        "sim_tp_count": sum(1 for trade in trades if trade["outcome"] == "take_profit"),
        "sim_sl_count": sum(1 for trade in trades if trade["outcome"] == "stop_loss"),
        "sim_time_exit_count": sum(1 for trade in trades if trade["outcome"] == "time_exit"),
    }


def run_target(target: ReportTarget) -> list[dict]:
    df = load_dataset(target)
    rows = []

    for lookahead_steps in LOOKAHEAD_STEPS_LIST:
        for tp_rate in TP_RATES:
            for sl_rate in SL_RATES:
                distribution = scan_all_entries(df, tp_rate, sl_rate, lookahead_steps)
                simulation = simulate_non_overlapping_entries(df, tp_rate, sl_rate, lookahead_steps)
                row = {
                    "symbol": target.symbol,
                    "timeframe": target.timeframe,
                    "tp_rate": tp_rate,
                    "sl_rate": sl_rate,
                    "lookahead_steps": lookahead_steps,
                    "reward_risk": tp_rate / sl_rate if sl_rate > 0 else 0.0,
                    **distribution,
                    **simulation,
                }
                rows.append(row)
                print(
                    f"{target.symbol} {target.timeframe} "
                    f"tp={tp_rate:.3f} sl={sl_rate:.3f} lookahead={lookahead_steps} "
                    f"tp%={row['tp_rate_pct']:.1f} sl%={row['sl_rate_pct']:.1f} "
                    f"time%={row['time_exit_rate_pct']:.1f} pnl={row['total_pnl']:.2f}"
                )

    return rows


def main() -> None:
    targets = parse_targets()
    all_rows = []
    skipped_targets = []

    print("===================================")
    print("TP/SL Grid Report")
    print("===================================")
    print(f"targets: {', '.join(f'{target.symbol}:{target.timeframe}' for target in targets)}")
    print(f"tp_rates: {', '.join(str(value) for value in TP_RATES)}")
    print(f"sl_rates: {', '.join(str(value) for value in SL_RATES)}")
    print(f"lookahead_steps: {', '.join(str(value) for value in LOOKAHEAD_STEPS_LIST)}")
    print(f"tie_breaker: {TIE_BREAKER}")
    print("===================================")

    for target in targets:
        try:
            all_rows.extend(run_target(target))
        except Exception as e:
            skipped_targets.append(
                {
                    "symbol": target.symbol,
                    "timeframe": target.timeframe,
                    "reason": str(e),
                }
            )
            print(f"Skipped {target.symbol} {target.timeframe}: {e}")

    if not all_rows:
        raise RuntimeError("No TP/SL grid rows were produced.")

    report_df = pd.DataFrame(all_rows)
    report_df = report_df.sort_values(
        ["total_pnl", "avg_net_return", "profit_factor"],
        ascending=[False, False, False],
    )

    report_path = REPORT_DIR / "tp_sl_grid_report.csv"
    skipped_path = REPORT_DIR / "tp_sl_grid_skipped.csv"
    report_df.to_csv(report_path, index=False)

    if skipped_targets:
        pd.DataFrame(skipped_targets).to_csv(skipped_path, index=False)

    print("\n===================================")
    print("Top Results")
    print("===================================")
    print(report_df.head(20).to_string(index=False))
    print("===================================")
    print(f"grid report: {report_path}")
    if skipped_targets:
        print(f"skipped targets: {skipped_path}")
    print("===================================")


if __name__ == "__main__":
    main()
