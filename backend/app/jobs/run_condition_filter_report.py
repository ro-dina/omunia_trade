import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class ReportTarget:
    symbol: str
    timeframe: str


TP_RATE = float(os.getenv("CONDITION_TP_RATE", "0.010"))
SL_RATE = float(os.getenv("CONDITION_SL_RATE", "0.008"))
LOOKAHEAD_STEPS = int(os.getenv("CONDITION_LOOKAHEAD_STEPS", "24"))
TIE_BREAKER = os.getenv("CONDITION_TIE_BREAKER", "stop_loss")
INITIAL_CASH = float(os.getenv("CONDITION_INITIAL_CASH", "10000"))
TRADE_NOTIONAL = float(os.getenv("CONDITION_TRADE_NOTIONAL", "1000"))
FEE_RATE = float(os.getenv("CONDITION_FEE_RATE", "0.0006"))
MIN_ENTRY_COUNT = int(os.getenv("CONDITION_MIN_ENTRY_COUNT", "100"))
FEATURES = [
    value.strip()
    for value in os.getenv(
        "CONDITION_FEATURES",
        "sma_30_slope,rolling_std_24,volume_zscore_24,rsi_14,macd_hist,atr_14",
    ).split(",")
    if value.strip()
]

DATA_DIR = Path(os.getenv("ML_DATA_DIR", "data/ml"))
REPORT_DIR = Path(os.getenv("CONDITION_REPORT_DIR", "data/reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if TIE_BREAKER not in {"stop_loss", "take_profit"}:
    raise ValueError("CONDITION_TIE_BREAKER must be stop_loss or take_profit.")


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
    required_columns = ["open_time", "high", "low", "close"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")

    available_features = [feature for feature in FEATURES if feature in df.columns]

    if not available_features:
        raise ValueError(f"No requested condition features exist in {path}: {FEATURES}")

    df = df.dropna(subset=required_columns + available_features).reset_index(drop=True)

    for column in ["high", "low", "close", *available_features]:
        df[column] = df[column].astype(float)

    return df


def resolve_exit(df: pd.DataFrame, entry_index: int) -> tuple[int, float, str, float]:
    entry_price = float(df.iloc[entry_index]["close"])
    tp_price = entry_price * (1 + TP_RATE)
    sl_price = entry_price * (1 - SL_RATE)
    last_index = min(entry_index + LOOKAHEAD_STEPS, len(df) - 1)

    for i in range(entry_index + 1, last_index + 1):
        row = df.iloc[i]
        hit_tp = float(row["high"]) >= tp_price
        hit_sl = float(row["low"]) <= sl_price

        if hit_tp and hit_sl:
            if TIE_BREAKER == "take_profit":
                return i, tp_price, "take_profit", TP_RATE

            return i, sl_price, "stop_loss", -SL_RATE

        if hit_tp:
            return i, tp_price, "take_profit", TP_RATE

        if hit_sl:
            return i, sl_price, "stop_loss", -SL_RATE

    exit_price = float(df.iloc[last_index]["close"])
    return last_index, exit_price, "time_exit", (exit_price - entry_price) / entry_price


def build_conditions(df: pd.DataFrame) -> list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]]:
    conditions: list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]] = []

    for feature in FEATURES:
        if feature not in df.columns:
            continue

        quantiles = df[feature].quantile([0.25, 0.5, 0.75])
        q25 = float(quantiles.loc[0.25])
        q50 = float(quantiles.loc[0.5])
        q75 = float(quantiles.loc[0.75])

        conditions.extend(
            [
                (feature, f"{feature} <= q25 ({q25:.6g})", lambda frame, f=feature, v=q25: frame[f] <= v),
                (
                    feature,
                    f"q25 < {feature} <= q50 ({q25:.6g},{q50:.6g})",
                    lambda frame, f=feature, lo=q25, hi=q50: (frame[f] > lo) & (frame[f] <= hi),
                ),
                (
                    feature,
                    f"q50 < {feature} <= q75 ({q50:.6g},{q75:.6g})",
                    lambda frame, f=feature, lo=q50, hi=q75: (frame[f] > lo) & (frame[f] <= hi),
                ),
                (feature, f"{feature} > q75 ({q75:.6g})", lambda frame, f=feature, v=q75: frame[f] > v),
            ]
        )

        if df[feature].min() < 0 < df[feature].max():
            conditions.extend(
                [
                    (feature, f"{feature} > 0", lambda frame, f=feature: frame[f] > 0),
                    (feature, f"{feature} <= 0", lambda frame, f=feature: frame[f] <= 0),
                ]
            )

    if "rsi_14" in df.columns:
        rsi_ranges = [(0, 35), (35, 45), (45, 55), (55, 65), (65, 100)]
        for low, high in rsi_ranges:
            conditions.append(
                (
                    "rsi_14",
                    f"{low} <= rsi_14 < {high}",
                    lambda frame, lo=low, hi=high: (frame["rsi_14"] >= lo) & (frame["rsi_14"] < hi),
                )
            )

    if "volume_zscore_24" in df.columns:
        conditions.extend(
            [
                ("volume_zscore_24", "volume_zscore_24 > 1", lambda frame: frame["volume_zscore_24"] > 1),
                ("volume_zscore_24", "volume_zscore_24 > 2", lambda frame: frame["volume_zscore_24"] > 2),
                ("volume_zscore_24", "volume_zscore_24 < -1", lambda frame: frame["volume_zscore_24"] < -1),
            ]
        )

    return conditions


def scan_condition_entries(df: pd.DataFrame, entry_indices: list[int]) -> dict:
    outcomes = []
    returns = []

    for entry_index in entry_indices:
        if entry_index >= len(df) - 1:
            continue

        _, _, outcome, gross_return = resolve_exit(df, entry_index)
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


def simulate_condition_entries(df: pd.DataFrame, condition_mask: pd.Series) -> dict:
    cash = INITIAL_CASH
    trades = []
    entry_index = 0

    while entry_index < len(df) - 1:
        if not bool(condition_mask.iloc[entry_index]):
            entry_index += 1
            continue

        entry_price = float(df.iloc[entry_index]["close"])
        notional = min(TRADE_NOTIONAL, cash)

        if notional <= 0:
            break

        exit_index, exit_price, outcome, gross_return = resolve_exit(df, entry_index)
        entry_fee = notional * FEE_RATE
        qty = (notional - entry_fee) / entry_price
        gross = qty * exit_price
        exit_fee = gross * FEE_RATE
        pnl = gross - exit_fee - notional
        cash += pnl

        trades.append({"outcome": outcome, "gross_return": gross_return, "pnl": pnl})
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

    for feature, condition_name, condition_fn in build_conditions(df):
        mask = condition_fn(df).fillna(False)
        entry_indices = [int(index) for index in df.index[mask]]

        if len(entry_indices) < MIN_ENTRY_COUNT:
            continue

        distribution = scan_condition_entries(df, entry_indices)
        simulation = simulate_condition_entries(df, mask)
        row = {
            "symbol": target.symbol,
            "timeframe": target.timeframe,
            "feature": feature,
            "condition": condition_name,
            "tp_rate": TP_RATE,
            "sl_rate": SL_RATE,
            "lookahead_steps": LOOKAHEAD_STEPS,
            "condition_rate_pct": len(entry_indices) / len(df) * 100 if len(df) else 0.0,
            **distribution,
            **simulation,
        }
        rows.append(row)
        print(
            f"{target.symbol} {target.timeframe} {condition_name} "
            f"entries={row['entry_count']} avg_net={row['avg_net_return']:.5f} "
            f"pf={row['profit_factor']:.3f} pnl={row['total_pnl']:.2f}"
        )

    return rows


def main() -> None:
    targets = parse_targets()
    all_rows = []
    skipped_targets = []

    print("===================================")
    print("Condition Filter Report")
    print("===================================")
    print(f"targets: {', '.join(f'{target.symbol}:{target.timeframe}' for target in targets)}")
    print(f"tp/sl/lookahead: {TP_RATE}/{SL_RATE}/{LOOKAHEAD_STEPS}")
    print(f"features: {', '.join(FEATURES)}")
    print(f"min_entry_count: {MIN_ENTRY_COUNT}")
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
        raise RuntimeError("No condition filter rows were produced.")

    report_df = pd.DataFrame(all_rows)
    report_df = report_df.sort_values(
        ["avg_net_return", "profit_factor", "total_pnl"],
        ascending=[False, False, False],
    )

    report_path = REPORT_DIR / "condition_filter_report.csv"
    skipped_path = REPORT_DIR / "condition_filter_skipped.csv"
    report_df.to_csv(report_path, index=False)

    if skipped_targets:
        pd.DataFrame(skipped_targets).to_csv(skipped_path, index=False)

    print("\n===================================")
    print("Top Results")
    print("===================================")
    print(report_df.head(30).to_string(index=False))
    print("===================================")
    print(f"condition report: {report_path}")
    if skipped_targets:
        print(f"skipped targets: {skipped_path}")
    print("===================================")


if __name__ == "__main__":
    main()
