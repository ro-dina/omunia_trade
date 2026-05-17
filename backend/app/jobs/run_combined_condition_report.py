import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class ReportTarget:
    symbol: str
    timeframe: str


@dataclass(frozen=True)
class Condition:
    feature: str
    name: str
    mask_fn: Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class ExitParams:
    tp_rate: float
    sl_rate: float
    lookahead_steps: int


@dataclass(frozen=True)
class ExitCache:
    exit_indices: list[int]
    exit_prices: list[float]
    outcomes: list[str]
    gross_returns: list[float]


def parse_float_list(list_env_name: str, single_env_name: str, default: str) -> list[float]:
    raw = os.getenv(list_env_name) or os.getenv(single_env_name) or default
    values = [float(value.strip()) for value in raw.replace(";", ",").split(",") if value.strip()]

    if not values:
        raise ValueError(f"{list_env_name} produced no values.")

    return values


def parse_int_list(list_env_name: str, single_env_name: str, default: str) -> list[int]:
    raw = os.getenv(list_env_name) or os.getenv(single_env_name) or default
    values = [int(value.strip()) for value in raw.replace(";", ",").split(",") if value.strip()]

    if not values:
        raise ValueError(f"{list_env_name} produced no values.")

    return values


TP_RATES = parse_float_list("COMBINED_CONDITION_TP_RATES", "COMBINED_CONDITION_TP_RATE", "0.010")
SL_RATES = parse_float_list("COMBINED_CONDITION_SL_RATES", "COMBINED_CONDITION_SL_RATE", "0.008")
LOOKAHEAD_STEPS_LIST = parse_int_list(
    "COMBINED_CONDITION_LOOKAHEAD_STEPS_LIST",
    "COMBINED_CONDITION_LOOKAHEAD_STEPS",
    "24",
)
TIE_BREAKER = os.getenv("COMBINED_CONDITION_TIE_BREAKER", "stop_loss")
INITIAL_CASH = float(os.getenv("COMBINED_CONDITION_INITIAL_CASH", "10000"))
TRADE_NOTIONAL = float(os.getenv("COMBINED_CONDITION_TRADE_NOTIONAL", "1000"))
FEE_RATE = float(os.getenv("COMBINED_CONDITION_FEE_RATE", "0.0006"))
MIN_ENTRY_COUNT = int(os.getenv("COMBINED_CONDITION_MIN_ENTRY_COUNT", "100"))
MAX_COMBINATION_SIZE = int(os.getenv("COMBINED_CONDITION_MAX_SIZE", "2"))
FEATURES = [
    value.strip()
    for value in os.getenv(
        "COMBINED_CONDITION_FEATURES",
        "sma_30_slope,rolling_std_24,volume_zscore_24,rsi_14,macd_hist,atr_14",
    ).split(",")
    if value.strip()
]

DATA_DIR = Path(os.getenv("ML_DATA_DIR", "data/ml"))
REPORT_DIR = Path(os.getenv("COMBINED_CONDITION_REPORT_DIR", "data/reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if TIE_BREAKER not in {"stop_loss", "take_profit"}:
    raise ValueError("COMBINED_CONDITION_TIE_BREAKER must be stop_loss or take_profit.")


def build_exit_param_grid() -> list[ExitParams]:
    return [
        ExitParams(tp_rate=tp_rate, sl_rate=sl_rate, lookahead_steps=lookahead_steps)
        for tp_rate in TP_RATES
        for sl_rate in SL_RATES
        for lookahead_steps in LOOKAHEAD_STEPS_LIST
    ]


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


def resolve_exit(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    entry_index: int,
    exit_params: ExitParams,
) -> tuple[int, float, str, float]:
    entry_price = closes[entry_index]
    tp_price = entry_price * (1 + exit_params.tp_rate)
    sl_price = entry_price * (1 - exit_params.sl_rate)
    last_index = min(entry_index + exit_params.lookahead_steps, len(closes) - 1)

    for i in range(entry_index + 1, last_index + 1):
        hit_tp = highs[i] >= tp_price
        hit_sl = lows[i] <= sl_price

        if hit_tp and hit_sl:
            if TIE_BREAKER == "take_profit":
                return i, tp_price, "take_profit", exit_params.tp_rate

            return i, sl_price, "stop_loss", -exit_params.sl_rate

        if hit_tp:
            return i, tp_price, "take_profit", exit_params.tp_rate

        if hit_sl:
            return i, sl_price, "stop_loss", -exit_params.sl_rate

    exit_price = closes[last_index]
    return last_index, exit_price, "time_exit", (exit_price - entry_price) / entry_price


def build_exit_cache(df: pd.DataFrame, exit_params: ExitParams) -> ExitCache:
    highs = df["high"].astype(float).tolist()
    lows = df["low"].astype(float).tolist()
    closes = df["close"].astype(float).tolist()
    exit_indices = [0] * len(df)
    exit_prices = [0.0] * len(df)
    outcomes = ["time_exit"] * len(df)
    gross_returns = [0.0] * len(df)

    for entry_index in range(len(df) - 1):
        exit_index, exit_price, outcome, gross_return = resolve_exit(
            highs,
            lows,
            closes,
            entry_index,
            exit_params,
        )
        exit_indices[entry_index] = exit_index
        exit_prices[entry_index] = exit_price
        outcomes[entry_index] = outcome
        gross_returns[entry_index] = gross_return

    return ExitCache(
        exit_indices=exit_indices,
        exit_prices=exit_prices,
        outcomes=outcomes,
        gross_returns=gross_returns,
    )


def build_base_conditions(df: pd.DataFrame) -> list[Condition]:
    conditions: list[Condition] = []

    for feature in FEATURES:
        if feature not in df.columns:
            continue

        quantiles = df[feature].quantile([0.25, 0.5, 0.75])
        q25 = float(quantiles.loc[0.25])
        q50 = float(quantiles.loc[0.5])
        q75 = float(quantiles.loc[0.75])

        conditions.extend(
            [
                Condition(feature, f"{feature} <= q25", lambda frame, f=feature, v=q25: frame[f] <= v),
                Condition(feature, f"{feature} > q75", lambda frame, f=feature, v=q75: frame[f] > v),
                Condition(feature, f"{feature} between q25-q50", lambda frame, f=feature, lo=q25, hi=q50: (frame[f] > lo) & (frame[f] <= hi)),
                Condition(feature, f"{feature} between q50-q75", lambda frame, f=feature, lo=q50, hi=q75: (frame[f] > lo) & (frame[f] <= hi)),
            ]
        )

        if df[feature].min() < 0 < df[feature].max():
            conditions.extend(
                [
                    Condition(feature, f"{feature} > 0", lambda frame, f=feature: frame[f] > 0),
                    Condition(feature, f"{feature} <= 0", lambda frame, f=feature: frame[f] <= 0),
                ]
            )

    if "rsi_14" in df.columns:
        for low, high in [(0, 35), (35, 45), (45, 55), (55, 65), (65, 100)]:
            conditions.append(
                Condition(
                    "rsi_14",
                    f"{low} <= rsi_14 < {high}",
                    lambda frame, lo=low, hi=high: (frame["rsi_14"] >= lo) & (frame["rsi_14"] < hi),
                )
            )

    if "volume_zscore_24" in df.columns:
        conditions.extend(
            [
                Condition("volume_zscore_24", "volume_zscore_24 > 1", lambda frame: frame["volume_zscore_24"] > 1),
                Condition("volume_zscore_24", "volume_zscore_24 > 2", lambda frame: frame["volume_zscore_24"] > 2),
                Condition("volume_zscore_24", "volume_zscore_24 < -1", lambda frame: frame["volume_zscore_24"] < -1),
            ]
        )

    return conditions


def build_condition_sets(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    base_conditions = build_base_conditions(df)
    condition_sets: list[tuple[str, pd.Series]] = []

    for size in range(2, MAX_COMBINATION_SIZE + 1):
        for condition_group in combinations(base_conditions, size):
            features = [condition.feature for condition in condition_group]

            if len(set(features)) != len(features):
                continue

            mask = pd.Series(True, index=df.index)
            names = []

            for condition in condition_group:
                mask = mask & condition.mask_fn(df).fillna(False)
                names.append(condition.name)

            if int(mask.sum()) < MIN_ENTRY_COUNT:
                continue

            condition_sets.append((" AND ".join(names), mask))

    return condition_sets


def scan_condition_entries(entry_indices: list[int], exit_cache: ExitCache) -> dict:
    outcomes = []
    returns = []
    max_entry_index = len(exit_cache.outcomes) - 1

    for entry_index in entry_indices:
        if entry_index >= max_entry_index:
            continue

        outcome = exit_cache.outcomes[entry_index]
        gross_return = exit_cache.gross_returns[entry_index]
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


def simulate_condition_entries(df: pd.DataFrame, condition_mask: pd.Series, exit_cache: ExitCache) -> dict:
    cash = INITIAL_CASH
    trades = []
    entry_index = 0
    closes = df["close"].astype(float).tolist()
    mask_values = condition_mask.tolist()

    while entry_index < len(df) - 1:
        if not bool(mask_values[entry_index]):
            entry_index += 1
            continue

        entry_price = closes[entry_index]
        notional = min(TRADE_NOTIONAL, cash)

        if notional <= 0:
            break

        exit_index = exit_cache.exit_indices[entry_index]
        exit_price = exit_cache.exit_prices[entry_index]
        outcome = exit_cache.outcomes[entry_index]
        gross_return = exit_cache.gross_returns[entry_index]
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
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
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
    condition_sets = build_condition_sets(df)
    exit_param_grid = build_exit_param_grid()

    print(
        f"{target.symbol} {target.timeframe}: evaluating "
        f"{len(condition_sets)} condition sets x {len(exit_param_grid)} exit parameter sets"
    )

    prepared_condition_sets = [
        (condition_name, mask, [int(index) for index in df.index[mask]])
        for condition_name, mask in condition_sets
    ]

    for exit_params in exit_param_grid:
        exit_cache = build_exit_cache(df, exit_params)

        for condition_name, mask, entry_indices in prepared_condition_sets:
            distribution = scan_condition_entries(entry_indices, exit_cache)
            simulation = simulate_condition_entries(df, mask, exit_cache)
            row = {
                "symbol": target.symbol,
                "timeframe": target.timeframe,
                "condition": condition_name,
                "condition_size": condition_name.count(" AND ") + 1,
                "tp_rate": exit_params.tp_rate,
                "sl_rate": exit_params.sl_rate,
                "lookahead_steps": exit_params.lookahead_steps,
                "reward_risk": exit_params.tp_rate / exit_params.sl_rate if exit_params.sl_rate else 0.0,
                "condition_rate_pct": len(entry_indices) / len(df) * 100 if len(df) else 0.0,
                **distribution,
                **simulation,
            }
            rows.append(row)

    return rows


def main() -> None:
    targets = parse_targets()
    all_rows = []
    skipped_targets = []

    print("===================================")
    print("Combined Condition Report")
    print("===================================")
    print(f"targets: {', '.join(f'{target.symbol}:{target.timeframe}' for target in targets)}")
    print(f"tp_rates: {', '.join(str(value) for value in TP_RATES)}")
    print(f"sl_rates: {', '.join(str(value) for value in SL_RATES)}")
    print(f"lookahead_steps: {', '.join(str(value) for value in LOOKAHEAD_STEPS_LIST)}")
    print(f"exit parameter sets: {len(build_exit_param_grid())}")
    print(f"fee_rate: {FEE_RATE}")
    print(f"features: {', '.join(FEATURES)}")
    print(f"max_combination_size: {MAX_COMBINATION_SIZE}")
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
        raise RuntimeError("No combined condition rows were produced.")

    report_df = pd.DataFrame(all_rows)
    report_df = report_df.sort_values(
        ["avg_net_return", "profit_factor", "total_pnl"],
        ascending=[False, False, False],
    )

    report_path = REPORT_DIR / "combined_condition_report.csv"
    skipped_path = REPORT_DIR / "combined_condition_skipped.csv"
    report_df.to_csv(report_path, index=False)

    if skipped_targets:
        pd.DataFrame(skipped_targets).to_csv(skipped_path, index=False)

    print("\n===================================")
    print("Top Results")
    print("===================================")
    print(report_df.head(30).to_string(index=False))
    print("===================================")
    print(f"combined condition report: {report_path}")
    if skipped_targets:
        print(f"skipped targets: {skipped_path}")
    print("===================================")


if __name__ == "__main__":
    main()
