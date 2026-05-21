import os
from pathlib import Path
from typing import Callable

import pandas as pd

from app.jobs.run_base_filter_walk_forward_report import simulate_entry_indices
from app.jobs.run_combined_condition_report import (
    FEE_RATE,
    INITIAL_CASH,
    ExitParams,
    build_exit_cache,
    build_exit_param_grid,
    load_dataset,
    parse_float_list,
    parse_int_list,
    parse_targets,
)
from app.jobs.run_combined_condition_walk_forward_report import prefixed


TRAIN_ROWS = int(os.getenv("BREAKOUT_WF_TRAIN_ROWS", os.getenv("WALK_FORWARD_TRAIN_ROWS", "5000")))
TEST_ROWS = int(os.getenv("BREAKOUT_WF_TEST_ROWS", os.getenv("WALK_FORWARD_TEST_ROWS", "1000")))
STEP_ROWS = int(os.getenv("BREAKOUT_WF_STEP_ROWS", os.getenv("WALK_FORWARD_STEP_ROWS", "1000")))
SELECTION_OBJECTIVE = os.getenv("BREAKOUT_SELECTION_OBJECTIVE", "total_pnl")
MIN_TRAIN_TRADES = int(os.getenv("BREAKOUT_MIN_TRAIN_TRADES", "20"))
MIN_TRAIN_TOTAL_PNL = float(os.getenv("BREAKOUT_MIN_TRAIN_TOTAL_PNL", "-999999"))
MIN_TRAIN_PROFIT_FACTOR = float(os.getenv("BREAKOUT_MIN_TRAIN_PROFIT_FACTOR", "0"))
BREAKOUT_WINDOWS = parse_int_list("BREAKOUT_WINDOWS", "BREAKOUT_WINDOW", "24,48,72,96")
RSI_MIN_VALUES = parse_float_list("BREAKOUT_RSI_MIN_VALUES", "BREAKOUT_RSI_MIN", "0")
FILTER_NAMES = [
    value.strip()
    for value in os.getenv(
        "BREAKOUT_FILTERS",
        "none,atr_14 > q75,volume_zscore_24 > 0,atr_14 > q75 AND volume_zscore_24 > 0,rolling_std_24 > q75 AND atr_14 > q75",
    )
    .replace(";", ",")
    .split(",")
    if value.strip()
]
REPORT_DIR = Path(os.getenv("BREAKOUT_REPORT_DIR", "data/reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OBJECTIVE_COLUMNS = {
    "total_pnl",
    "profit_factor",
    "avg_pnl",
    "total_return",
}

if SELECTION_OBJECTIVE not in OBJECTIVE_COLUMNS:
    raise ValueError(f"BREAKOUT_SELECTION_OBJECTIVE must be one of: {sorted(OBJECTIVE_COLUMNS)}")


def ensure_breakout_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["high", "close"]

    if any(value > 0 for value in RSI_MIN_VALUES):
        required.append("rsi_14")

    for filter_name in FILTER_NAMES:
        if "atr_14" in filter_name:
            required.append("atr_14")
        if "volume_zscore_24" in filter_name:
            required.append("volume_zscore_24")
        if "rolling_std_24" in filter_name:
            required.append("rolling_std_24")

    missing = [column for column in sorted(set(required)) if column not in df.columns]

    if missing:
        raise ValueError(f"Dataset is missing breakout strategy columns: {missing}")

    return df


def build_breakout_mask(df: pd.DataFrame, breakout_window: int, rsi_min: float) -> pd.Series:
    prior_high = df["high"].rolling(breakout_window).max().shift(1)
    mask = df["close"] > prior_high

    if rsi_min > 0:
        mask = mask & (df["rsi_14"] >= rsi_min)

    return mask.fillna(False)


def build_filter_mask_fn(train_df: pd.DataFrame, filter_name: str) -> Callable[[pd.DataFrame], pd.Series]:
    if filter_name == "none":
        return lambda frame: pd.Series(True, index=frame.index)

    parts = [part.strip() for part in filter_name.split(" AND ") if part.strip()]
    thresholds = {}

    for part in parts:
        if part == "atr_14 > q75":
            thresholds[part] = float(train_df["atr_14"].quantile(0.75))
        elif part == "rolling_std_24 > q75":
            thresholds[part] = float(train_df["rolling_std_24"].quantile(0.75))
        elif part == "volume_zscore_24 > q75":
            thresholds[part] = float(train_df["volume_zscore_24"].quantile(0.75))
        elif part == "volume_zscore_24 > 0":
            thresholds[part] = 0.0
        else:
            raise ValueError(f"Unsupported BREAKOUT_FILTERS item: {part}")

    def mask_fn(frame: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=frame.index)

        for part, threshold in thresholds.items():
            feature = part.split(" > ", 1)[0]
            mask = mask & (frame[feature] > threshold)

        return mask.fillna(False)

    return mask_fn


def evaluate_strategy(
    df: pd.DataFrame,
    breakout_window: int,
    rsi_min: float,
    filter_name: str,
    filter_mask_fn: Callable[[pd.DataFrame], pd.Series],
    exit_params: ExitParams,
) -> dict:
    base_mask = build_breakout_mask(df, breakout_window, rsi_min)
    filter_mask = filter_mask_fn(df)
    entry_mask = base_mask & filter_mask
    entry_indices = [int(index) for index in df.index[entry_mask]]
    exit_cache = build_exit_cache(df, exit_params)
    result = simulate_entry_indices(df, entry_indices, exit_cache)

    return {
        "breakout_window": breakout_window,
        "rsi_min": rsi_min,
        "filter_name": filter_name,
        "tp_rate": exit_params.tp_rate,
        "sl_rate": exit_params.sl_rate,
        "lookahead_steps": exit_params.lookahead_steps,
        "reward_risk": exit_params.tp_rate / exit_params.sl_rate if exit_params.sl_rate else 0.0,
        **result,
    }


def iter_strategy_params(train_df: pd.DataFrame):
    for breakout_window in BREAKOUT_WINDOWS:
        for rsi_min in RSI_MIN_VALUES:
            for filter_name in FILTER_NAMES:
                yield breakout_window, rsi_min, filter_name, build_filter_mask_fn(train_df, filter_name)


def select_train_strategy(train_df: pd.DataFrame) -> dict | None:
    best_row = None
    best_key = None

    for exit_params in build_exit_param_grid():
        exit_cache = build_exit_cache(train_df, exit_params)

        for breakout_window, rsi_min, filter_name, filter_mask_fn in iter_strategy_params(train_df):
            base_mask = build_breakout_mask(train_df, breakout_window, rsi_min)
            filter_mask = filter_mask_fn(train_df)
            entry_indices = [int(index) for index in train_df.index[base_mask & filter_mask]]
            result = simulate_entry_indices(train_df, entry_indices, exit_cache)
            row = {
                "breakout_window": breakout_window,
                "rsi_min": rsi_min,
                "filter_name": filter_name,
                "filter_mask_fn": filter_mask_fn,
                "tp_rate": exit_params.tp_rate,
                "sl_rate": exit_params.sl_rate,
                "lookahead_steps": exit_params.lookahead_steps,
                "reward_risk": exit_params.tp_rate / exit_params.sl_rate if exit_params.sl_rate else 0.0,
                **result,
            }

            if (
                row["trade_count"] < MIN_TRAIN_TRADES
                or row["total_pnl"] < MIN_TRAIN_TOTAL_PNL
                or row["profit_factor"] < MIN_TRAIN_PROFIT_FACTOR
            ):
                continue

            key = (
                row[SELECTION_OBJECTIVE],
                row["profit_factor"],
                row["total_pnl"],
                row["trade_count"],
            )

            if best_key is None or key > best_key:
                best_key = key
                best_row = row

    return best_row


def run_target(target) -> list[dict]:
    df = ensure_breakout_columns(load_dataset(target))
    rows = []
    fold = 1

    for start in range(0, len(df) - TRAIN_ROWS - TEST_ROWS + 1, STEP_ROWS):
        train_df = df.iloc[start : start + TRAIN_ROWS].reset_index(drop=True)
        test_df = df.iloc[start + TRAIN_ROWS : start + TRAIN_ROWS + TEST_ROWS].reset_index(drop=True)
        selected = select_train_strategy(train_df)

        if not selected:
            print(f"{target.symbol} {target.timeframe} fold={fold}: skipped, no eligible train breakout")
            fold += 1
            continue

        test_result = evaluate_strategy(
            test_df,
            breakout_window=selected["breakout_window"],
            rsi_min=selected["rsi_min"],
            filter_name=selected["filter_name"],
            filter_mask_fn=selected["filter_mask_fn"],
            exit_params=ExitParams(
                tp_rate=selected["tp_rate"],
                sl_rate=selected["sl_rate"],
                lookahead_steps=selected["lookahead_steps"],
            ),
        )
        train_row = {key: value for key, value in selected.items() if key != "filter_mask_fn"}
        row = {
            "symbol": target.symbol,
            "timeframe": target.timeframe,
            "fold": fold,
            "train_start": train_df.iloc[0]["open_time"],
            "train_end": train_df.iloc[-1]["open_time"],
            "test_start": test_df.iloc[0]["open_time"],
            "test_end": test_df.iloc[-1]["open_time"],
            "selection_objective": SELECTION_OBJECTIVE,
            **prefixed("train", train_row),
            **prefixed("test", test_result),
        }
        rows.append(row)
        print(
            f"{target.symbol} {target.timeframe} fold={fold} "
            f"breakout={selected['breakout_window']} rsi_min={selected['rsi_min']} "
            f"filter='{selected['filter_name']}' "
            f"tp/sl/lookahead={selected['tp_rate']}/{selected['sl_rate']}/{selected['lookahead_steps']} "
            f"train_pnl={selected['total_pnl']:.2f} test_pnl={test_result['total_pnl']:.2f} "
            f"test_trades={test_result['trade_count']}"
        )
        fold += 1

    return rows


def build_summary(report_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (symbol, timeframe), group in report_df.groupby(["symbol", "timeframe"], sort=False):
        test_trade_count = int(group["test_trade_count"].sum())
        test_total_pnl = float(group["test_total_pnl"].sum())
        test_gross_profit = float(group["test_gross_profit"].sum())
        test_gross_loss = float(group["test_gross_loss"].sum())
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "folds": int(len(group)),
                "train_rows": TRAIN_ROWS,
                "test_rows": TEST_ROWS,
                "step_rows": STEP_ROWS,
                "selection_objective": SELECTION_OBJECTIVE,
                "fee_rate": FEE_RATE,
                "initial_cash": INITIAL_CASH,
                "test_trade_count": test_trade_count,
                "test_total_pnl": test_total_pnl,
                "test_avg_pnl": test_total_pnl / test_trade_count if test_trade_count else 0.0,
                "avg_test_win_rate": float(group["test_win_rate"].mean()),
                "test_profit_factor": test_gross_profit / test_gross_loss if test_gross_loss > 0 else (999.0 if test_gross_profit > 0 else 0.0),
                "profitable_folds": int((group["test_total_pnl"] > 0).sum()),
                "profitable_fold_rate": float((group["test_total_pnl"] > 0).mean() * 100),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    targets = parse_targets()
    all_rows = []
    skipped_targets = []

    print("===================================")
    print("Breakout Strategy Walk-forward Report")
    print("===================================")
    print(f"targets: {', '.join(f'{target.symbol}:{target.timeframe}' for target in targets)}")
    print(f"breakout_windows: {', '.join(str(value) for value in BREAKOUT_WINDOWS)}")
    print(f"rsi_min_values: {', '.join(str(value) for value in RSI_MIN_VALUES)}")
    print(f"filters: {', '.join(FILTER_NAMES)}")
    print(f"train/test/step rows: {TRAIN_ROWS}/{TEST_ROWS}/{STEP_ROWS}")
    print(f"selection_objective: {SELECTION_OBJECTIVE}")
    print(f"min_train_trades: {MIN_TRAIN_TRADES}")
    print(f"min_train_total_pnl: {MIN_TRAIN_TOTAL_PNL}")
    print(f"min_train_profit_factor: {MIN_TRAIN_PROFIT_FACTOR}")
    print(f"exit parameter sets: {len(build_exit_param_grid())}")
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
        raise RuntimeError("No breakout strategy walk-forward rows were produced.")

    report_df = pd.DataFrame(all_rows)
    summary_df = build_summary(report_df)
    folds_path = REPORT_DIR / "breakout_strategy_walk_forward_folds.csv"
    summary_path = REPORT_DIR / "breakout_strategy_walk_forward_summary.csv"
    skipped_path = REPORT_DIR / "breakout_strategy_walk_forward_skipped.csv"
    report_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    if skipped_targets:
        pd.DataFrame(skipped_targets).to_csv(skipped_path, index=False)

    print("\n===================================")
    print("Summary")
    print("===================================")
    print(summary_df.to_string(index=False))
    print("===================================")
    print(f"fold report: {folds_path}")
    print(f"summary report: {summary_path}")
    if skipped_targets:
        print(f"skipped targets: {skipped_path}")
    print("===================================")


if __name__ == "__main__":
    main()
