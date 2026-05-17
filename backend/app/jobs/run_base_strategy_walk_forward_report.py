import os
from pathlib import Path

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


TRAIN_ROWS = int(os.getenv("BASE_STRATEGY_WF_TRAIN_ROWS", os.getenv("WALK_FORWARD_TRAIN_ROWS", "5000")))
TEST_ROWS = int(os.getenv("BASE_STRATEGY_WF_TEST_ROWS", os.getenv("WALK_FORWARD_TEST_ROWS", "1000")))
STEP_ROWS = int(os.getenv("BASE_STRATEGY_WF_STEP_ROWS", os.getenv("WALK_FORWARD_STEP_ROWS", "1000")))
SELECTION_OBJECTIVE = os.getenv("BASE_STRATEGY_SELECTION_OBJECTIVE", "total_pnl")
MIN_TRAIN_TRADES = int(os.getenv("BASE_STRATEGY_MIN_TRAIN_TRADES", "20"))
MIN_TRAIN_TOTAL_PNL = float(os.getenv("BASE_STRATEGY_MIN_TRAIN_TOTAL_PNL", "-999999"))
MIN_TRAIN_PROFIT_FACTOR = float(os.getenv("BASE_STRATEGY_MIN_TRAIN_PROFIT_FACTOR", "0"))
SHORT_SMA_VALUES = parse_int_list("BASE_STRATEGY_SHORT_SMAS", "BASE_STRATEGY_SHORT_SMA", "5,8,10,12")
LONG_SMA_VALUES = parse_int_list("BASE_STRATEGY_LONG_SMAS", "BASE_STRATEGY_LONG_SMA", "30,50,80,100")
RSI_BUY_THRESHOLDS = parse_float_list(
    "BASE_STRATEGY_RSI_BUY_THRESHOLDS",
    "BASE_STRATEGY_RSI_BUY_THRESHOLD",
    "50,55,60,65",
)
MACD_FILTER_VALUES = [
    value.strip() not in {"0", "false", "False", "off", "OFF"}
    for value in os.getenv("BASE_STRATEGY_MACD_FILTERS", "1,0").replace(";", ",").split(",")
    if value.strip()
]
REPORT_DIR = Path(os.getenv("BASE_STRATEGY_REPORT_DIR", "data/reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OBJECTIVE_COLUMNS = {
    "total_pnl",
    "profit_factor",
    "avg_pnl",
    "total_return",
}

if SELECTION_OBJECTIVE not in OBJECTIVE_COLUMNS:
    raise ValueError(f"BASE_STRATEGY_SELECTION_OBJECTIVE must be one of: {sorted(OBJECTIVE_COLUMNS)}")


def ensure_sma_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for period in sorted(set(SHORT_SMA_VALUES + LONG_SMA_VALUES)):
        column = f"sma_{period}"

        if column not in df.columns:
            df[column] = df["close"].rolling(period).mean()

    required = ["rsi_14", "macd", "macd_signal", "macd_hist"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Dataset is missing base strategy columns: {missing}")

    return df


def build_buy_mask(
    df: pd.DataFrame,
    short_sma: int,
    long_sma: int,
    rsi_buy_threshold: float,
    use_macd_filter: bool,
) -> pd.Series:
    short_col = f"sma_{short_sma}"
    long_col = f"sma_{long_sma}"
    prev_diff = df[short_col].shift(1) - df[long_col].shift(1)
    curr_diff = df[short_col] - df[long_col]
    mask = (prev_diff <= 0) & (curr_diff > 0) & (df["rsi_14"] > rsi_buy_threshold)

    if use_macd_filter:
        mask = mask & (df["macd"] > df["macd_signal"]) & (df["macd_hist"] > 0)

    return mask.fillna(False)


def evaluate_strategy(
    df: pd.DataFrame,
    short_sma: int,
    long_sma: int,
    rsi_buy_threshold: float,
    use_macd_filter: bool,
    exit_params: ExitParams,
) -> dict:
    buy_mask = build_buy_mask(df, short_sma, long_sma, rsi_buy_threshold, use_macd_filter)
    entry_indices = [int(index) for index in df.index[buy_mask]]
    exit_cache = build_exit_cache(df, exit_params)
    result = simulate_entry_indices(df, entry_indices, exit_cache)

    return {
        "short_sma": short_sma,
        "long_sma": long_sma,
        "rsi_buy_threshold": rsi_buy_threshold,
        "use_macd_filter": use_macd_filter,
        "tp_rate": exit_params.tp_rate,
        "sl_rate": exit_params.sl_rate,
        "lookahead_steps": exit_params.lookahead_steps,
        "reward_risk": exit_params.tp_rate / exit_params.sl_rate if exit_params.sl_rate else 0.0,
        **result,
    }


def iter_strategy_params():
    for short_sma in SHORT_SMA_VALUES:
        for long_sma in LONG_SMA_VALUES:
            if short_sma >= long_sma:
                continue

            for rsi_buy_threshold in RSI_BUY_THRESHOLDS:
                for use_macd_filter in MACD_FILTER_VALUES:
                    yield short_sma, long_sma, rsi_buy_threshold, use_macd_filter


def select_train_strategy(train_df: pd.DataFrame) -> dict | None:
    best_row = None
    best_key = None

    for exit_params in build_exit_param_grid():
        exit_cache = build_exit_cache(train_df, exit_params)

        for short_sma, long_sma, rsi_buy_threshold, use_macd_filter in iter_strategy_params():
            buy_mask = build_buy_mask(train_df, short_sma, long_sma, rsi_buy_threshold, use_macd_filter)
            entry_indices = [int(index) for index in train_df.index[buy_mask]]
            result = simulate_entry_indices(train_df, entry_indices, exit_cache)
            row = {
                "short_sma": short_sma,
                "long_sma": long_sma,
                "rsi_buy_threshold": rsi_buy_threshold,
                "use_macd_filter": use_macd_filter,
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


def prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def run_target(target) -> list[dict]:
    df = ensure_sma_columns(load_dataset(target))
    rows = []
    fold = 1

    for start in range(0, len(df) - TRAIN_ROWS - TEST_ROWS + 1, STEP_ROWS):
        train_df = df.iloc[start : start + TRAIN_ROWS].reset_index(drop=True)
        test_df = df.iloc[start + TRAIN_ROWS : start + TRAIN_ROWS + TEST_ROWS].reset_index(drop=True)
        selected = select_train_strategy(train_df)

        if not selected:
            print(f"{target.symbol} {target.timeframe} fold={fold}: skipped, no eligible train strategy")
            fold += 1
            continue

        test_result = evaluate_strategy(
            test_df,
            short_sma=selected["short_sma"],
            long_sma=selected["long_sma"],
            rsi_buy_threshold=selected["rsi_buy_threshold"],
            use_macd_filter=selected["use_macd_filter"],
            exit_params=ExitParams(
                tp_rate=selected["tp_rate"],
                sl_rate=selected["sl_rate"],
                lookahead_steps=selected["lookahead_steps"],
            ),
        )
        row = {
            "symbol": target.symbol,
            "timeframe": target.timeframe,
            "fold": fold,
            "train_start": train_df.iloc[0]["open_time"],
            "train_end": train_df.iloc[-1]["open_time"],
            "test_start": test_df.iloc[0]["open_time"],
            "test_end": test_df.iloc[-1]["open_time"],
            "selection_objective": SELECTION_OBJECTIVE,
            **prefixed("train", selected),
            **prefixed("test", test_result),
        }
        rows.append(row)
        print(
            f"{target.symbol} {target.timeframe} fold={fold} "
            f"sma={selected['short_sma']}/{selected['long_sma']} "
            f"rsi>{selected['rsi_buy_threshold']} macd={selected['use_macd_filter']} "
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
    print("Base Strategy Walk-forward Report")
    print("===================================")
    print(f"targets: {', '.join(f'{target.symbol}:{target.timeframe}' for target in targets)}")
    print(f"short_smas: {', '.join(str(value) for value in SHORT_SMA_VALUES)}")
    print(f"long_smas: {', '.join(str(value) for value in LONG_SMA_VALUES)}")
    print(f"rsi_buy_thresholds: {', '.join(str(value) for value in RSI_BUY_THRESHOLDS)}")
    print(f"macd_filters: {', '.join(str(value) for value in MACD_FILTER_VALUES)}")
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
        raise RuntimeError("No base strategy walk-forward rows were produced.")

    report_df = pd.DataFrame(all_rows)
    summary_df = build_summary(report_df)
    folds_path = REPORT_DIR / "base_strategy_walk_forward_folds.csv"
    summary_path = REPORT_DIR / "base_strategy_walk_forward_summary.csv"
    skipped_path = REPORT_DIR / "base_strategy_walk_forward_skipped.csv"
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
