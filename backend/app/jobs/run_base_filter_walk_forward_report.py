import os
from pathlib import Path

import pandas as pd

from app.jobs.run_combined_condition_report import (
    FEE_RATE,
    INITIAL_CASH,
    TRADE_NOTIONAL,
    ExitParams,
    build_exit_cache,
    build_exit_param_grid,
    load_dataset,
    parse_targets,
)
from app.jobs.run_combined_condition_walk_forward_report import (
    build_condition_groups,
    build_condition_mask,
    prefixed,
)


TRAIN_ROWS = int(os.getenv("BASE_FILTER_WF_TRAIN_ROWS", os.getenv("WALK_FORWARD_TRAIN_ROWS", "5000")))
TEST_ROWS = int(os.getenv("BASE_FILTER_WF_TEST_ROWS", os.getenv("WALK_FORWARD_TEST_ROWS", "1000")))
STEP_ROWS = int(os.getenv("BASE_FILTER_WF_STEP_ROWS", os.getenv("WALK_FORWARD_STEP_ROWS", "1000")))
SELECTION_OBJECTIVE = os.getenv("BASE_FILTER_SELECTION_OBJECTIVE", "delta_pnl")
MIN_BASE_TRAIN_TRADES = int(os.getenv("BASE_FILTER_MIN_BASE_TRAIN_TRADES", "20"))
MIN_FILTERED_TRAIN_TRADES = int(os.getenv("BASE_FILTER_MIN_FILTERED_TRAIN_TRADES", "10"))
MIN_TRAIN_DELTA_PNL = float(os.getenv("BASE_FILTER_MIN_TRAIN_DELTA_PNL", "0"))
MIN_TRAIN_FILTERED_PROFIT_FACTOR = float(os.getenv("BASE_FILTER_MIN_TRAIN_FILTERED_PROFIT_FACTOR", "0"))
SHORT_SMA = int(os.getenv("BASE_FILTER_SHORT_SMA", "5"))
LONG_SMA = int(os.getenv("BASE_FILTER_LONG_SMA", "30"))
RSI_BUY_THRESHOLD = float(os.getenv("BASE_FILTER_RSI_BUY_THRESHOLD", "60"))
USE_MACD_FILTER = os.getenv("BASE_FILTER_USE_MACD_FILTER", "1") != "0"
REPORT_DIR = Path(os.getenv("BASE_FILTER_REPORT_DIR", "data/reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OBJECTIVE_COLUMNS = {
    "delta_pnl",
    "filtered_total_pnl",
    "filtered_profit_factor",
    "filtered_avg_pnl",
}

if SELECTION_OBJECTIVE not in OBJECTIVE_COLUMNS:
    raise ValueError(f"BASE_FILTER_SELECTION_OBJECTIVE must be one of: {sorted(OBJECTIVE_COLUMNS)}")


def ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    short_col = f"sma_{SHORT_SMA}"
    long_col = f"sma_{LONG_SMA}"

    if short_col not in df.columns:
        df[short_col] = df["close"].rolling(SHORT_SMA).mean()

    if long_col not in df.columns:
        df[long_col] = df["close"].rolling(LONG_SMA).mean()

    required = [short_col, long_col, "rsi_14"]

    if USE_MACD_FILTER:
        required.extend(["macd", "macd_signal", "macd_hist"])

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Dataset is missing base signal columns: {missing}")

    return df


def build_base_buy_mask(df: pd.DataFrame) -> pd.Series:
    df = ensure_base_columns(df)
    short_col = f"sma_{SHORT_SMA}"
    long_col = f"sma_{LONG_SMA}"
    prev_diff = df[short_col].shift(1) - df[long_col].shift(1)
    curr_diff = df[short_col] - df[long_col]
    mask = (prev_diff <= 0) & (curr_diff > 0) & (df["rsi_14"] > RSI_BUY_THRESHOLD)

    if USE_MACD_FILTER:
        mask = (
            mask
            & (df["macd"] > df["macd_signal"])
            & (df["macd_hist"] > 0)
        )

    return mask.fillna(False)


def simulate_entry_indices(df: pd.DataFrame, entry_indices: list[int], exit_cache) -> dict:
    cash = INITIAL_CASH
    trades = []
    entry_pointer = 0
    closes = df["close"].astype(float).tolist()
    sorted_indices = sorted(index for index in entry_indices if index < len(df) - 1)

    while entry_pointer < len(sorted_indices):
        entry_index = sorted_indices[entry_pointer]
        entry_price = closes[entry_index]
        notional = min(TRADE_NOTIONAL, cash)

        if notional <= 0:
            break

        exit_index = exit_cache.exit_indices[entry_index]
        exit_price = exit_cache.exit_prices[entry_index]
        outcome = exit_cache.outcomes[entry_index]
        entry_fee = notional * FEE_RATE
        qty = (notional - entry_fee) / entry_price
        gross = qty * exit_price
        exit_fee = gross * FEE_RATE
        pnl = gross - exit_fee - notional
        cash += pnl
        trades.append({"outcome": outcome, "pnl": pnl})

        entry_pointer += 1

        while entry_pointer < len(sorted_indices) and sorted_indices[entry_pointer] <= exit_index:
            entry_pointer += 1

    trade_count = len(trades)
    wins = sum(1 for trade in trades if trade["pnl"] > 0)
    gross_profit = sum(trade["pnl"] for trade in trades if trade["pnl"] > 0)
    gross_loss = -sum(trade["pnl"] for trade in trades if trade["pnl"] < 0)
    total_pnl = sum(trade["pnl"] for trade in trades)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    return {
        "signal_count": len(sorted_indices),
        "trade_count": trade_count,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / trade_count if trade_count else 0.0,
        "win_rate": wins / trade_count * 100 if trade_count else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "final_equity": cash,
        "total_return": (cash - INITIAL_CASH) / INITIAL_CASH * 100,
        "take_profit_count": sum(1 for trade in trades if trade["outcome"] == "take_profit"),
        "stop_loss_count": sum(1 for trade in trades if trade["outcome"] == "stop_loss"),
        "time_exit_count": sum(1 for trade in trades if trade["outcome"] == "time_exit"),
    }


def evaluate_filter(
    df: pd.DataFrame,
    base_mask: pd.Series,
    filter_mask: pd.Series,
    exit_params: ExitParams,
) -> dict:
    exit_cache = build_exit_cache(df, exit_params)
    base_indices = [int(index) for index in df.index[base_mask]]
    filtered_mask = base_mask & filter_mask
    filtered_indices = [int(index) for index in df.index[filtered_mask]]
    base_result = simulate_entry_indices(df, base_indices, exit_cache)
    filtered_result = simulate_entry_indices(df, filtered_indices, exit_cache)

    return {
        "tp_rate": exit_params.tp_rate,
        "sl_rate": exit_params.sl_rate,
        "lookahead_steps": exit_params.lookahead_steps,
        "base_signal_count": base_result["signal_count"],
        "filtered_signal_count": filtered_result["signal_count"],
        "blocked_signal_count": base_result["signal_count"] - filtered_result["signal_count"],
        "blocked_trade_count": base_result["trade_count"] - filtered_result["trade_count"],
        "delta_pnl": filtered_result["total_pnl"] - base_result["total_pnl"],
        "delta_avg_pnl": filtered_result["avg_pnl"] - base_result["avg_pnl"],
        **prefixed("base", base_result),
        **prefixed("filtered", filtered_result),
    }


def select_train_filter(train_df: pd.DataFrame) -> dict | None:
    condition_groups = build_condition_groups(train_df)
    prepared = []

    for condition_name, condition_group in condition_groups:
        prepared.append((condition_name, condition_group, build_condition_mask(train_df, condition_group)))

    base_mask = build_base_buy_mask(train_df)
    best_row = None
    best_key = None

    for exit_params in build_exit_param_grid():
        exit_cache = build_exit_cache(train_df, exit_params)
        base_indices = [int(index) for index in train_df.index[base_mask]]
        base_result = simulate_entry_indices(train_df, base_indices, exit_cache)

        if base_result["trade_count"] < MIN_BASE_TRAIN_TRADES:
            continue

        for condition_name, condition_group, filter_mask in prepared:
            filtered_mask = base_mask & filter_mask
            filtered_indices = [int(index) for index in train_df.index[filtered_mask]]
            filtered_result = simulate_entry_indices(train_df, filtered_indices, exit_cache)
            delta_pnl = filtered_result["total_pnl"] - base_result["total_pnl"]
            row = {
                "condition": condition_name,
                "condition_group": condition_group,
                "condition_size": len(condition_group),
                "tp_rate": exit_params.tp_rate,
                "sl_rate": exit_params.sl_rate,
                "lookahead_steps": exit_params.lookahead_steps,
                "base_signal_count": base_result["signal_count"],
                "filtered_signal_count": filtered_result["signal_count"],
                "blocked_signal_count": base_result["signal_count"] - filtered_result["signal_count"],
                "blocked_trade_count": base_result["trade_count"] - filtered_result["trade_count"],
                "delta_pnl": delta_pnl,
                "delta_avg_pnl": filtered_result["avg_pnl"] - base_result["avg_pnl"],
                **prefixed("base", base_result),
                **prefixed("filtered", filtered_result),
            }

            if (
                row["filtered_trade_count"] < MIN_FILTERED_TRAIN_TRADES
                or row["delta_pnl"] < MIN_TRAIN_DELTA_PNL
                or row["filtered_profit_factor"] < MIN_TRAIN_FILTERED_PROFIT_FACTOR
            ):
                continue

            key = (
                row[SELECTION_OBJECTIVE],
                row["delta_pnl"],
                row["filtered_profit_factor"],
                row["filtered_trade_count"],
            )

            if best_key is None or key > best_key:
                best_key = key
                best_row = row

    return best_row


def run_target(target) -> list[dict]:
    df = ensure_base_columns(load_dataset(target))
    rows = []
    fold = 1

    for start in range(0, len(df) - TRAIN_ROWS - TEST_ROWS + 1, STEP_ROWS):
        train_df = df.iloc[start : start + TRAIN_ROWS].reset_index(drop=True)
        test_df = df.iloc[start + TRAIN_ROWS : start + TRAIN_ROWS + TEST_ROWS].reset_index(drop=True)
        selected = select_train_filter(train_df)

        if not selected:
            print(f"{target.symbol} {target.timeframe} fold={fold}: skipped, no eligible train filter")
            fold += 1
            continue

        test_base_mask = build_base_buy_mask(test_df)
        test_filter_mask = build_condition_mask(test_df, selected["condition_group"])
        test_result = evaluate_filter(
            test_df,
            test_base_mask,
            test_filter_mask,
            ExitParams(
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
            "condition": selected["condition"],
            "condition_size": selected["condition_size"],
            **prefixed("train", selected),
            **prefixed("test", test_result),
        }
        rows.append(row)
        print(
            f"{target.symbol} {target.timeframe} fold={fold} "
            f"condition='{selected['condition']}' "
            f"tp/sl/lookahead={selected['tp_rate']}/{selected['sl_rate']}/{selected['lookahead_steps']} "
            f"train_delta={selected['delta_pnl']:.2f} test_delta={test_result['delta_pnl']:.2f} "
            f"base_pnl={test_result['base_total_pnl']:.2f} filtered_pnl={test_result['filtered_total_pnl']:.2f} "
            f"filtered_trades={test_result['filtered_trade_count']}"
        )
        fold += 1

    return rows


def build_summary(report_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (symbol, timeframe), group in report_df.groupby(["symbol", "timeframe"], sort=False):
        base_total_pnl = float(group["test_base_total_pnl"].sum())
        filtered_total_pnl = float(group["test_filtered_total_pnl"].sum())
        base_gross_profit = float(group["test_base_gross_profit"].sum())
        base_gross_loss = float(group["test_base_gross_loss"].sum())
        filtered_gross_profit = float(group["test_filtered_gross_profit"].sum())
        filtered_gross_loss = float(group["test_filtered_gross_loss"].sum())
        filtered_trade_count = int(group["test_filtered_trade_count"].sum())
        base_trade_count = int(group["test_base_trade_count"].sum())

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
                "base_trade_count": base_trade_count,
                "filtered_trade_count": filtered_trade_count,
                "blocked_trade_count": int(group["test_blocked_trade_count"].sum()),
                "base_total_pnl": base_total_pnl,
                "filtered_total_pnl": filtered_total_pnl,
                "delta_pnl": filtered_total_pnl - base_total_pnl,
                "base_avg_pnl": base_total_pnl / base_trade_count if base_trade_count else 0.0,
                "filtered_avg_pnl": filtered_total_pnl / filtered_trade_count if filtered_trade_count else 0.0,
                "avg_base_win_rate": float(group["test_base_win_rate"].mean()),
                "avg_filtered_win_rate": float(group["test_filtered_win_rate"].mean()),
                "base_profit_factor": base_gross_profit / base_gross_loss if base_gross_loss > 0 else (999.0 if base_gross_profit > 0 else 0.0),
                "filtered_profit_factor": filtered_gross_profit / filtered_gross_loss if filtered_gross_loss > 0 else (999.0 if filtered_gross_profit > 0 else 0.0),
                "delta_positive_folds": int((group["test_delta_pnl"] > 0).sum()),
                "delta_positive_fold_rate": float((group["test_delta_pnl"] > 0).mean() * 100),
                "filtered_profitable_folds": int((group["test_filtered_total_pnl"] > 0).sum()),
                "filtered_profitable_fold_rate": float((group["test_filtered_total_pnl"] > 0).mean() * 100),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    targets = parse_targets()
    all_rows = []
    skipped_targets = []

    print("===================================")
    print("Base Filter Walk-forward Report")
    print("===================================")
    print(f"targets: {', '.join(f'{target.symbol}:{target.timeframe}' for target in targets)}")
    print(f"base_signal: SMA{SHORT_SMA}/SMA{LONG_SMA} cross, rsi>{RSI_BUY_THRESHOLD}, macd_filter={USE_MACD_FILTER}")
    print(f"train/test/step rows: {TRAIN_ROWS}/{TEST_ROWS}/{STEP_ROWS}")
    print(f"selection_objective: {SELECTION_OBJECTIVE}")
    print(f"min_base_train_trades: {MIN_BASE_TRAIN_TRADES}")
    print(f"min_filtered_train_trades: {MIN_FILTERED_TRAIN_TRADES}")
    print(f"min_train_delta_pnl: {MIN_TRAIN_DELTA_PNL}")
    print(f"min_train_filtered_profit_factor: {MIN_TRAIN_FILTERED_PROFIT_FACTOR}")
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
        raise RuntimeError("No base filter walk-forward rows were produced.")

    report_df = pd.DataFrame(all_rows)
    summary_df = build_summary(report_df)
    folds_path = REPORT_DIR / "base_filter_walk_forward_folds.csv"
    summary_path = REPORT_DIR / "base_filter_walk_forward_summary.csv"
    skipped_path = REPORT_DIR / "base_filter_walk_forward_skipped.csv"
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
