import os
from itertools import combinations
from pathlib import Path

import pandas as pd

from app.jobs.run_combined_condition_report import (
    Condition,
    ExitParams,
    FEATURES,
    FEE_RATE,
    INITIAL_CASH,
    MAX_COMBINATION_SIZE,
    MIN_ENTRY_COUNT,
    build_base_conditions,
    build_exit_cache,
    build_exit_param_grid,
    load_dataset,
    parse_targets,
    scan_condition_entries,
    simulate_condition_entries,
)


TRAIN_ROWS = int(os.getenv("COMBINED_WF_TRAIN_ROWS", os.getenv("WALK_FORWARD_TRAIN_ROWS", "5000")))
TEST_ROWS = int(os.getenv("COMBINED_WF_TEST_ROWS", os.getenv("WALK_FORWARD_TEST_ROWS", "1000")))
STEP_ROWS = int(os.getenv("COMBINED_WF_STEP_ROWS", os.getenv("WALK_FORWARD_STEP_ROWS", "1000")))
SELECTION_OBJECTIVE = os.getenv("COMBINED_WF_SELECTION_OBJECTIVE", "total_pnl")
MIN_TRAIN_ENTRY_COUNT = int(os.getenv("COMBINED_WF_MIN_TRAIN_ENTRY_COUNT", str(MIN_ENTRY_COUNT)))
MIN_TRAIN_TRADES = int(os.getenv("COMBINED_WF_MIN_TRAIN_TRADES", "20"))
MIN_TRAIN_TOTAL_PNL = float(os.getenv("COMBINED_WF_MIN_TRAIN_TOTAL_PNL", "0"))
MIN_TRAIN_PROFIT_FACTOR = float(os.getenv("COMBINED_WF_MIN_TRAIN_PROFIT_FACTOR", "1.0"))
REPORT_DIR = Path(os.getenv("COMBINED_WF_REPORT_DIR", "data/reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OBJECTIVE_COLUMNS = {
    "total_pnl",
    "avg_net_return",
    "profit_factor",
    "total_return",
    "avg_pnl",
}

if SELECTION_OBJECTIVE not in OBJECTIVE_COLUMNS:
    raise ValueError(f"COMBINED_WF_SELECTION_OBJECTIVE must be one of: {sorted(OBJECTIVE_COLUMNS)}")


def build_condition_groups(train_df: pd.DataFrame) -> list[tuple[str, tuple[Condition, ...]]]:
    base_conditions = build_base_conditions(train_df)
    condition_groups: list[tuple[str, tuple[Condition, ...]]] = []

    for size in range(2, MAX_COMBINATION_SIZE + 1):
        for condition_group in combinations(base_conditions, size):
            features = [condition.feature for condition in condition_group]

            if len(set(features)) != len(features):
                continue

            train_mask = build_condition_mask(train_df, condition_group)

            if int(train_mask.sum()) < MIN_TRAIN_ENTRY_COUNT:
                continue

            name = " AND ".join(condition.name for condition in condition_group)
            condition_groups.append((name, condition_group))

    return condition_groups


def build_condition_mask(df: pd.DataFrame, condition_group: tuple[Condition, ...]) -> pd.Series:
    mask = pd.Series(True, index=df.index)

    for condition in condition_group:
        mask = mask & condition.mask_fn(df).fillna(False)

    return mask


def evaluate_candidate(
    df: pd.DataFrame,
    condition_mask: pd.Series,
    exit_params: ExitParams,
) -> dict:
    entry_indices = [int(index) for index in df.index[condition_mask]]
    exit_cache = build_exit_cache(df, exit_params)
    distribution = scan_condition_entries(entry_indices, exit_cache)
    simulation = simulate_condition_entries(df, condition_mask, exit_cache)

    return {
        "tp_rate": exit_params.tp_rate,
        "sl_rate": exit_params.sl_rate,
        "lookahead_steps": exit_params.lookahead_steps,
        "reward_risk": exit_params.tp_rate / exit_params.sl_rate if exit_params.sl_rate else 0.0,
        "condition_rate_pct": len(entry_indices) / len(df) * 100 if len(df) else 0.0,
        **distribution,
        **simulation,
    }


def select_train_candidate(train_df: pd.DataFrame) -> dict | None:
    condition_groups = build_condition_groups(train_df)
    prepared_condition_groups = []

    for condition_name, condition_group in condition_groups:
        train_mask = build_condition_mask(train_df, condition_group)
        entry_indices = [int(index) for index in train_df.index[train_mask]]
        prepared_condition_groups.append((condition_name, condition_group, train_mask, entry_indices))

    exit_param_grid = build_exit_param_grid()
    best_row = None
    best_key = None

    for exit_params in exit_param_grid:
        exit_cache = build_exit_cache(train_df, exit_params)

        for condition_name, condition_group, train_mask, entry_indices in prepared_condition_groups:
            distribution = scan_condition_entries(entry_indices, exit_cache)
            simulation = simulate_condition_entries(train_df, train_mask, exit_cache)
            row = {
                "condition": condition_name,
                "condition_group": condition_group,
                "condition_size": len(condition_group),
                "tp_rate": exit_params.tp_rate,
                "sl_rate": exit_params.sl_rate,
                "lookahead_steps": exit_params.lookahead_steps,
                "reward_risk": exit_params.tp_rate / exit_params.sl_rate if exit_params.sl_rate else 0.0,
                "condition_rate_pct": len(entry_indices) / len(train_df) * 100 if len(train_df) else 0.0,
                **distribution,
                **simulation,
            }

            if (
                row["entry_count"] < MIN_TRAIN_ENTRY_COUNT
                or row["trade_count"] < MIN_TRAIN_TRADES
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
    excluded = {"condition_group"}
    return {f"{prefix}_{key}": value for key, value in values.items() if key not in excluded}


def run_target(target) -> list[dict]:
    df = load_dataset(target)
    rows = []
    fold = 1

    for start in range(0, len(df) - TRAIN_ROWS - TEST_ROWS + 1, STEP_ROWS):
        train_df = df.iloc[start : start + TRAIN_ROWS].reset_index(drop=True)
        test_df = df.iloc[start + TRAIN_ROWS : start + TRAIN_ROWS + TEST_ROWS].reset_index(drop=True)
        selected = select_train_candidate(train_df)

        if not selected:
            print(f"{target.symbol} {target.timeframe} fold={fold}: skipped, no eligible train candidate")
            fold += 1
            continue

        test_mask = build_condition_mask(test_df, selected["condition_group"])
        test_result = evaluate_candidate(
            test_df,
            test_mask,
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
                "avg_test_profit_factor_capped": float(group["test_profit_factor"].clip(upper=10).mean()),
                "folds_profitable": int((group["test_total_pnl"] > 0).sum()),
                "profitable_fold_rate": float((group["test_total_pnl"] > 0).mean() * 100),
                "fold_level_profit_factor": test_gross_profit / test_gross_loss if test_gross_loss > 0 else (999.0 if test_gross_profit > 0 else 0.0),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    targets = parse_targets()
    all_rows = []
    skipped_targets = []

    print("===================================")
    print("Combined Condition Walk-forward Report")
    print("===================================")
    print(f"targets: {', '.join(f'{target.symbol}:{target.timeframe}' for target in targets)}")
    print(f"features: {', '.join(FEATURES)}")
    print(f"fee_rate: {FEE_RATE}")
    print(f"train/test/step rows: {TRAIN_ROWS}/{TEST_ROWS}/{STEP_ROWS}")
    print(f"selection_objective: {SELECTION_OBJECTIVE}")
    print(f"min_train_entry_count: {MIN_TRAIN_ENTRY_COUNT}")
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
        raise RuntimeError("No combined condition walk-forward rows were produced.")

    report_df = pd.DataFrame(all_rows)
    summary_df = build_summary(report_df)
    folds_path = REPORT_DIR / "combined_condition_walk_forward_folds.csv"
    summary_path = REPORT_DIR / "combined_condition_walk_forward_summary.csv"
    skipped_path = REPORT_DIR / "combined_condition_walk_forward_skipped.csv"
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
