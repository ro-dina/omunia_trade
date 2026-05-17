import os
from itertools import combinations
from pathlib import Path

import pandas as pd

from app.jobs.run_base_filter_walk_forward_report import (
    build_base_buy_mask,
    ensure_base_columns,
    evaluate_filter,
)
from app.jobs.run_combined_condition_report import (
    Condition,
    ExitParams,
    FEATURES,
    FEE_RATE,
    MAX_COMBINATION_SIZE,
    build_base_conditions,
    load_dataset,
    parse_float_list,
    parse_int_list,
    parse_targets,
)
from app.jobs.run_combined_condition_walk_forward_report import build_condition_mask


TRAIN_ROWS = int(os.getenv("FIXED_FILTER_TRAIN_ROWS", os.getenv("WALK_FORWARD_TRAIN_ROWS", "5000")))
TEST_ROWS = int(os.getenv("FIXED_FILTER_TEST_ROWS", os.getenv("WALK_FORWARD_TEST_ROWS", "1000")))
STEP_ROWS = int(os.getenv("FIXED_FILTER_STEP_ROWS", os.getenv("WALK_FORWARD_STEP_ROWS", "1000")))
REPORT_DIR = Path(os.getenv("FIXED_FILTER_REPORT_DIR", "data/reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TP_RATES = parse_float_list("FIXED_FILTER_TP_RATES", "FIXED_FILTER_TP_RATE", os.getenv("BASE_FILTER_TP_RATE", "0.030"))
SL_RATES = parse_float_list("FIXED_FILTER_SL_RATES", "FIXED_FILTER_SL_RATE", os.getenv("BASE_FILTER_SL_RATE", "0.012"))
LOOKAHEAD_STEPS_LIST = parse_int_list(
    "FIXED_FILTER_LOOKAHEAD_STEPS_LIST",
    "FIXED_FILTER_LOOKAHEAD_STEPS",
    os.getenv("BASE_FILTER_LOOKAHEAD_STEPS", "72"),
)

DEFAULT_FILTERS = (
    "volume_zscore_24 <= 0 AND rsi_14 > q75,"
    "volume_zscore_24 > 0 AND rsi_14 > q75,"
    "volume_zscore_24 between q50-q75 AND rsi_14 > q75,"
    "rolling_std_24 > q75 AND atr_14 > q75"
)


def parse_filter_names() -> list[str]:
    raw = os.getenv("FIXED_FILTERS", DEFAULT_FILTERS)
    filters = [value.strip() for value in raw.replace(";", ",").split(",") if value.strip()]

    if not filters:
        raise ValueError("No fixed filters were provided.")

    return filters


def build_fixed_exit_param_grid() -> list[ExitParams]:
    return [
        ExitParams(tp_rate=tp_rate, sl_rate=sl_rate, lookahead_steps=lookahead_steps)
        for tp_rate in TP_RATES
        for sl_rate in SL_RATES
        for lookahead_steps in LOOKAHEAD_STEPS_LIST
    ]


def build_all_condition_groups(train_df: pd.DataFrame) -> dict[str, tuple[Condition, ...]]:
    base_conditions = build_base_conditions(train_df)
    groups: dict[str, tuple[Condition, ...]] = {}

    for size in range(2, MAX_COMBINATION_SIZE + 1):
        for condition_group in combinations(base_conditions, size):
            features = [condition.feature for condition in condition_group]

            if len(set(features)) != len(features):
                continue

            name = " AND ".join(condition.name for condition in condition_group)
            groups[name] = condition_group

    return groups


def evaluate_fold_filters(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    filter_names: list[str],
) -> list[dict]:
    rows = []
    condition_groups = build_all_condition_groups(train_df)
    test_base_mask = build_base_buy_mask(test_df)

    for exit_params in build_fixed_exit_param_grid():
        for filter_name in filter_names:
            condition_group = condition_groups.get(filter_name)

            if condition_group is None:
                rows.append(
                    {
                        "filter_name": filter_name,
                        "skipped": True,
                        "skip_reason": "filter condition not available",
                    }
                )
                continue

            test_filter_mask = build_condition_mask(test_df, condition_group)
            result = evaluate_filter(test_df, test_base_mask, test_filter_mask, exit_params)
            rows.append(
                {
                    "filter_name": filter_name,
                    "skipped": False,
                    "skip_reason": "",
                    **result,
                }
            )

    return rows


def run_target(target, filter_names: list[str]) -> list[dict]:
    df = ensure_base_columns(load_dataset(target))
    rows = []
    fold = 1

    for start in range(0, len(df) - TRAIN_ROWS - TEST_ROWS + 1, STEP_ROWS):
        train_df = df.iloc[start : start + TRAIN_ROWS].reset_index(drop=True)
        test_df = df.iloc[start + TRAIN_ROWS : start + TRAIN_ROWS + TEST_ROWS].reset_index(drop=True)
        fold_rows = evaluate_fold_filters(train_df, test_df, filter_names)

        for fold_row in fold_rows:
            rows.append(
                {
                    "symbol": target.symbol,
                    "timeframe": target.timeframe,
                    "fold": fold,
                    "train_start": train_df.iloc[0]["open_time"],
                    "train_end": train_df.iloc[-1]["open_time"],
                    "test_start": test_df.iloc[0]["open_time"],
                    "test_end": test_df.iloc[-1]["open_time"],
                    **fold_row,
                }
            )

        print(f"{target.symbol} {target.timeframe} fold={fold}: evaluated {len(fold_rows)} fixed filters")
        fold += 1

    return rows


def build_summary(report_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid_df = report_df[~report_df["skipped"]].copy()

    for (symbol, timeframe, filter_name, tp_rate, sl_rate, lookahead_steps), group in valid_df.groupby(
        ["symbol", "timeframe", "filter_name", "tp_rate", "sl_rate", "lookahead_steps"],
        sort=False,
    ):
        base_total_pnl = float(group["base_total_pnl"].sum())
        filtered_total_pnl = float(group["filtered_total_pnl"].sum())
        base_gross_profit = float(group["base_gross_profit"].sum())
        base_gross_loss = float(group["base_gross_loss"].sum())
        filtered_gross_profit = float(group["filtered_gross_profit"].sum())
        filtered_gross_loss = float(group["filtered_gross_loss"].sum())
        base_trade_count = int(group["base_trade_count"].sum())
        filtered_trade_count = int(group["filtered_trade_count"].sum())

        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "filter_name": filter_name,
                "tp_rate": tp_rate,
                "sl_rate": sl_rate,
                "lookahead_steps": lookahead_steps,
                "folds": int(len(group)),
                "train_rows": TRAIN_ROWS,
                "test_rows": TEST_ROWS,
                "step_rows": STEP_ROWS,
                "features": ",".join(FEATURES),
                "fee_rate": FEE_RATE,
                "base_trade_count": base_trade_count,
                "filtered_trade_count": filtered_trade_count,
                "blocked_trade_count": int(group["blocked_trade_count"].sum()),
                "base_total_pnl": base_total_pnl,
                "filtered_total_pnl": filtered_total_pnl,
                "delta_pnl": filtered_total_pnl - base_total_pnl,
                "base_avg_pnl": base_total_pnl / base_trade_count if base_trade_count else 0.0,
                "filtered_avg_pnl": filtered_total_pnl / filtered_trade_count if filtered_trade_count else 0.0,
                "avg_base_win_rate": float(group["base_win_rate"].mean()),
                "avg_filtered_win_rate": float(group["filtered_win_rate"].mean()),
                "base_profit_factor": base_gross_profit / base_gross_loss if base_gross_loss > 0 else (999.0 if base_gross_profit > 0 else 0.0),
                "filtered_profit_factor": filtered_gross_profit / filtered_gross_loss if filtered_gross_loss > 0 else (999.0 if filtered_gross_profit > 0 else 0.0),
                "delta_positive_folds": int((group["delta_pnl"] > 0).sum()),
                "delta_positive_fold_rate": float((group["delta_pnl"] > 0).mean() * 100),
                "filtered_profitable_folds": int((group["filtered_total_pnl"] > 0).sum()),
                "filtered_profitable_fold_rate": float((group["filtered_total_pnl"] > 0).mean() * 100),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["filtered_total_pnl", "filtered_profit_factor", "delta_pnl"],
        ascending=[False, False, False],
    )


def main() -> None:
    targets = parse_targets()
    filter_names = parse_filter_names()
    all_rows = []
    skipped_targets = []

    print("===================================")
    print("Fixed Base Filter Report")
    print("===================================")
    print(f"targets: {', '.join(f'{target.symbol}:{target.timeframe}' for target in targets)}")
    print(f"filters: {len(filter_names)}")
    for filter_name in filter_names:
        print(f"- {filter_name}")
    print(f"train/test/step rows: {TRAIN_ROWS}/{TEST_ROWS}/{STEP_ROWS}")
    print(f"tp_rates: {', '.join(str(value) for value in TP_RATES)}")
    print(f"sl_rates: {', '.join(str(value) for value in SL_RATES)}")
    print(f"lookahead_steps: {', '.join(str(value) for value in LOOKAHEAD_STEPS_LIST)}")
    print(f"exit parameter sets: {len(build_fixed_exit_param_grid())}")
    print("===================================")

    for target in targets:
        try:
            all_rows.extend(run_target(target, filter_names))
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
        raise RuntimeError("No fixed base filter rows were produced.")

    report_df = pd.DataFrame(all_rows)
    summary_df = build_summary(report_df)
    folds_path = REPORT_DIR / "fixed_base_filter_folds.csv"
    summary_path = REPORT_DIR / "fixed_base_filter_summary.csv"
    skipped_path = REPORT_DIR / "fixed_base_filter_skipped.csv"
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
