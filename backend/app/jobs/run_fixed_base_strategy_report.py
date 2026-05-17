import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.jobs.run_base_strategy_walk_forward_report import (
    ensure_sma_columns,
    evaluate_strategy,
)
from app.jobs.run_combined_condition_report import (
    FEE_RATE,
    ExitParams,
    load_dataset,
    parse_targets,
)


TRAIN_ROWS = int(os.getenv("FIXED_BASE_TRAIN_ROWS", os.getenv("WALK_FORWARD_TRAIN_ROWS", "5000")))
TEST_ROWS = int(os.getenv("FIXED_BASE_TEST_ROWS", os.getenv("WALK_FORWARD_TEST_ROWS", "1000")))
STEP_ROWS = int(os.getenv("FIXED_BASE_STEP_ROWS", os.getenv("WALK_FORWARD_STEP_ROWS", "1000")))
REPORT_DIR = Path(os.getenv("FIXED_BASE_REPORT_DIR", "data/reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FixedStrategy:
    name: str
    short_sma: int
    long_sma: int
    rsi_buy_threshold: float
    use_macd_filter: bool
    tp_rate: float
    sl_rate: float
    lookahead_steps: int


DEFAULT_STRATEGIES = (
    "sma5_30_rsi60_macd_tp020_sl015_lh96:5:30:60:1:0.020:0.015:96,"
    "sma5_30_rsi60_nomacd_tp020_sl015_lh96:5:30:60:0:0.020:0.015:96,"
    "sma8_30_rsi65_macd_tp030_sl015_lh72:8:30:65:1:0.030:0.015:72,"
    "sma12_50_rsi65_macd_tp030_sl012_lh72:12:50:65:1:0.030:0.012:72,"
    "sma10_80_rsi50_macd_tp030_sl006_lh96:10:80:50:1:0.030:0.006:96"
)


def parse_bool(value: str) -> bool:
    return value.strip() not in {"0", "false", "False", "off", "OFF"}


def parse_strategies() -> list[FixedStrategy]:
    raw = os.getenv("FIXED_BASE_STRATEGIES", DEFAULT_STRATEGIES)
    strategies: list[FixedStrategy] = []

    for raw_item in raw.replace(";", ",").split(","):
        item = raw_item.strip()

        if not item:
            continue

        parts = [part.strip() for part in item.split(":")]

        if len(parts) != 8:
            raise ValueError(
                "Invalid FIXED_BASE_STRATEGIES item. "
                "Use name:short_sma:long_sma:rsi_buy:macd:tp:sl:lookahead"
            )

        name, short_sma, long_sma, rsi_buy, macd, tp_rate, sl_rate, lookahead = parts
        strategies.append(
            FixedStrategy(
                name=name,
                short_sma=int(short_sma),
                long_sma=int(long_sma),
                rsi_buy_threshold=float(rsi_buy),
                use_macd_filter=parse_bool(macd),
                tp_rate=float(tp_rate),
                sl_rate=float(sl_rate),
                lookahead_steps=int(lookahead),
            )
        )

    if not strategies:
        raise ValueError("No fixed base strategies were provided.")

    return strategies


def run_target(target, strategies: list[FixedStrategy]) -> list[dict]:
    df = ensure_sma_columns(load_dataset(target))

    for period in sorted({strategy.short_sma for strategy in strategies} | {strategy.long_sma for strategy in strategies}):
        column = f"sma_{period}"

        if column not in df.columns:
            df[column] = df["close"].rolling(period).mean()

    rows = []
    fold = 1

    for start in range(0, len(df) - TRAIN_ROWS - TEST_ROWS + 1, STEP_ROWS):
        train_df = df.iloc[start : start + TRAIN_ROWS].reset_index(drop=True)
        test_df = df.iloc[start + TRAIN_ROWS : start + TRAIN_ROWS + TEST_ROWS].reset_index(drop=True)

        for strategy in strategies:
            result = evaluate_strategy(
                test_df,
                short_sma=strategy.short_sma,
                long_sma=strategy.long_sma,
                rsi_buy_threshold=strategy.rsi_buy_threshold,
                use_macd_filter=strategy.use_macd_filter,
                exit_params=ExitParams(
                    tp_rate=strategy.tp_rate,
                    sl_rate=strategy.sl_rate,
                    lookahead_steps=strategy.lookahead_steps,
                ),
            )
            row = {
                "symbol": target.symbol,
                "timeframe": target.timeframe,
                "strategy_name": strategy.name,
                "fold": fold,
                "train_start": train_df.iloc[0]["open_time"],
                "train_end": train_df.iloc[-1]["open_time"],
                "test_start": test_df.iloc[0]["open_time"],
                "test_end": test_df.iloc[-1]["open_time"],
                **{f"test_{key}": value for key, value in result.items()},
            }
            rows.append(row)

        print(f"{target.symbol} {target.timeframe} fold={fold}: evaluated {len(strategies)} fixed strategies")
        fold += 1

    return rows


def build_fixed_summary(report_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_columns = ["symbol", "timeframe", "strategy_name"]

    for (symbol, timeframe, strategy_name), group in report_df.groupby(group_columns, sort=False):
        test_trade_count = int(group["test_trade_count"].sum())
        test_total_pnl = float(group["test_total_pnl"].sum())
        test_gross_profit = float(group["test_gross_profit"].sum())
        test_gross_loss = float(group["test_gross_loss"].sum())
        first = group.iloc[0]
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy_name": strategy_name,
                "folds": int(len(group)),
                "train_rows": TRAIN_ROWS,
                "test_rows": TEST_ROWS,
                "step_rows": STEP_ROWS,
                "short_sma": int(first["test_short_sma"]),
                "long_sma": int(first["test_long_sma"]),
                "rsi_buy_threshold": float(first["test_rsi_buy_threshold"]),
                "use_macd_filter": bool(first["test_use_macd_filter"]),
                "tp_rate": float(first["test_tp_rate"]),
                "sl_rate": float(first["test_sl_rate"]),
                "lookahead_steps": int(first["test_lookahead_steps"]),
                "fee_rate": FEE_RATE,
                "test_trade_count": test_trade_count,
                "test_total_pnl": test_total_pnl,
                "test_avg_pnl": test_total_pnl / test_trade_count if test_trade_count else 0.0,
                "avg_test_win_rate": float(group["test_win_rate"].mean()),
                "test_profit_factor": test_gross_profit / test_gross_loss if test_gross_loss > 0 else (999.0 if test_gross_profit > 0 else 0.0),
                "profitable_folds": int((group["test_total_pnl"] > 0).sum()),
                "profitable_fold_rate": float((group["test_total_pnl"] > 0).mean() * 100),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["test_total_pnl", "test_profit_factor", "test_trade_count"],
        ascending=[False, False, False],
    )


def main() -> None:
    targets = parse_targets()
    strategies = parse_strategies()
    all_rows = []
    skipped_targets = []

    print("===================================")
    print("Fixed Base Strategy Report")
    print("===================================")
    print(f"targets: {', '.join(f'{target.symbol}:{target.timeframe}' for target in targets)}")
    print(f"strategies: {len(strategies)}")
    print(f"train/test/step rows: {TRAIN_ROWS}/{TEST_ROWS}/{STEP_ROWS}")
    print("===================================")

    for target in targets:
        try:
            all_rows.extend(run_target(target, strategies))
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
        raise RuntimeError("No fixed base strategy rows were produced.")

    report_df = pd.DataFrame(all_rows)
    summary_df = build_fixed_summary(report_df)
    folds_path = REPORT_DIR / "fixed_base_strategy_folds.csv"
    summary_path = REPORT_DIR / "fixed_base_strategy_summary.csv"
    skipped_path = REPORT_DIR / "fixed_base_strategy_skipped.csv"
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
