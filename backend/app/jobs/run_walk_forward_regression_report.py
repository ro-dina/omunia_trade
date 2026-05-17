import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.services.ml_features import FEATURE_COLUMNS


@dataclass(frozen=True)
class ReportTarget:
    symbol: str
    timeframe: str


MODEL_NAME = os.getenv("REGRESSION_MODEL_NAME", "RandomForestRegressor")
TARGET_COLUMN = "future_return"

TRAIN_ROWS = int(os.getenv("WALK_FORWARD_TRAIN_ROWS", "5000"))
TEST_ROWS = int(os.getenv("WALK_FORWARD_TEST_ROWS", "1000"))
STEP_ROWS = int(os.getenv("WALK_FORWARD_STEP_ROWS", str(TEST_ROWS)))
MAX_FOLDS = int(os.getenv("WALK_FORWARD_MAX_FOLDS", "0"))
STRICT_DATASETS = os.getenv("WALK_FORWARD_STRICT", "false").lower() == "true"
TOP_QUANTILES = [
    float(value.strip())
    for value in os.getenv("REGRESSION_TOP_QUANTILES", "0.80,0.90,0.95").split(",")
    if value.strip()
]
HOLD_STEPS = int(os.getenv("REGRESSION_HOLD_STEPS", os.getenv("ML_FUTURE_STEPS", "3")))
INITIAL_CASH = 10_000.0
TRADE_NOTIONAL = float(os.getenv("REGRESSION_TRADE_NOTIONAL", "1000"))
FEE_RATE = float(os.getenv("REGRESSION_FEE_RATE", "0.0006"))

DATA_DIR = Path(os.getenv("ML_DATA_DIR", "data/ml"))
REPORT_DIR = Path(os.getenv("REGRESSION_REPORT_DIR", "data/reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)


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
            raise ValueError(
                f"Invalid target '{item}'. Use SYMBOL:TIMEFRAME, e.g. BTCUSDT:5m."
            )

        targets.append(
            ReportTarget(
                symbol=symbol.strip().upper(),
                timeframe=timeframe.strip().lower(),
            )
        )

    if not targets:
        raise ValueError("No report targets were provided.")

    return targets


def create_model():
    if MODEL_NAME == "Ridge":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )

    if MODEL_NAME == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported REGRESSION_MODEL_NAME: {MODEL_NAME}")


def load_dataset(target: ReportTarget) -> pd.DataFrame:
    path = DATA_DIR / f"{target.symbol}_{target.timeframe}_dataset.csv"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing_columns = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN, "open_time", "close"]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    return df


def iter_fold_ranges(row_count: int):
    fold_number = 1
    train_start = 0

    while True:
        train_end = train_start + TRAIN_ROWS
        test_end = train_end + TEST_ROWS

        if test_end > row_count:
            break

        yield fold_number, train_start, train_end, test_end

        if MAX_FOLDS > 0 and fold_number >= MAX_FOLDS:
            break

        fold_number += 1
        train_start += STEP_ROWS


def correlation(actual: pd.Series, predicted: np.ndarray) -> float:
    if len(actual) < 2 or actual.nunique() < 2 or np.std(predicted) == 0:
        return 0.0

    return float(np.corrcoef(actual, predicted)[0, 1])


def evaluate_regression(actual: pd.Series, predicted: np.ndarray) -> dict:
    mse = mean_squared_error(actual, predicted)

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mse)),
        "corr": correlation(actual, predicted),
    }


def simulate_top_prediction_trades(test_df: pd.DataFrame, threshold: float) -> dict:
    cash = INITIAL_CASH
    trades = []
    i = 0

    while i < len(test_df) - HOLD_STEPS:
        row = test_df.iloc[i]

        if float(row["predicted_return"]) < threshold:
            i += 1
            continue

        exit_i = i + HOLD_STEPS
        exit_row = test_df.iloc[exit_i]
        entry_price = float(row["close"])
        exit_price = float(exit_row["close"])
        notional = min(TRADE_NOTIONAL, cash)

        if notional <= 0:
            break

        entry_fee = notional * FEE_RATE
        qty = (notional - entry_fee) / entry_price
        gross = qty * exit_price
        exit_fee = gross * FEE_RATE
        pnl = gross - exit_fee - notional
        cash += pnl

        trades.append(
            {
                "entry_time": row["open_time"],
                "exit_time": exit_row["open_time"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "predicted_return": float(row["predicted_return"]),
                "actual_return": (exit_price - entry_price) / entry_price,
                "pnl": pnl,
            }
        )

        i = exit_i + 1

    trade_count = len(trades)
    wins = sum(1 for trade in trades if trade["pnl"] > 0)
    total_pnl = sum(trade["pnl"] for trade in trades)

    return {
        "trade_count": trade_count,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / trade_count if trade_count else 0.0,
        "win_rate": (wins / trade_count * 100) if trade_count else 0.0,
        "final_equity": cash,
        "total_return": (cash - INITIAL_CASH) / INITIAL_CASH * 100,
    }


def summarize_top_bucket(test_df: pd.DataFrame, threshold: float) -> dict:
    selected = test_df[test_df["predicted_return"] >= threshold]

    if selected.empty:
        return {
            "top_count": 0,
            "top_avg_actual_return": 0.0,
            "top_win_rate": 0.0,
            "top_avg_predicted_return": 0.0,
        }

    return {
        "top_count": len(selected),
        "top_avg_actual_return": float(selected[TARGET_COLUMN].mean()),
        "top_win_rate": float((selected[TARGET_COLUMN] > 0).mean() * 100),
        "top_avg_predicted_return": float(selected["predicted_return"].mean()),
    }


def run_target(target: ReportTarget) -> list[dict]:
    df = load_dataset(target)
    rows: list[dict] = []

    for fold_number, train_start, train_end, test_end in iter_fold_ranges(len(df)):
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        model = create_model()
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

        train_pred = model.predict(train_df[FEATURE_COLUMNS])
        test_pred = model.predict(test_df[FEATURE_COLUMNS])
        test_df["predicted_return"] = test_pred
        regression_metrics = evaluate_regression(test_df[TARGET_COLUMN], test_pred)

        for quantile in TOP_QUANTILES:
            threshold = float(np.quantile(train_pred, quantile))
            top_metrics = summarize_top_bucket(test_df, threshold)
            trade_metrics = simulate_top_prediction_trades(test_df, threshold)

            row = {
                "symbol": target.symbol,
                "timeframe": target.timeframe,
                "model_name": MODEL_NAME,
                "fold": fold_number,
                "top_quantile": quantile,
                "prediction_threshold": threshold,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_start": train_df["open_time"].iloc[0],
                "train_end": train_df["open_time"].iloc[-1],
                "test_start": test_df["open_time"].iloc[0],
                "test_end": test_df["open_time"].iloc[-1],
                **regression_metrics,
                **top_metrics,
                **trade_metrics,
            }
            rows.append(row)

            print(
                f"{target.symbol} {target.timeframe} fold={fold_number} "
                f"q={quantile:.2f} mae={row['mae']:.5f} rmse={row['rmse']:.5f} "
                f"corr={row['corr']:.3f} trades={row['trade_count']} "
                f"pnl={row['total_pnl']:.2f}"
            )

    return rows


def build_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    grouped = fold_df.groupby(
        ["symbol", "timeframe", "model_name", "top_quantile"],
        as_index=False,
    )
    summary = grouped.agg(
        folds=("fold", "count"),
        total_test_rows=("test_rows", "sum"),
        avg_mae=("mae", "mean"),
        avg_rmse=("rmse", "mean"),
        avg_corr=("corr", "mean"),
        top_count=("top_count", "sum"),
        top_avg_actual_return=("top_avg_actual_return", "mean"),
        top_win_rate=("top_win_rate", "mean"),
        trade_count=("trade_count", "sum"),
        total_pnl=("total_pnl", "sum"),
        avg_fold_return=("total_return", "mean"),
        avg_trade_pnl=("avg_pnl", "mean"),
        avg_trade_win_rate=("win_rate", "mean"),
    )
    summary["top_selection_rate"] = summary["top_count"] / summary["total_test_rows"]

    return summary.sort_values(["total_pnl", "avg_corr"], ascending=[False, False])


def main() -> None:
    targets = parse_targets()
    all_rows: list[dict] = []
    skipped_targets: list[dict] = []

    print("===================================")
    print("Walk-forward Regression Report")
    print("===================================")
    print(f"targets: {', '.join(f'{t.symbol}:{t.timeframe}' for t in targets)}")
    print(f"model: {MODEL_NAME}")
    print(f"target: {TARGET_COLUMN}")
    print(f"top_quantiles: {', '.join(f'{value:.2f}' for value in TOP_QUANTILES)}")
    print(f"train_rows: {TRAIN_ROWS}")
    print(f"test_rows: {TEST_ROWS}")
    print(f"step_rows: {STEP_ROWS}")
    print(f"hold_steps: {HOLD_STEPS}")
    print("===================================")

    for target in targets:
        try:
            all_rows.extend(run_target(target))
        except Exception as e:
            if STRICT_DATASETS:
                raise

            skipped_targets.append(
                {
                    "symbol": target.symbol,
                    "timeframe": target.timeframe,
                    "reason": str(e),
                }
            )
            print(f"Skipped {target.symbol} {target.timeframe}: {e}")

    if not all_rows:
        raise RuntimeError("No walk-forward regression folds were produced.")

    fold_df = pd.DataFrame(all_rows)
    summary_df = build_summary(fold_df)

    fold_path = REPORT_DIR / "walk_forward_regression_folds.csv"
    summary_path = REPORT_DIR / "walk_forward_regression_summary.csv"
    skipped_path = REPORT_DIR / "walk_forward_regression_skipped.csv"
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    if skipped_targets:
        pd.DataFrame(skipped_targets).to_csv(skipped_path, index=False)

    print("\n===================================")
    print("Summary")
    print("===================================")
    print(summary_df.to_string(index=False))
    print("===================================")
    print(f"fold report: {fold_path}")
    print(f"summary report: {summary_path}")
    if skipped_targets:
        print(f"skipped targets: {skipped_path}")
    print("===================================")


if __name__ == "__main__":
    main()
