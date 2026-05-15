import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.jobs.backtest_ml_filter import add_ml_proba, add_sma_cross_columns, run_backtest
from app.services.ml_features import FEATURE_COLUMNS


@dataclass(frozen=True)
class ReportTarget:
    symbol: str
    timeframe: str


MODEL_NAME = os.getenv("ML_MODEL_NAME", "RandomForest")
TARGET_COLUMN = "label"
ML_PROBA_THRESHOLDS = [
    float(value.strip())
    for value in os.getenv("ML_PROBA_THRESHOLDS", "0.50,0.55,0.60,0.65").split(",")
    if value.strip()
]

TRAIN_ROWS = int(os.getenv("WALK_FORWARD_TRAIN_ROWS", "5000"))
TEST_ROWS = int(os.getenv("WALK_FORWARD_TEST_ROWS", "1000"))
STEP_ROWS = int(os.getenv("WALK_FORWARD_STEP_ROWS", str(TEST_ROWS)))
MAX_FOLDS = int(os.getenv("WALK_FORWARD_MAX_FOLDS", "0"))
STRICT_DATASETS = os.getenv("WALK_FORWARD_STRICT", "false").lower() == "true"

DATA_DIR = Path(os.getenv("ML_DATA_DIR", "data/ml"))
REPORT_DIR = Path(os.getenv("WALK_FORWARD_REPORT_DIR", "data/reports"))
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
    if MODEL_NAME == "LogisticRegression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    if MODEL_NAME == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported ML_MODEL_NAME: {MODEL_NAME}")


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
    df = add_sma_cross_columns(df)
    df = df.dropna().reset_index(drop=True)

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


def evaluate_classifier(model, test_df: pd.DataFrame) -> dict:
    y_test = test_df[TARGET_COLUMN]
    pred = model.predict(test_df[FEATURE_COLUMNS])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        pred,
        labels=[1],
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_1": float(precision[0]),
        "recall_1": float(recall[0]),
        "f1_1": float(f1[0]),
    }


def summarize_result(prefix: str, result: dict) -> dict:
    return {
        f"{prefix}_final_equity": result["final_equity"],
        f"{prefix}_total_return": result["total_return"],
        f"{prefix}_realized_pnl": result["realized_pnl"],
        f"{prefix}_buy_count": result["buy_count"],
        f"{prefix}_sell_count": result["sell_count"],
        f"{prefix}_win_rate": result["win_rate"],
        f"{prefix}_take_profit_count": result["take_profit_count"],
        f"{prefix}_stop_loss_count": result["stop_loss_count"],
        f"{prefix}_sma_exit_count": result["sma_exit_count"],
        f"{prefix}_blocked_buy_count": result["ml_blocked_buy_count"],
    }


def run_target(target: ReportTarget) -> list[dict]:
    df = load_dataset(target)
    rows: list[dict] = []

    for fold_number, train_start, train_end, test_end in iter_fold_ranges(len(df)):
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        if train_df[TARGET_COLUMN].nunique() < 2 or test_df[TARGET_COLUMN].nunique() < 2:
            print(
                f"Skipped {target.symbol} {target.timeframe} fold={fold_number}: "
                "train/test contains only one class."
            )
            continue

        model = create_model()
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
        test_df = add_ml_proba(test_df, model, FEATURE_COLUMNS)

        base_result = run_backtest(test_df, use_ml_filter=False)
        classifier_metrics = evaluate_classifier(model, test_df)

        for threshold in ML_PROBA_THRESHOLDS:
            ml_result = run_backtest(
                test_df,
                use_ml_filter=True,
                ml_proba_threshold=threshold,
            )

            row = {
                "symbol": target.symbol,
                "timeframe": target.timeframe,
                "model_name": MODEL_NAME,
                "ml_proba_threshold": threshold,
                "fold": fold_number,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_start": train_df["open_time"].iloc[0],
                "train_end": train_df["open_time"].iloc[-1],
                "test_start": test_df["open_time"].iloc[0],
                "test_end": test_df["open_time"].iloc[-1],
                **classifier_metrics,
                **summarize_result("base", base_result),
                **summarize_result("ml", ml_result),
            }
            row["delta_pnl"] = row["ml_realized_pnl"] - row["base_realized_pnl"]
            row["delta_return"] = row["ml_total_return"] - row["base_total_return"]
            rows.append(row)

            print(
                f"{target.symbol} {target.timeframe} fold={fold_number} "
                f"threshold={threshold:.2f} "
                f"base_pnl={row['base_realized_pnl']:.2f} "
                f"ml_pnl={row['ml_realized_pnl']:.2f} "
                f"delta={row['delta_pnl']:.2f} "
                f"ml_buys={row['ml_buy_count']}"
            )

    return rows


def build_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    grouped = fold_df.groupby(
        ["symbol", "timeframe", "model_name", "ml_proba_threshold"],
        as_index=False,
    )
    summary = grouped.agg(
        folds=("fold", "count"),
        total_test_rows=("test_rows", "sum"),
        base_pnl=("base_realized_pnl", "sum"),
        ml_pnl=("ml_realized_pnl", "sum"),
        delta_pnl=("delta_pnl", "sum"),
        base_avg_return=("base_total_return", "mean"),
        ml_avg_return=("ml_total_return", "mean"),
        delta_avg_return=("delta_return", "mean"),
        base_buys=("base_buy_count", "sum"),
        ml_buys=("ml_buy_count", "sum"),
        ml_blocked_buys=("ml_blocked_buy_count", "sum"),
        avg_precision_1=("precision_1", "mean"),
        avg_recall_1=("recall_1", "mean"),
        avg_f1_1=("f1_1", "mean"),
    )
    summary["ml_pnl_better"] = summary["delta_pnl"] > 0

    return summary.sort_values(["delta_pnl", "ml_pnl"], ascending=[False, False])


def main() -> None:
    targets = parse_targets()
    all_rows: list[dict] = []
    skipped_targets: list[dict] = []

    print("===================================")
    print("Walk-forward Report")
    print("===================================")
    print(f"targets: {', '.join(f'{t.symbol}:{t.timeframe}' for t in targets)}")
    print(f"model: {MODEL_NAME}")
    print(f"thresholds: {', '.join(f'{value:.2f}' for value in ML_PROBA_THRESHOLDS)}")
    print(f"train_rows: {TRAIN_ROWS}")
    print(f"test_rows: {TEST_ROWS}")
    print(f"step_rows: {STEP_ROWS}")
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
        raise RuntimeError("No walk-forward folds were produced.")

    fold_df = pd.DataFrame(all_rows)
    summary_df = build_summary(fold_df)

    fold_path = REPORT_DIR / "walk_forward_folds.csv"
    summary_path = REPORT_DIR / "walk_forward_summary.csv"
    skipped_path = REPORT_DIR / "walk_forward_skipped.csv"
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
