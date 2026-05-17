import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.jobs.backtest_ml_filter import add_ml_proba
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

TRAIN_ROWS = int(os.getenv("WALK_FORWARD_TRAIN_ROWS", "3000"))
TEST_ROWS = int(os.getenv("WALK_FORWARD_TEST_ROWS", "500"))
STEP_ROWS = int(os.getenv("WALK_FORWARD_STEP_ROWS", str(TEST_ROWS)))
MAX_FOLDS = int(os.getenv("WALK_FORWARD_MAX_FOLDS", "0"))
STRICT_DATASETS = os.getenv("WALK_FORWARD_STRICT", "false").lower() == "true"

TAKE_PROFIT_RATE = float(os.getenv("ML_ENTRY_TP_RATE", os.getenv("ML_TP_RATE", "0.010")))
STOP_LOSS_RATE = float(os.getenv("ML_ENTRY_SL_RATE", os.getenv("ML_SL_RATE", "0.008")))
LOOKAHEAD_STEPS = int(os.getenv("ML_ENTRY_LOOKAHEAD_STEPS", os.getenv("ML_TP_SL_LOOKAHEAD_STEPS", "12")))
TIE_BREAKER = os.getenv("ML_ENTRY_TIE_BREAKER", os.getenv("ML_TP_SL_TIE_BREAKER", "stop_loss"))
TRADE_NOTIONAL = float(os.getenv("ML_ENTRY_TRADE_NOTIONAL", "1000"))
INITIAL_CASH = float(os.getenv("ML_ENTRY_INITIAL_CASH", "10000"))
FEE_RATE = float(os.getenv("ML_ENTRY_FEE_RATE", "0.0006"))

MIN_TRADES = int(os.getenv("ML_ENTRY_MIN_TRADES", "10"))
MIN_TRADE_RATIO = float(os.getenv("ML_ENTRY_MIN_TRADE_RATIO", "0.01"))
SELECT_OBJECTIVE = os.getenv("ML_ENTRY_SELECT_OBJECTIVE", "total_pnl")

DATA_DIR = Path(os.getenv("ML_DATA_DIR", "data/ml"))
REPORT_DIR = Path(os.getenv("ML_ENTRY_REPORT_DIR", "data/reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if TIE_BREAKER not in {"stop_loss", "take_profit"}:
    raise ValueError("ML_ENTRY_TIE_BREAKER must be stop_loss or take_profit.")

if SELECT_OBJECTIVE not in {"total_pnl", "profit_factor", "win_rate"}:
    raise ValueError("ML_ENTRY_SELECT_OBJECTIVE must be total_pnl, profit_factor, or win_rate.")


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
        for column in FEATURE_COLUMNS + [TARGET_COLUMN, "open_time", "open", "high", "low", "close"]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")

    return df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)


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


def resolve_exit(test_df: pd.DataFrame, entry_index: int) -> tuple[int, float, str]:
    entry_price = float(test_df.iloc[entry_index]["close"])
    tp_price = entry_price * (1 + TAKE_PROFIT_RATE)
    sl_price = entry_price * (1 - STOP_LOSS_RATE)
    last_index = min(entry_index + LOOKAHEAD_STEPS, len(test_df) - 1)

    for i in range(entry_index + 1, last_index + 1):
        row = test_df.iloc[i]
        hit_tp = float(row["high"]) >= tp_price
        hit_sl = float(row["low"]) <= sl_price

        if hit_tp and hit_sl:
            if TIE_BREAKER == "take_profit":
                return i, tp_price, "TAKE_PROFIT"

            return i, sl_price, "STOP_LOSS"

        if hit_tp:
            return i, tp_price, "TAKE_PROFIT"

        if hit_sl:
            return i, sl_price, "STOP_LOSS"

    return last_index, float(test_df.iloc[last_index]["close"]), "TIME_EXIT"


def simulate_ml_entry(test_df: pd.DataFrame, threshold: float) -> dict:
    cash = INITIAL_CASH
    trades = []
    i = 0
    signal_count = int((test_df["ml_proba_1"] >= threshold).sum())

    while i < len(test_df) - 1:
        row = test_df.iloc[i]

        if float(row["ml_proba_1"]) < threshold:
            i += 1
            continue

        entry_price = float(row["close"])
        notional = min(TRADE_NOTIONAL, cash)

        if notional <= 0:
            break

        exit_i, exit_price, exit_reason = resolve_exit(test_df, i)
        entry_fee = notional * FEE_RATE
        qty = (notional - entry_fee) / entry_price
        gross = qty * exit_price
        exit_fee = gross * FEE_RATE
        pnl = gross - exit_fee - notional
        cash += pnl

        trades.append(
            {
                "entry_time": row["open_time"],
                "exit_time": test_df.iloc[exit_i]["open_time"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "ml_proba_1": float(row["ml_proba_1"]),
            }
        )
        i = exit_i + 1

    trade_count = len(trades)
    wins = sum(1 for trade in trades if trade["pnl"] > 0)
    losses = trade_count - wins
    total_pnl = sum(trade["pnl"] for trade in trades)
    gross_profit = sum(trade["pnl"] for trade in trades if trade["pnl"] > 0)
    gross_loss = -sum(trade["pnl"] for trade in trades if trade["pnl"] < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    return {
        "signal_count": signal_count,
        "trade_count": trade_count,
        "win_count": wins,
        "loss_count": losses,
        "win_rate": wins / trade_count * 100 if trade_count else 0.0,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / trade_count if trade_count else 0.0,
        "final_equity": cash,
        "total_return": (cash - INITIAL_CASH) / INITIAL_CASH * 100,
        "take_profit_count": sum(1 for trade in trades if trade["exit_reason"] == "TAKE_PROFIT"),
        "stop_loss_count": sum(1 for trade in trades if trade["exit_reason"] == "STOP_LOSS"),
        "time_exit_count": sum(1 for trade in trades if trade["exit_reason"] == "TIME_EXIT"),
        "profit_factor": profit_factor,
    }


def run_target(target: ReportTarget) -> list[dict]:
    df = load_dataset(target)
    rows: list[dict] = []

    for fold_number, train_start, train_end, test_end in iter_fold_ranges(len(df)):
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        if train_df[TARGET_COLUMN].nunique() < 2 or test_df[TARGET_COLUMN].nunique() < 2:
            print(f"Skipped {target.symbol} {target.timeframe} fold={fold_number}: one class only.")
            continue

        model = create_model()
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
        test_df = add_ml_proba(test_df, model, FEATURE_COLUMNS)
        classifier_metrics = evaluate_classifier(model, test_df)

        for threshold in ML_PROBA_THRESHOLDS:
            result = simulate_ml_entry(test_df, threshold)
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
                **result,
            }
            rows.append(row)

            print(
                f"{target.symbol} {target.timeframe} fold={fold_number} "
                f"threshold={threshold:.2f} trades={row['trade_count']} "
                f"pnl={row['total_pnl']:.2f} win_rate={row['win_rate']:.1f}%"
            )

    return rows


def build_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    grouped = fold_df.groupby(["symbol", "timeframe", "model_name", "ml_proba_threshold"], as_index=False)
    summary = grouped.agg(
        folds=("fold", "count"),
        total_test_rows=("test_rows", "sum"),
        signal_count=("signal_count", "sum"),
        trade_count=("trade_count", "sum"),
        total_pnl=("total_pnl", "sum"),
        avg_fold_return=("total_return", "mean"),
        avg_pnl=("avg_pnl", "mean"),
        avg_win_rate=("win_rate", "mean"),
        take_profit_count=("take_profit_count", "sum"),
        stop_loss_count=("stop_loss_count", "sum"),
        time_exit_count=("time_exit_count", "sum"),
        avg_profit_factor=("profit_factor", "mean"),
        avg_precision_1=("precision_1", "mean"),
        avg_recall_1=("recall_1", "mean"),
        avg_f1_1=("f1_1", "mean"),
    )
    summary["trade_ratio"] = summary["trade_count"] / summary["total_test_rows"]
    summary["selection_eligible"] = (
        (summary["trade_count"] >= MIN_TRADES)
        & (summary["trade_ratio"] >= MIN_TRADE_RATIO)
    )

    return summary.sort_values(["total_pnl", "avg_profit_factor"], ascending=[False, False])


def select_thresholds(summary_df: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []

    for _, group in summary_df.groupby(["symbol", "timeframe", "model_name"], as_index=False):
        candidates = group[group["selection_eligible"]].copy()
        used_fallback = False

        if candidates.empty:
            candidates = group.copy()
            used_fallback = True

        selected = candidates.sort_values(
            [SELECT_OBJECTIVE, "total_pnl", "trade_count"],
            ascending=[False, False, False],
        ).iloc[0].copy()
        selected["selection_objective"] = SELECT_OBJECTIVE
        selected["selection_min_trades"] = MIN_TRADES
        selected["selection_min_trade_ratio"] = MIN_TRADE_RATIO
        selected["selection_used_fallback"] = used_fallback
        selected_rows.append(selected)

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def main() -> None:
    targets = parse_targets()
    all_rows: list[dict] = []
    skipped_targets: list[dict] = []

    print("===================================")
    print("ML Entry Walk-forward Report")
    print("===================================")
    print(f"targets: {', '.join(f'{target.symbol}:{target.timeframe}' for target in targets)}")
    print(f"model: {MODEL_NAME}")
    print(f"thresholds: {', '.join(f'{value:.2f}' for value in ML_PROBA_THRESHOLDS)}")
    print(f"tp/sl/lookahead: {TAKE_PROFIT_RATE}/{STOP_LOSS_RATE}/{LOOKAHEAD_STEPS}")
    print(f"train/test/step rows: {TRAIN_ROWS}/{TEST_ROWS}/{STEP_ROWS}")
    print("===================================")

    for target in targets:
        try:
            all_rows.extend(run_target(target))
        except Exception as e:
            if STRICT_DATASETS:
                raise

            skipped_targets.append(
                {"symbol": target.symbol, "timeframe": target.timeframe, "reason": str(e)}
            )
            print(f"Skipped {target.symbol} {target.timeframe}: {e}")

    if not all_rows:
        raise RuntimeError("No ML-entry walk-forward folds were produced.")

    fold_df = pd.DataFrame(all_rows)
    summary_df = build_summary(fold_df)
    selected_df = select_thresholds(summary_df)

    fold_path = REPORT_DIR / "ml_entry_walk_forward_folds.csv"
    summary_path = REPORT_DIR / "ml_entry_walk_forward_summary.csv"
    selected_path = REPORT_DIR / "ml_entry_selected_thresholds.csv"
    skipped_path = REPORT_DIR / "ml_entry_walk_forward_skipped.csv"
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    if skipped_targets:
        pd.DataFrame(skipped_targets).to_csv(skipped_path, index=False)

    print("\n===================================")
    print("Summary")
    print("===================================")
    print(summary_df.to_string(index=False))
    print("\n===================================")
    print("Selected Thresholds")
    print("===================================")
    print(selected_df.to_string(index=False))
    print("===================================")
    print(f"fold report: {fold_path}")
    print(f"summary report: {summary_path}")
    print(f"selected thresholds: {selected_path}")
    if skipped_targets:
        print(f"skipped targets: {skipped_path}")
    print("===================================")


if __name__ == "__main__":
    main()
