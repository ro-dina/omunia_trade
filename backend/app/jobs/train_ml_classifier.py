import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.services.ml_features import FEATURE_COLUMNS

SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("TRADE_TIMEFRAME", "5m")
DATASET_PATH = Path(os.getenv("ML_DATASET_PATH", f"data/ml/{SYMBOL}_{TIMEFRAME}_dataset.csv"))
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "label"
TRAIN_RATIO = 0.8


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    missing_columns = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    if len(df) < 200:
        raise RuntimeError(f"Dataset too small: rows={len(df)}")

    return df


def time_series_split(df: pd.DataFrame):
    split_index = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return x_train, x_test, y_train, y_test, train_df, test_df


def print_dataset_summary(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("===================================")
    print("Dataset Summary")
    print("===================================")
    print(f"dataset: {DATASET_PATH}")
    print(f"rows: {len(df)}")
    print(f"train rows: {len(train_df)}")
    print(f"test rows: {len(test_df)}")

    if "open_time" in df.columns:
        print(f"train range: {train_df['open_time'].iloc[0]} -> {train_df['open_time'].iloc[-1]}")
        print(f"test range:  {test_df['open_time'].iloc[0]} -> {test_df['open_time'].iloc[-1]}")

    print("label counts:")
    print(df[TARGET_COLUMN].value_counts().sort_index())
    print("===================================")


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        pred,
        labels=[1],
        zero_division=0,
    )

    precision_1 = float(precision[0])
    recall_1 = float(recall[0])
    f1_1 = float(f1[0])

    print("\n===================================")
    print(name)
    print("===================================")
    print(f"accuracy: {accuracy:.4f}")
    print(f"label=1 precision: {precision_1:.4f}")
    print(f"label=1 recall:    {recall_1:.4f}")
    print(f"label=1 f1:        {f1_1:.4f}")
    print("confusion matrix:")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print("classification report:")
    print(classification_report(y_test, pred, labels=[0, 1], zero_division=0))

    metrics = {
        "accuracy": accuracy,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
    }

    return metrics, model


def main() -> None:
    df = load_dataset()
    x_train, x_test, y_train, y_test, train_df, test_df = time_series_split(df)

    print_dataset_summary(df, train_df, test_df)


    if y_train.nunique() < 2:
        print("Skipped: train data contains only one class.")
        return

    if y_test.nunique() < 2:
        print("Skipped: test data contains only one class.")
        return

    models = [
        (
            "LogisticRegression",
            Pipeline(
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
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=20,
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        ),
    ]

    best_name = ""
    best_score = -1.0
    best_metrics = None
    best_model = None

    for name, model in models:
        metrics, fitted_model = evaluate_model(
            name=name,
            model=model,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )

        selection_score = metrics["f1_1"]

        if selection_score > best_score:
            best_name = name
            best_score = selection_score
            best_metrics = metrics
            best_model = fitted_model

    output_path = MODEL_DIR / f"{SYMBOL}_{TIMEFRAME}_{best_name}.joblib"
    joblib.dump(
        {
            "model_name": best_name,
            "model": best_model,
            "feature_columns": FEATURE_COLUMNS,
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "metrics": best_metrics,
            "selection_metric": "f1_1",
            "selection_score": best_score,
        },
        output_path,
    )

    print("\n===================================")
    print("Best Model Saved")
    print("===================================")
    print(f"model: {best_name}")
    print("selection_metric: f1_1")
    print(f"selection_score: {best_score:.4f}")
    print(f"metrics: {best_metrics}")
    print(f"path: {output_path}")
    print("===================================")


if __name__ == "__main__":
    main()
