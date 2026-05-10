import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("TRADE_TIMEFRAME", "5m")

DATASET_PATH = Path(
    os.getenv("ML_DATASET_PATH", f"data/ml/{SYMBOL}_{TIMEFRAME}_dataset.csv")
)
MODEL_PATH = Path(
    os.getenv("ML_MODEL_PATH", f"data/models/{SYMBOL}_{TIMEFRAME}_LogisticRegression.joblib")
)

TRAIN_RATIO = 0.8
PROBA_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]


def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def load_dataset(feature_columns: list[str]) -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=feature_columns + ["label"]).reset_index(drop=True)

    return df


def get_positive_class_index(model, positive_label: int = 1) -> int:
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif hasattr(model, "named_steps") and "model" in model.named_steps:
        classes = list(model.named_steps["model"].classes_)
    else:
        raise RuntimeError("Could not read model classes.")

    if positive_label not in classes:
        raise RuntimeError(f"Positive label {positive_label} not found in classes: {classes}")

    return classes.index(positive_label)


def evaluate_threshold(y_true, proba_1, threshold: float) -> dict:
    y_pred = (proba_1 >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    signal_count = tp + fp
    total_count = len(y_true)
    signal_rate = signal_count / total_count if total_count > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "signal_count": signal_count,
        "signal_rate": signal_rate,
    }


def main() -> None:
    bundle = load_model_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    df = load_dataset(feature_columns)

    split_index = int(len(df) * TRAIN_RATIO)
    test_df = df.iloc[split_index:].copy()

    x_test = test_df[feature_columns]
    y_true = test_df["label"].astype(int)

    positive_index = get_positive_class_index(model, positive_label=1)
    proba_1 = model.predict_proba(x_test)[:, positive_index]

    print("===================================")
    print("ML Probability Filter Evaluation")
    print("===================================")
    print(f"dataset: {DATASET_PATH}")
    print(f"model: {MODEL_PATH}")
    print(f"test rows: {len(test_df)}")
    print("test label counts:")
    print(y_true.value_counts().sort_index())
    print("===================================")

    rows = []

    for threshold in PROBA_THRESHOLDS:
        result = evaluate_threshold(y_true, proba_1, threshold)
        rows.append(result)

        print(
            f"threshold={threshold:.2f} | "
            f"precision={result['precision']:.4f} | "
            f"recall={result['recall']:.4f} | "
            f"f1={result['f1']:.4f} | "
            f"signals={result['signal_count']} "
            f"({result['signal_rate'] * 100:.2f}%) | "
            f"TP={result['tp']} FP={result['fp']} FN={result['fn']} TN={result['tn']}"
        )

    output_path = Path("data/ml") / f"{SYMBOL}_{TIMEFRAME}_proba_filter_eval.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print("===================================")
    print(f"saved: {output_path}")
    print("===================================")


if __name__ == "__main__":
    main()