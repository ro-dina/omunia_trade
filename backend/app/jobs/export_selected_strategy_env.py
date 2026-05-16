from pathlib import Path
import pandas as pd

SELECTED_PATH = Path("data/reports/walk_forward_selected_thresholds.csv")
OUTPUT_PATH = Path(".env.paper")

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"

TAKE_PROFIT_RATE = "0.010"
STOP_LOSS_RATE = "0.008"
ML_MODEL_NAME = "RandomForest"


def main() -> None:
    df = pd.read_csv(SELECTED_PATH)

    row = df[
        (df["symbol"] == SYMBOL)
        & (df["timeframe"] == TIMEFRAME)
        & (df["model_name"] == ML_MODEL_NAME)
    ].iloc[0]

    threshold = float(row["ml_proba_threshold"])

    content = f"""TRADE_SYMBOL={SYMBOL}
TRADE_TIMEFRAME={TIMEFRAME}
ML_MODEL_NAME={ML_MODEL_NAME}
ML_PROBA_THRESHOLD={threshold:.2f}
TAKE_PROFIT_RATE={TAKE_PROFIT_RATE}
STOP_LOSS_RATE={STOP_LOSS_RATE}
USE_ML_FILTER=true
"""

    OUTPUT_PATH.write_text(content, encoding="utf-8")

    print("===================================")
    print("Exported selected paper strategy env")
    print("===================================")
    print(content)
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()