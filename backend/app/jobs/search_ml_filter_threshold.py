import os
import subprocess

SEARCH_MODE = os.getenv("SEARCH_MODE", "threshold")

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TAKE_PROFIT_RATES = [0.010, 0.015, 0.020]
STOP_LOSS_RATES = [0.003, 0.005, 0.008]

SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("TRADE_TIMEFRAME", "5m")
ML_MODEL_NAME = os.getenv("ML_MODEL_NAME", "RandomForest")
ML_PROBA_THRESHOLD = os.getenv("ML_PROBA_THRESHOLD", "0.60")


def run_backtest(env: dict[str, str]) -> None:
    subprocess.run(
        ["python", "-m", "app.jobs.backtest_ml_filter"],
        env=env,
        check=False,
    )


def run_threshold_search(base_env: dict[str, str]) -> None:
    print("===================================")
    print("ML Filter Threshold Search")
    print(f"symbol: {SYMBOL}")
    print(f"timeframe: {TIMEFRAME}")
    print(f"model: {ML_MODEL_NAME}")
    print("===================================")

    for threshold in THRESHOLDS:
        print("\n===================================")
        print(f"ML_PROBA_THRESHOLD={threshold}")
        print("===================================")

        env = base_env.copy()
        env["TRADE_SYMBOL"] = SYMBOL
        env["TRADE_TIMEFRAME"] = TIMEFRAME
        env["ML_MODEL_NAME"] = ML_MODEL_NAME
        env["ML_PROBA_THRESHOLD"] = str(threshold)

        run_backtest(env)

    print("\n===================================")
    print("Threshold search completed")
    print("===================================")


def run_tp_sl_search(base_env: dict[str, str]) -> None:
    print("===================================")
    print("TP/SL Search with ML Filter")
    print(f"symbol: {SYMBOL}")
    print(f"timeframe: {TIMEFRAME}")
    print(f"model: {ML_MODEL_NAME}")
    print(f"ml_proba_threshold: {ML_PROBA_THRESHOLD}")
    print("===================================")

    for take_profit_rate in TAKE_PROFIT_RATES:
        for stop_loss_rate in STOP_LOSS_RATES:
            print("\n===================================")
            print(
                f"TP={take_profit_rate * 100:.1f}% "
                f"SL={stop_loss_rate * 100:.1f}% "
                f"ML={ML_PROBA_THRESHOLD}"
            )
            print("===================================")

            env = base_env.copy()
            env["TRADE_SYMBOL"] = SYMBOL
            env["TRADE_TIMEFRAME"] = TIMEFRAME
            env["ML_MODEL_NAME"] = ML_MODEL_NAME
            env["ML_PROBA_THRESHOLD"] = ML_PROBA_THRESHOLD
            env["TAKE_PROFIT_RATE"] = str(take_profit_rate)
            env["STOP_LOSS_RATE"] = str(stop_loss_rate)

            run_backtest(env)

    print("\n===================================")
    print("TP/SL search completed")
    print("===================================")


def main() -> None:
    base_env = os.environ.copy()

    if SEARCH_MODE == "tp_sl":
        run_tp_sl_search(base_env)
        return

    run_threshold_search(base_env)


if __name__ == "__main__":
    main()