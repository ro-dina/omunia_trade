import os
import subprocess

FUTURE_STEPS_LIST = [3, 6, 12]
THRESHOLD_LIST = [0.002, 0.003, 0.005]

SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("TRADE_TIMEFRAME", "5m")


def run_command(command: list[str], env: dict[str, str]) -> bool:
    result = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=False,
    )

    return result.returncode == 0


def main() -> None:
    base_env = os.environ.copy()

    print("===================================")
    print("ML Experiment Runner")
    print(f"symbol: {SYMBOL}")
    print(f"timeframe: {TIMEFRAME}")
    print("===================================")

    for future_steps in FUTURE_STEPS_LIST:
        for threshold in THRESHOLD_LIST:
            print("\n===================================")
            print(f"future_steps={future_steps}, threshold={threshold}")
            print("===================================")

            env = base_env.copy()
            env["ML_FUTURE_STEPS"] = str(future_steps)
            env["ML_THRESHOLD"] = str(threshold)
            env["TRADE_SYMBOL"] = SYMBOL
            env["TRADE_TIMEFRAME"] = TIMEFRAME

            run_command(["python", "-m", "app.jobs.build_ml_dataset"], env)
            run_command(["python", "-m", "app.jobs.train_ml_classifier"], env)

    print("\n===================================")
    print("All experiments completed")
    print("===================================")


if __name__ == "__main__":
    main()