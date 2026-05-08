from app.services.backtest_service import (
    optimize_sma_parameters,
    print_optimization_results,
    run_sma_cross_backtest,
)


def main() -> None:
    short_periods = [5, 10, 15, 20, 25, 30]
    long_periods = [20, 30, 40, 50, 80, 100, 150]

    results = optimize_sma_parameters(
        short_periods=short_periods,
        long_periods=long_periods,
        limit=1000,
    )

    print_optimization_results(results, top_n=10)

    best = results[0]

    print("\nBest strategy detail:\n")

    run_sma_cross_backtest(
        short_period=best["short_period"],
        long_period=best["long_period"],
        limit=1000,
    )


if __name__ == "__main__":
    main()