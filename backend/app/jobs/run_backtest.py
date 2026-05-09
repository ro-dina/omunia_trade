from app.services.backtest_service import (
    optimize_sma_parameters,
    print_optimization_results,
    run_sma_cross_backtest,
    save_backtest_results,
)


def main() -> None:
    short_periods = [5, 10, 15, 20, 25, 30]
    long_periods = [20, 30, 40, 50, 80, 100, 150]
    timeframe = "5m"
    limit = 5000

    results = optimize_sma_parameters(
        short_periods=short_periods,
        long_periods=long_periods,
        timeframe=timeframe,
        limit=limit,
        rsi_buy_thresholds=[45.0, 50.0, 55.0, 60.0],
        rsi_sell_thresholds=[35.0, 40.0, 45.0, 50.0],
    )

    print_optimization_results(results, top_n=10)
    save_backtest_results(results, top_n=20)

    best = results[0]

    print("\nBest strategy detail:\n")

    run_sma_cross_backtest(
        short_period=best["short_period"],
        long_period=best["long_period"],
        timeframe=best["timeframe"],
        limit=limit,
        take_profit_rate=best["take_profit_rate"],
        stop_loss_rate=best["stop_loss_rate"],
        rsi_buy_threshold=best["rsi_buy_threshold"],
        rsi_sell_threshold=best["rsi_sell_threshold"],
    )


if __name__ == "__main__":
    main()