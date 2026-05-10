"use client";

import { useCallback, useEffect, useState } from "react";

import CandlestickChart from "@/components/chart/CandlestickChart";
import { supabase } from "@/lib/supabase";

import BotStatusPanel from "@/components/trading/BotStatusPanel";
import EquityCurveChart from "@/components/trading/EquityCurveChart";
import TradeHistoryTable from "@/components/trading/TradeHistoryTable";
import type {
  Candle,
  Order,
  PortfolioSnapshot,
  Position,
  Signal,
} from "@/lib/types";

const POLLING_INTERVAL_MS = 30_000;
const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"] as const;
type TradeSymbol = (typeof SYMBOLS)[number];

type BacktestResult = {
  id: string;
  strategy_name: string;
  exchange: string;
  symbol: string;
  market_type: string;
  timeframe: string;
  candle_count: number;
  short_period: number;
  long_period: number;
  take_profit_rate: number;
  stop_loss_rate: number;
  initial_cash: number;
  final_equity: number;
  total_return: number;
  realized_pnl: number;
  buy_count: number;
  sell_count: number;
  win_rate: number;
  take_profit_count: number;
  stop_loss_count: number;
  open_position_value: number;
  created_at: string;
};

function formatNumber(value: number | string | null | undefined, digits = 2) {
  if (value === null || value === undefined) return "-";

  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function formatPercent(value: number | string | null | undefined, digits = 2) {
  if (value === null || value === undefined) return "-";

  return `${formatNumber(value, digits)}%`;
}

function BacktestRankingTable({ results }: { results: BacktestResult[] }) {
  return (
    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-4">
      <h2 className="mb-4 text-lg font-bold text-white">Backtest Ranking</h2>

      {results.length === 0 ? (
        <p className="text-sm text-slate-500">No backtest results yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-800 text-slate-400">
              <tr>
                <th className="py-2 pr-4">Rank</th>
                <th className="py-2 pr-4">TF</th>
                <th className="py-2 pr-4">SMA</th>
                <th className="py-2 pr-4">TP / SL</th>
                <th className="py-2 pr-4">Equity</th>
                <th className="py-2 pr-4">Return</th>
                <th className="py-2 pr-4">PnL</th>
                <th className="py-2 pr-4">Trades</th>
                <th className="py-2 pr-4">Win</th>
              </tr>
            </thead>
            <tbody>
              {results.map((result, index) => (
                <tr key={result.id} className="border-b border-slate-800/60">
                  <td className="py-2 pr-4 text-slate-400">{index + 1}</td>
                  <td className="py-2 pr-4">{result.timeframe}</td>
                  <td className="py-2 pr-4">
                    {result.short_period}/{result.long_period}
                  </td>
                  <td className="py-2 pr-4">
                    {formatPercent(Number(result.take_profit_rate) * 100, 1)} /{" "}
                    {formatPercent(Number(result.stop_loss_rate) * 100, 1)}
                  </td>
                  <td className="py-2 pr-4">
                    {formatNumber(result.final_equity, 2)}
                  </td>
                  <td
                    className={
                      Number(result.total_return) >= 0
                        ? "py-2 pr-4 font-medium text-emerald-400"
                        : "py-2 pr-4 font-medium text-red-400"
                    }
                  >
                    {formatPercent(result.total_return, 2)}
                  </td>
                  <td className="py-2 pr-4">
                    {formatNumber(result.realized_pnl, 2)}
                  </td>
                  <td className="py-2 pr-4">
                    {result.buy_count}/{result.sell_count}
                  </td>
                  <td className="py-2 pr-4">
                    {formatPercent(result.win_rate, 1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default function ChartClient() {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [latestSignal, setLatestSignal] = useState<Signal | null>(null);
  const [latestPortfolio, setLatestPortfolio] =
    useState<PortfolioSnapshot | null>(null);
  const [openPosition, setOpenPosition] = useState<Position | null>(null);
  const [latestOrder, setLatestOrder] = useState<Order | null>(null);
  const [timeframe, setTimeframe] = useState<"1m" | "5m">("1m");
  const [symbol, setSymbol] = useState<TradeSymbol>("BTCUSDT");
  const [orders, setOrders] = useState<Order[]>([]);
  const [portfolioSnapshots, setPortfolioSnapshots] = useState<
    PortfolioSnapshot[]
  >([]);
  const [backtestResults, setBacktestResults] = useState<BacktestResult[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);

  const fetchCandles = useCallback(async () => {
    const { data: marketData, error: marketError } = await supabase
      .from("markets")
      .select("id")
      .eq("exchange", "bybit")
      .eq("symbol", symbol)
      .eq("market_type", "linear")
      .eq("timeframe", timeframe)
      .limit(1)
      .maybeSingle();

    if (marketError || !marketData) {
      setErrorMessage(
        marketError?.message ?? `Market not found: ${symbol} ${timeframe}`,
      );
      setCandles([]);
      setLoading(false);
      return;
    }

    const marketId = marketData.id as string;

    const { data, error } = await supabase
      .from("candles")
      .select("*")
      .eq("market_id", marketId)
      .order("open_time", { ascending: false })
      .limit(300);

    if (error) {
      setErrorMessage(error.message);
      setLoading(false);
      return;
    }

    const sortedCandles = [...((data ?? []) as Candle[])].reverse();

    setCandles(sortedCandles);
    setLastUpdated(new Date().toLocaleTimeString());
    setErrorMessage(null);
    setLoading(false);

    const [
      signalRes,
      portfolioRes,
      positionRes,
      orderRes,
      ordersRes,
      portfolioSnapshotsRes,
      backtestResultsRes,
      signalsRes,
    ] = await Promise.all([
      supabase
        .from("signals")
        .select("*")
        .eq("market_id", marketId)
        .order("signal_time", { ascending: false })
        .limit(1)
        .maybeSingle(),

      supabase
        .from("portfolio_snapshots")
        .select("*")
        .order("snapshot_time", { ascending: false })
        .limit(1)
        .maybeSingle(),

      supabase
        .from("positions")
        .select("*")
        .eq("market_id", marketId)
        .eq("status", "open")
        .limit(1)
        .maybeSingle(),

      supabase
        .from("orders")
        .select("*")
        .eq("market_id", marketId)
        .order("created_at", { ascending: false })
        .limit(1)
        .maybeSingle(),

      supabase
        .from("orders")
        .select("*")
        .eq("market_id", marketId)
        .order("created_at", { ascending: false })
        .limit(20),

      supabase
        .from("portfolio_snapshots")
        .select("*")
        .order("snapshot_time", { ascending: false })
        .limit(200),

      supabase
        .from("backtest_results")
        .select("*")
        .eq("symbol", symbol)
        .eq("timeframe", timeframe)
        .order("final_equity", { ascending: false })
        .limit(10),

      supabase
        .from("signals")
        .select("*")
        .eq("market_id", marketId)
        .order("signal_time", { ascending: false })
        .limit(100),
    ]);

    setLatestSignal((signalRes.data ?? null) as Signal | null);
    setLatestPortfolio((portfolioRes.data ?? null) as PortfolioSnapshot | null);
    setOpenPosition((positionRes.data ?? null) as Position | null);
    setLatestOrder((orderRes.data ?? null) as Order | null);
    setOrders(((ordersRes.data ?? []) as Order[]).reverse());
    setPortfolioSnapshots(
      ((portfolioSnapshotsRes.data ?? []) as PortfolioSnapshot[]).reverse(),
    );
    setBacktestResults((backtestResultsRes.data ?? []) as BacktestResult[]);
    setSignals(((signalsRes.data ?? []) as Signal[]).reverse());
  }, [symbol, timeframe]);

  useEffect(() => {
    const fetchInitialCandles = window.setTimeout(() => {
      void fetchCandles();
    }, 0);

    const timer = window.setInterval(() => {
      void fetchCandles();
    }, POLLING_INTERVAL_MS);

    return () => {
      window.clearTimeout(fetchInitialCandles);
      window.clearInterval(timer);
    };
  }, [fetchCandles]);

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900 p-4">
      <div className="mb-3 flex items-center justify-between text-sm text-slate-400">
        <span>
          {loading
            ? "Loading..."
            : `${symbol} ${timeframe}: ${candles.length} candles loaded`}
        </span>

        <div className="mb-4 flex flex-wrap gap-2">
          {SYMBOLS.map((item) => (
            <button
              key={item}
              onClick={() => setSymbol(item)}
              className={`rounded-lg px-3 py-1 text-sm ${
                symbol === item
                  ? "bg-emerald-500 text-white"
                  : "bg-slate-800 text-slate-400"
              }`}
            >
              {item.replace("USDT", "")}
            </button>
          ))}

          <button
            onClick={() => setTimeframe("1m")}
            className={`rounded-lg px-3 py-1 text-sm ${
              timeframe === "1m"
                ? "bg-blue-500 text-white"
                : "bg-slate-800 text-slate-400"
            }`}
          >
            1m
          </button>

          <button
            onClick={() => setTimeframe("5m")}
            className={`rounded-lg px-3 py-1 text-sm ${
              timeframe === "5m"
                ? "bg-blue-500 text-white"
                : "bg-slate-800 text-slate-400"
            }`}
          >
            5m
          </button>
        </div>

        <span>{lastUpdated ? `Last updated: ${lastUpdated}` : ""}</span>
      </div>

      {candles.length === 0 && !errorMessage ? (
        <p className="text-sm text-slate-500">
          No candles for {symbol} {timeframe} yet.
        </p>
      ) : null}

      {errorMessage ? (
        <p className="text-red-400">Error: {errorMessage}</p>
      ) : candles.length > 0 ? (
        <CandlestickChart candles={candles} orders={orders} />
      ) : null}

      <BotStatusPanel
        latestSignal={latestSignal}
        latestPortfolio={latestPortfolio}
        openPosition={openPosition}
        latestOrder={latestOrder}
      />
      <EquityCurveChart snapshots={portfolioSnapshots ?? []} />
      <TradeHistoryTable orders={orders ?? []} />
      <BacktestRankingTable results={backtestResults} />
    </section>
  );
}