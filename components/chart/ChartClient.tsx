"use client";

import { useEffect, useState } from "react";

import CandlestickChart from "@/components/chart/CandlestickChart";
import { supabase } from "@/lib/supabase";

import BotStatusPanel from "@/components/trading/BotStatusPanel";
import type {
  Candle,
  Order,
  PortfolioSnapshot,
  Position,
  Signal,
} from "@/lib/types";

const POLLING_INTERVAL_MS = 30_000;

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

  async function fetchCandles() {
    const { data, error } = await supabase
      .from("candles")
      .select("*")
      .eq("source", "bybit-mainnet-public")
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

    const [signalRes, portfolioRes, positionRes, orderRes] = await Promise.all([
      supabase
        .from("signals")
        .select("*")
        .order("signal_time", { ascending: false })
        .limit(1)
        .single(),

      supabase
        .from("portfolio_snapshots")
        .select("*")
        .order("snapshot_time", { ascending: false })
        .limit(1)
        .single(),

      supabase
        .from("positions")
        .select("*")
        .eq("status", "open")
        .limit(1)
        .maybeSingle(),

      supabase
        .from("orders")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(1)
        .maybeSingle(),
]);

setLatestSignal((signalRes.data ?? null) as Signal | null);
setLatestPortfolio((portfolioRes.data ?? null) as PortfolioSnapshot | null);
setOpenPosition((positionRes.data ?? null) as Position | null);
setLatestOrder((orderRes.data ?? null) as Order | null);
  }

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
  }, []);

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900 p-4">
      <div className="mb-3 flex items-center justify-between text-sm text-slate-400">
        <span>
          {loading
            ? "Loading..."
            : `${candles.length} candles loaded`}
        </span>

        <span>
          {lastUpdated ? `Last updated: ${lastUpdated}` : ""}
        </span>
      </div>

      {errorMessage ? (
        <p className="text-red-400">Error: {errorMessage}</p>
      ) : (
        <CandlestickChart candles={candles} />
      )}
      <BotStatusPanel
        latestSignal={latestSignal}
        latestPortfolio={latestPortfolio}
        openPosition={openPosition}
        latestOrder={latestOrder}
      />
    </section>
  );
}