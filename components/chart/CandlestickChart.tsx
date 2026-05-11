"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  createSeriesMarkers,
  ColorType,
  CandlestickSeries,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";

import type { Candle, Order } from "@/lib/types";
import { calculateSMA } from "@/lib/indicators";

type Props = {
  candles: Candle[];
  orders?: Order[];
};

function normalizeSide(side: string | null | undefined) {
  return String(side ?? "").toUpperCase();
}

function getOrderTime(order: Order) {
  return Math.floor(new Date(order.created_at).getTime() / 1000);
}

function findNearestCandleTime(orderTime: number, candleTimes: number[]) {
  if (candleTimes.length === 0) return null;

  let nearestTime = candleTimes[0];
  let nearestDiff = Math.abs(orderTime - nearestTime);

  for (const candleTime of candleTimes) {
    const diff = Math.abs(orderTime - candleTime);

    if (diff < nearestDiff) {
      nearestTime = candleTime;
      nearestDiff = diff;
    }
  }

  // Allow matching to nearby 1m / 5m candles.
  // This is needed because order.created_at is the execution time,
  // while candle.open_time is the bar open time.
  if (nearestDiff > 10 * 60) {
    return null;
  }

  return nearestTime;
}

export default function CandlestickChart({ candles, orders = [] }: Props) {
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const sma20Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const sma50Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const markerApiRef = useRef<{
    setMarkers: (markers: SeriesMarker<Time>[]) => void;
  } | null>(null);
  const hasFitContentRef = useRef(false);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      height: 460,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#d1d5db",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: "#334155",
      },
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    const sma20Series = chart.addSeries(LineSeries, {
      color: "#f59e0b",
      lineWidth: 2,
      title: "SMA20",
    });

    const sma50Series = chart.addSeries(LineSeries, {
      color: "#3b82f6",
      lineWidth: 2,
      title: "SMA50",
    });

    chartRef.current = chart;
    seriesRef.current = candlestickSeries;
    sma20Ref.current = sma20Series;
    sma50Ref.current = sma50Series;
    markerApiRef.current = createSeriesMarkers(candlestickSeries, []) as {
      setMarkers: (markers: SeriesMarker<Time>[]) => void;
    };

    const handleResize = () => {
      if (!chartContainerRef.current) return;

      chart.applyOptions({
        width: chartContainerRef.current.clientWidth,
      });
    };

    handleResize();
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (!seriesRef.current) return;

    const dataByTime = new Map<
      number,
      {
        time: Time;
        open: number;
        high: number;
        low: number;
        close: number;
      }
    >();

    for (const candle of candles) {
      const time = Math.floor(new Date(candle.open_time).getTime() / 1000);
      const open = Number(candle.open);
      const high = Number(candle.high);
      const low = Number(candle.low);
      const close = Number(candle.close);

      if (
        !Number.isFinite(time) ||
        !Number.isFinite(open) ||
        !Number.isFinite(high) ||
        !Number.isFinite(low) ||
        !Number.isFinite(close)
      ) {
        continue;
      }

      dataByTime.set(time, {
        time: time as Time,
        open,
        high,
        low,
        close,
      });
    }

    const data = Array.from(dataByTime.values()).sort(
      (a, b) => Number(a.time) - Number(b.time),
    );

    seriesRef.current.setData(data);

    const sma20 = calculateSMA(candles, 20).map((item) => ({
      time: item.time as Time,
      value: item.value,
    }));

    const sma50 = calculateSMA(candles, 50).map((item) => ({
      time: item.time as Time,
      value: item.value,
    }));

    sma20Ref.current?.setData(sma20);
    sma50Ref.current?.setData(sma50);

    const candleTimeList = data.map((item) => Number(item.time));
    const markers: SeriesMarker<Time>[] = [];

    for (const order of orders) {
      if (order.status !== "filled") {
        continue;
      }

      const orderTime = getOrderTime(order);
      const nearestCandleTime = findNearestCandleTime(orderTime, candleTimeList);

      if (nearestCandleTime === null) {
        continue;
      }

      const time = nearestCandleTime as Time;
      const side = normalizeSide(order.side);

      if (side === "BUY") {
        markers.push({
          time,
          position: "belowBar",
          color: "#22c55e",
          shape: "arrowUp",
          text: "BUY",
        });
      }

      if (side === "SELL") {
        markers.push({
          time,
          position: "aboveBar",
          color: "#ef4444",
          shape: "arrowDown",
          text: "SELL",
        });
      }
    }

    markers.sort((a, b) => Number(a.time) - Number(b.time));

    markerApiRef.current?.setMarkers(markers);

    if (!hasFitContentRef.current && data.length > 0) {
      chartRef.current?.timeScale().fitContent();
      hasFitContentRef.current = true;
    }
  }, [candles, orders]);

  const handleResetZoom = () => {
    chartRef.current?.timeScale().fitContent();
  };

  return (
    <div className="relative">
      <div className="mb-2 flex justify-end">
        <button
          type="button"
          onClick={handleResetZoom}
          className="rounded-lg border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:bg-slate-800"
        >
          Reset zoom
        </button>
      </div>
      <div ref={chartContainerRef} className="w-full" />
    </div>
  );
}