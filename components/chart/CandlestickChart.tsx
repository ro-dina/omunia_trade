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

import type { Candle } from "@/lib/types";
import { calculateSMA } from "@/lib/indicators";
import { generateSmaCrossSignals } from "@/lib/signals";

type Props = {
  candles: Candle[];
};

export default function CandlestickChart({ candles }: Props) {
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const sma20Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const sma50Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const markerApiRef = useRef<{
    setMarkers: (markers: SeriesMarker<Time>[]) => void;
  } | null>(null);

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
    markerApiRef.current = createSeriesMarkers(candlestickSeries, []);

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

    const data = candles.map((candle) => ({
      time: Math.floor(new Date(candle.open_time).getTime() / 1000) as Time,
      open: Number(candle.open),
      high: Number(candle.high),
      low: Number(candle.low),
      close: Number(candle.close),
    }));

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

    const tradeSignals = generateSmaCrossSignals(candles, 20, 50);

    const markers: SeriesMarker<Time>[] = tradeSignals.map((signal) => ({
      time: signal.time as Time,
      position: signal.type === "BUY" ? "belowBar" : "aboveBar",
      color: signal.type === "BUY" ? "#22c55e" : "#ef4444",
      shape: signal.type === "BUY" ? "arrowUp" : "arrowDown",
      text: signal.type,
    }));

    markerApiRef.current?.setMarkers(markers);

    chartRef.current?.timeScale().fitContent();
  }, [candles]);

  return <div ref={chartContainerRef} className="w-full" />;
}