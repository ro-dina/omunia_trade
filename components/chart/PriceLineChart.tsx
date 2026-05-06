"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  ColorType,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from "lightweight-charts";

import type { Candle } from "@/lib/types";

type Props = {
  candles: Candle[];
};

export default function PriceLineChart({ candles }: Props) {
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      height: 420,
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
    });

    const lineSeries = chart.addSeries(LineSeries, {
      lineWidth: 2,
    });

    chartRef.current = chart;
    seriesRef.current = lineSeries;

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
      value: Number(candle.close),
    }));

    seriesRef.current.setData(data);
    chartRef.current?.timeScale().fitContent();
  }, [candles]);

  return <div ref={chartContainerRef} className="w-full" />;
}