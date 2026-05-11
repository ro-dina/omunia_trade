"use client";

import { useEffect, useMemo, useRef } from "react";
import {
  createChart,
  ColorType,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from "lightweight-charts";

import type { PortfolioSnapshot } from "@/lib/types";

type Props = {
  snapshots?: PortfolioSnapshot[];
};

function formatCurrency(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "-";

  return value.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatPercent(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "-";

  return `${value.toFixed(2)}%`;
}

function removeSpikeDownPoints(
  points: { time: Time; value: number }[],
  dropThresholdRate = 0.05,
) {
  if (points.length < 3) return points;

  return points.filter((point, index) => {
    if (index === 0 || index === points.length - 1) {
      return true;
    }

    const prev = points[index - 1];
    const next = points[index + 1];

    const droppedFromPrev = point.value < prev.value * (1 - dropThresholdRate);
    const recoveredToNext = next.value > point.value * (1 + dropThresholdRate);

    return !(droppedFromPrev && recoveredToNext);
  });
}

function formatRatio(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "-";

  return value.toFixed(2);
}

export default function EquityCurveChart({ snapshots = [] }: Props) {
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const hasFitContentRef = useRef(false);

  const equityPoints = useMemo(() => {
    const dataByTime = new Map<number, number>();

    for (const snapshot of snapshots) {
      const time = Math.floor(new Date(snapshot.snapshot_time).getTime() / 1000);
      const value = Number(snapshot.total_equity);

      if (!Number.isFinite(time) || !Number.isFinite(value)) {
        continue;
      }

      // lightweight-charts requires unique ascending times.
      // If duplicated snapshots exist in the same second, keep the latest value.
      dataByTime.set(time, value);
    }

    const rawPoints = Array.from(dataByTime.entries())
      .sort(([timeA], [timeB]) => timeA - timeB)
      .map(([time, value]) => ({
        time: time as Time,
        value,
      }));

    return removeSpikeDownPoints(rawPoints);
  }, [snapshots]);

  const equityStats = useMemo(() => {
    if (equityPoints.length === 0) {
    return {
      currentEquity: null,
      peakEquity: null,
      maxDrawdown: null,
      sharpeRatio: null,
    };
    }

    let peak = equityPoints[0].value;
    let maxDrawdown = 0;

    for (const point of equityPoints) {
      if (point.value > peak) {
        peak = point.value;
      }

      if (peak > 0) {
        const drawdown = ((peak - point.value) / peak) * 100;
        maxDrawdown = Math.max(maxDrawdown, drawdown);
      }
    }

    const returns: number[] = [];

    for (let i = 1; i < equityPoints.length; i++) {
      const prev = equityPoints[i - 1].value;
      const curr = equityPoints[i].value;

      if (prev > 0) {
        returns.push((curr - prev) / prev);
      }
    }

    const meanReturn =
      returns.length > 0
        ? returns.reduce((sum, value) => sum + value, 0) / returns.length
        : 0;

    const variance =
      returns.length > 1
        ? returns.reduce((sum, value) => sum + (value - meanReturn) ** 2, 0) /
          (returns.length - 1)
        : 0;

    const stdReturn = Math.sqrt(variance);

    const sharpeRatio =
      stdReturn > 0 ? (meanReturn / stdReturn) * Math.sqrt(returns.length) : null;

    return {
      currentEquity: equityPoints[equityPoints.length - 1].value,
      peakEquity: peak,
      maxDrawdown,
      sharpeRatio,
    };
  }, [equityPoints]);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      height: 260,
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

    const lineSeries = chart.addSeries(LineSeries, {
      lineWidth: 2,
      title: "Equity",
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

    seriesRef.current.setData(equityPoints);

    if (!hasFitContentRef.current && equityPoints.length > 0) {
      chartRef.current?.timeScale().fitContent();
      hasFitContentRef.current = true;
    }
  }, [equityPoints]);

  const handleResetZoom = () => {
    chartRef.current?.timeScale().fitContent();
  };

  return (
    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-4">
      <div className="mb-4 flex items-center justify-between gap-4">
        <h2 className="text-lg font-bold">Equity Curve</h2>
        <div className="flex items-center gap-3">
          <p className="text-xs text-slate-500">
            {equityPoints.length} / {snapshots.length} snapshots
          </p>
          <button
            type="button"
            onClick={handleResetZoom}
            className="rounded-lg border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:bg-slate-800"
          >
            Reset zoom
          </button>
        </div>
      </div>

      <div className="mb-4 grid gap-3 md:grid-cols-4">
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-xs text-slate-500">Current Equity</p>
          <p className="mt-1 text-lg font-semibold text-white">
            {formatCurrency(equityStats.currentEquity)} USDT
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-xs text-slate-500">Peak Equity</p>
          <p className="mt-1 text-lg font-semibold text-white">
            {formatCurrency(equityStats.peakEquity)} USDT
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-xs text-slate-500">Max Drawdown</p>
          <p className="mt-1 text-lg font-semibold text-red-400">
            -{formatPercent(equityStats.maxDrawdown)}
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-xs text-slate-500">Sharpe Ratio</p>
          <p className="mt-1 text-lg font-semibold text-white">
            {formatRatio(equityStats.sharpeRatio)}
          </p>
        </div>
      </div>

      <div ref={chartContainerRef} className="w-full" />
    </section>
  );
}