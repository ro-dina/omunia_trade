// lib/indicators.ts

import type { Candle } from "@/lib/types";

export type SMAData = {
  time: number;
  value: number;
};

export function calculateSMA(
  candles: Candle[],
  period: number,
): SMAData[] {
  const result: SMAData[] = [];

  for (let i = 0; i < candles.length; i++) {
    if (i < period - 1) continue;

    const slice = candles.slice(i - period + 1, i + 1);

    const sum = slice.reduce((acc, candle) => {
      return acc + Number(candle.close);
    }, 0);

    const sma = sum / period;

    result.push({
      time: Math.floor(
        new Date(candles[i].open_time).getTime() / 1000,
      ),
      value: sma,
    });
  }

  return result;
}