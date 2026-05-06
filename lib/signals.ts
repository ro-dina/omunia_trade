// lib/signals.ts

import type { Candle } from "@/lib/types";
import { calculateSMA } from "@/lib/indicators";

export type TradeSignal = {
  time: number;
  type: "BUY" | "SELL";
  price: number;
};

export function generateSmaCrossSignals(
  candles: Candle[],
  shortPeriod = 20,
  longPeriod = 50,
): TradeSignal[] {
  const shortSma = calculateSMA(candles, shortPeriod);
  const longSma = calculateSMA(candles, longPeriod);

  const shortMap = new Map(shortSma.map((item) => [item.time, item.value]));
  const longMap = new Map(longSma.map((item) => [item.time, item.value]));

  const signals: TradeSignal[] = [];

  for (let i = 1; i < candles.length; i++) {
    const prevOpenTime = new Date(candles[i - 1].open_time).getTime();
    const currOpenTime = new Date(candles[i].open_time).getTime();

    const diffMinutes = (currOpenTime - prevOpenTime) / 1000 / 60;

    if (diffMinutes !== 1) {
      continue;
    }

    const prevTime = Math.floor(prevOpenTime / 1000);
    const currTime = Math.floor(currOpenTime / 1000);

    const prevShort = shortMap.get(prevTime);
    const prevLong = longMap.get(prevTime);
    const currShort = shortMap.get(currTime);
    const currLong = longMap.get(currTime);

    if (
      prevShort === undefined ||
      prevLong === undefined ||
      currShort === undefined ||
      currLong === undefined
    ) {
      continue;
    }

    const prevDiff = prevShort - prevLong;
    const currDiff = currShort - currLong;

    if (prevDiff <= 0 && currDiff > 0) {
      signals.push({
        time: currTime,
        type: "BUY",
        price: Number(candles[i].close),
      });
    }

    if (prevDiff >= 0 && currDiff < 0) {
      signals.push({
        time: currTime,
        type: "SELL",
        price: Number(candles[i].close),
      });
    }
  }

  return signals;
}