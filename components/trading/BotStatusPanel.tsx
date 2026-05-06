import type {
  Order,
  PortfolioSnapshot,
  Position,
  Signal,
} from "@/lib/types";

type Props = {
  latestSignal: Signal | null;
  latestPortfolio: PortfolioSnapshot | null;
  openPosition: Position | null;
  latestOrder: Order | null;
};

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined) return "-";
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

export default function BotStatusPanel({
  latestSignal,
  latestPortfolio,
  openPosition,
  latestOrder,
}: Props) {
  return (
    <section className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4">
        <p className="text-sm text-slate-400">Latest Signal</p>
        <p className="mt-2 text-2xl font-bold">
          {latestSignal?.signal_type ?? "-"}
        </p>
        <p className="mt-1 text-sm text-slate-500">
          {latestSignal?.reason ?? "No signal yet"}
        </p>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4">
        <p className="text-sm text-slate-400">Portfolio</p>
        <p className="mt-2 text-xl font-bold">
          {formatNumber(latestPortfolio?.total_equity)} USDT
        </p>
        <p className="mt-1 text-sm text-slate-500">
          Cash: {formatNumber(latestPortfolio?.cash_balance)} / Asset:{" "}
          {formatNumber(latestPortfolio?.asset_value)}
        </p>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4">
        <p className="text-sm text-slate-400">Open Position</p>
        <p className="mt-2 text-xl font-bold">
          {openPosition ? openPosition.side.toUpperCase() : "None"}
        </p>
        <p className="mt-1 text-sm text-slate-500">
          PnL: {formatNumber(openPosition?.unrealized_pnl)} USDT
        </p>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4">
        <p className="text-sm text-slate-400">Latest Order</p>
        <p className="mt-2 text-xl font-bold">
          {latestOrder ? latestOrder.side.toUpperCase() : "-"}
        </p>
        <p className="mt-1 text-sm text-slate-500">
          Price: {formatNumber(latestOrder?.filled_price)}
        </p>
      </div>
    </section>
  );
}