import type { Order } from "@/lib/types";

type Props = {
  orders: Order[];
};

type FlexibleOrder = Order & {
  price?: number | string | null;
  filled_price?: number | string | null;
  realized_pnl?: number | string | null;
};

function toNumber(value: number | string | null | undefined) {
  const numericValue = Number(value ?? 0);
  return Number.isFinite(numericValue) ? numericValue : 0;
}

function formatNumber(value: number | string | null | undefined, digits = 2) {
  if (value === null || value === undefined) return "-";

  const numericValue = Number(value);

  if (!Number.isFinite(numericValue)) return "-";

  return numericValue.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

function formatDateTime(value: string | null | undefined) {
  if (!value) return "-";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return "-";

  return date.toLocaleString(undefined, {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function normalizeSide(side: string | null | undefined) {
  return String(side ?? "").toUpperCase();
}

function getSideClass(side: string | null | undefined) {
  const normalizedSide = normalizeSide(side);

  if (normalizedSide === "BUY") return "text-emerald-400";
  if (normalizedSide === "SELL") return "text-red-400";
  return "text-slate-300";
}

function getPnlClass(value: number) {
  if (value > 0) return "text-emerald-400";
  if (value < 0) return "text-red-400";
  return "text-slate-300";
}

function getOrderPrice(order: FlexibleOrder) {
  return order.price ?? order.filled_price ?? null;
}

export default function TradeHistoryTable({ orders }: Props) {
  const flexibleOrders = orders ?? [];

  const realizedPnl = flexibleOrders.reduce((sum, order) => {
    return sum + toNumber((order as FlexibleOrder).realized_pnl);
  }, 0);

  const totalFees = flexibleOrders.reduce((sum, order) => {
    return sum + toNumber(order.fee);
  }, 0);

  const filledOrders = flexibleOrders.filter((order) => order.status === "filled");
  const buyCount = flexibleOrders.filter((order) => normalizeSide(order.side) === "BUY").length;
  const sellCount = flexibleOrders.filter((order) => normalizeSide(order.side) === "SELL").length;

  return (
    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-4">
      <div className="mb-4 flex items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-white">Paper Trade History</h2>
        <p className="text-xs text-slate-500">{flexibleOrders.length} orders</p>
      </div>

      <div className="mb-4 grid gap-3 md:grid-cols-4">
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-xs text-slate-500">Realized PnL</p>
          <p className={`mt-1 text-lg font-semibold ${getPnlClass(realizedPnl)}`}>
            {formatNumber(realizedPnl, 2)} USDT
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-xs text-slate-500">Total Fees</p>
          <p className="mt-1 text-lg font-semibold text-white">
            {formatNumber(totalFees, 4)} USDT
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-xs text-slate-500">Filled</p>
          <p className="mt-1 text-lg font-semibold text-white">
            {filledOrders.length}
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-xs text-slate-500">BUY / SELL</p>
          <p className="mt-1 text-lg font-semibold text-white">
            {buyCount} / {sellCount}
          </p>
        </div>
      </div>

      {flexibleOrders.length === 0 ? (
        <p className="text-sm text-slate-500">No orders yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-800 text-slate-400">
              <tr>
                <th className="py-2 pr-4">Time</th>
                <th className="py-2 pr-4">Side</th>
                <th className="py-2 pr-4">Type</th>
                <th className="py-2 pr-4">Qty</th>
                <th className="py-2 pr-4">Price</th>
                <th className="py-2 pr-4">Notional</th>
                <th className="py-2 pr-4">Fee</th>
                <th className="py-2 pr-4">PnL</th>
                <th className="py-2 pr-4">Status</th>
              </tr>
            </thead>
            <tbody>
              {flexibleOrders.map((order) => {
                const flexibleOrder = order as FlexibleOrder;
                const side = normalizeSide(order.side);
                const qty = toNumber(order.qty);
                const price = toNumber(getOrderPrice(flexibleOrder));
                const notional = qty * price;
                const pnl = toNumber(flexibleOrder.realized_pnl);

                return (
                  <tr key={order.id} className="border-b border-slate-800/60">
                    <td className="py-2 pr-4 text-slate-400">
                      {formatDateTime(order.created_at)}
                    </td>
                    <td className={`py-2 pr-4 font-semibold ${getSideClass(order.side)}`}>
                      {side || "-"}
                    </td>
                    <td className="py-2 pr-4 text-slate-300">
                      {order.order_type ?? "-"}
                    </td>
                    <td className="py-2 pr-4 text-slate-300">
                      {formatNumber(qty, 6)}
                    </td>
                    <td className="py-2 pr-4 text-slate-300">
                      {formatNumber(price, 2)}
                    </td>
                    <td className="py-2 pr-4 text-slate-300">
                      {formatNumber(notional, 2)}
                    </td>
                    <td className="py-2 pr-4 text-slate-300">
                      {formatNumber(order.fee, 4)}
                    </td>
                    <td className={`py-2 pr-4 font-medium ${getPnlClass(pnl)}`}>
                      {formatNumber(pnl, 2)}
                    </td>
                    <td className="py-2 pr-4 text-slate-300">
                      {order.status ?? "-"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}