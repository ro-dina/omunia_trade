import type { Order } from "@/lib/types";

type Props = {
  orders: Order[];
};

function formatNumber(value: number | null | undefined, digits = 4) {
  if (value === null || value === undefined) return "-";
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function formatDate(value: string) {
  return new Date(value).toLocaleString();
}

export default function TradeHistoryTable({ orders }: Props) {
  return (
    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-4">
      <h2 className="mb-4 text-lg font-bold">Trade History</h2>

      {orders.length === 0 ? (
        <p className="text-sm text-slate-500">No orders yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-800 text-slate-400">
              <tr>
                <th className="py-2">Time</th>
                <th className="py-2">Side</th>
                <th className="py-2">Type</th>
                <th className="py-2">Qty</th>
                <th className="py-2">Price</th>
                <th className="py-2">Fee</th>
                <th className="py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {orders.map((order) => (
                <tr key={order.id} className="border-b border-slate-800/60">
                  <td className="py-2 text-slate-400">
                    {formatDate(order.created_at)}
                  </td>
                  <td
                    className={
                      order.side === "buy"
                        ? "py-2 font-bold text-emerald-400"
                        : "py-2 font-bold text-red-400"
                    }
                  >
                    {order.side.toUpperCase()}
                  </td>
                  <td className="py-2">{order.order_type}</td>
                  <td className="py-2">{formatNumber(order.qty, 6)}</td>
                  <td className="py-2">{formatNumber(order.filled_price, 2)}</td>
                  <td className="py-2">{formatNumber(order.fee, 4)}</td>
                  <td className="py-2">{order.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}