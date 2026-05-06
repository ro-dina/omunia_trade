import ChartClient from "@/components/chart/ChartClient";

export default function ChartPage() {
  return (
    <main className="min-h-screen bg-slate-950 p-8 text-white">
      <h1 className="mb-2 text-2xl font-bold">OMNIA TRADE Chart</h1>
      <p className="mb-6 text-sm text-slate-400">
        Bybit BTCUSDT 1分足 ローソク足チャート
      </p>

      <ChartClient />
    </main>
  );
}