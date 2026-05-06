// lib/types.ts

export type Candle = {
  id: string;
  market_id: string;
  open_time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  source: string;
  created_at: string;
};

export type Signal = {
  id: string;
  market_id: string;
  strategy_name: string;
  signal_time: string;
  signal_type: "BUY" | "SELL" | "HOLD";
  price: number;
  reason: string | null;
  created_at: string;
};

export type PortfolioSnapshot = {
  id: string;
  cash_balance: number;
  asset_value: number;
  total_equity: number;
  used_margin: number;
  free_balance: number;
  snapshot_time: string;
};

export type Position = {
  id: string;
  market_id: string;
  side: string;
  qty: number;
  entry_price: number;
  current_price: number | null;
  unrealized_pnl: number;
  realized_pnl: number;
  status: string;
  opened_at: string;
  closed_at: string | null;
};

export type Order = {
  id: string;
  market_id: string;
  side: string;
  order_type: string;
  qty: number;
  filled_price: number | null;
  status: string;
  is_paper: boolean;
  fee: number;
  created_at: string;
};