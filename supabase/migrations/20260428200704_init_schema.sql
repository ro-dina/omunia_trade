-- =========================================
-- extensions
-- =========================================
create extension if not exists pgcrypto;;

-- =========================================
-- markets
-- =========================================
create table markets (
  id uuid primary key default gen_random_uuid(),
  exchange text not null,
  symbol text not null,
  market_type text not null,
  timeframe text not null,
  is_active boolean default true,
  created_at timestamptz default now()
);

-- =========================================
-- candles
-- =========================================
create table candles (
  id uuid primary key default gen_random_uuid(),
  market_id uuid references markets(id) on delete cascade,
  open_time timestamptz not null,
  open numeric not null,
  high numeric not null,
  low numeric not null,
  close numeric not null,
  volume numeric not null,
  source text default 'bybit',
  created_at timestamptz default now(),
  unique (market_id, open_time)
);

-- =========================================
-- orders (paper & real)
-- =========================================
create table orders (
  id uuid primary key default gen_random_uuid(),
  market_id uuid references markets(id),
  side text not null,              -- buy / sell
  order_type text not null,        -- market / limit
  qty numeric not null,
  requested_price numeric,
  filled_price numeric,
  status text not null,            -- pending / filled / canceled
  is_paper boolean default true,
  exchange_order_id text,
  fee numeric default 0,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- =========================================
-- positions
-- =========================================
create table positions (
  id uuid primary key default gen_random_uuid(),
  market_id uuid references markets(id),
  side text not null,              -- long / short
  qty numeric not null,
  entry_price numeric not null,
  current_price numeric,
  unrealized_pnl numeric default 0,
  realized_pnl numeric default 0,
  status text not null,            -- open / closed
  opened_at timestamptz default now(),
  closed_at timestamptz,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- =========================================
-- portfolio snapshots
-- =========================================
create table portfolio_snapshots (
  id uuid primary key default gen_random_uuid(),
  cash_balance numeric not null,
  asset_value numeric not null,
  total_equity numeric not null,
  used_margin numeric default 0,
  free_balance numeric default 0,
  snapshot_time timestamptz default now(),
  created_at timestamptz default now()
);

-- =========================================
-- index（地味に重要）
-- =========================================
create index idx_candles_market_time on candles(market_id, open_time desc);
create index idx_orders_market on orders(market_id);
create index idx_positions_market on positions(market_id);