-- Runtime tables/constraints used by the paper trading service.

alter table portfolio_snapshots
  add column if not exists market_id uuid references markets(id);

update portfolio_snapshots
set market_id = (select id from markets limit 1)
where market_id is null
  and (select count(*) from markets) = 1;

create index if not exists idx_portfolio_snapshots_market_time
  on portfolio_snapshots(market_id, snapshot_time desc);

create table if not exists signals (
  id uuid primary key default gen_random_uuid(),
  market_id uuid references markets(id) on delete cascade,
  strategy_name text not null,
  signal_time timestamptz not null,
  signal_type text not null,
  price numeric not null,
  reason text,
  meta jsonb default '{}'::jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create unique index if not exists idx_signals_unique_strategy_time
  on signals(market_id, strategy_name, signal_time);

create index if not exists idx_signals_market_time
  on signals(market_id, signal_time desc);

create unique index if not exists idx_orders_exchange_order_id_unique
  on orders(exchange_order_id)
  where exchange_order_id is not null;

create unique index if not exists idx_positions_one_open_per_market
  on positions(market_id)
  where status = 'open';
