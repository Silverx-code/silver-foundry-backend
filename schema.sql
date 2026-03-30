-- ============================================================
-- Silver AI Foundry — Supabase Database Schema
-- Run this in the Supabase SQL editor to set up your tables
-- ============================================================

-- 1. Datasets table
create table if not exists public.datasets (
  id            uuid primary key default gen_random_uuid(),
  original_name text not null,
  storage_path  text not null,
  size_bytes    integer,
  created_at    timestamptz default now()
);

-- 2. Experiments (training jobs) table
create table if not exists public.experiments (
  id               uuid primary key default gen_random_uuid(),
  dataset_id       uuid references public.datasets(id) on delete set null,
  experiment_name  text not null,
  model_type       text not null check (model_type in ('classification','regression','clustering')),
  status           text not null default 'queued' check (status in ('queued','running','done','error')),
  metrics          jsonb,           -- { accuracy, f1, precision, recall, loss, history, ... }
  created_at       timestamptz default now(),
  completed_at     timestamptz
);

-- 3. Row-level security (enable if using Supabase Auth)
alter table public.datasets    enable row level security;
alter table public.experiments enable row level security;

-- Allow service role full access (backend uses service key)
create policy "Service role full access - datasets"
  on public.datasets for all
  using (true)
  with check (true);

create policy "Service role full access - experiments"
  on public.experiments for all
  using (true)
  with check (true);

-- 4. Indexes for fast lookups
create index if not exists idx_experiments_status     on public.experiments(status);
create index if not exists idx_experiments_created_at on public.experiments(created_at desc);
create index if not exists idx_experiments_dataset_id on public.experiments(dataset_id);

-- 5. Supabase Storage bucket (run once in dashboard or via API)
-- Bucket name: "datasets"
-- Settings: private (backend uploads/downloads only)
-- insert into storage.buckets (id, name, public) values ('datasets', 'datasets', false);
