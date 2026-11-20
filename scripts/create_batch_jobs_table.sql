create table batch_jobs (
  id text primary key,
  created_at timestamptz default now(),
  status text default 'running',
  total int default 0,
  completed int default 0,
  auto_confirmed int default 0,
  human_review int default 0,
  governance_mode boolean default false,
  results jsonb default '[]'::jsonb,
  error text
);

alter table batch_jobs enable row level security;

create policy "service role full access"
on batch_jobs for all to service_role
using (true) with check (true);
