-- Fields the web app needs that the first schema left out (build plan Parts 16-18, 21, 24, 25).

-- The What-if context (web WhatIfContext), written when an analysis finishes.
alter table experiments add column whatif jsonb;

-- Ghost renders: the divergence threshold and look-ahead for the error chart, and the revision
-- (diagnosis, adopted, note) that produced or was tried against this ghost.
alter table ghost_renders
    add column threshold_mm double precision,
    add column predicted_ahead_s double precision,
    add column revision jsonb;

-- Benchmark rows: what the /benchmark table shows next to the ground truth.
alter table benchmark
    add column clip_id text,
    add column object text not null default '',
    add column unit text not null default '',
    add column expected_model text,
    add column is_failure_case boolean not null default false,
    add column failure_expected text;

-- Wonder's answers to the What-if "Explain" button, so the same question costs nothing twice.
create table explanations (
    experiment_id uuid not null references experiments (id) on delete cascade,
    request_hash text not null,
    text text not null,
    created_at timestamptz not null default now(),
    primary key (experiment_id, request_hash)
);
alter table explanations enable row level security;
