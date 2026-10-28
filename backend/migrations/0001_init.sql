-- Ghostline schema (build plan Part 22). Plain SQL, applied in file order by
-- `python -m ghostline.store.migrate`. Works on Supabase (Postgres 15) and local Postgres 14+.
-- Timestamps are timestamptz, ids are uuid (unguessable) except models (per-experiment text ids),
-- events (bigserial) and the spend ledger.

create table experiments (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    title text not null,
    description text not null default '',
    system_type text not null default 'auto',
    status text not null default 'draft'
        check (status in ('draft', 'ingesting', 'ready', 'analyzing', 'complete', 'failed')),
    declared_capture jsonb not null default '{"mode": "normal", "fps": null}',
    setup jsonb not null default '{}',
    scene jsonb,
    winner_model_id text,
    conclusion jsonb,
    confidence text check (confidence in ('high', 'medium', 'low')),
    review_status text,
    is_example boolean not null default false,
    is_benchmark boolean not null default false,
    visibility text not null default 'private' check (visibility in ('private', 'public')),
    cost_usd double precision not null default 0,
    error text,
    replay_check jsonb,
    -- Last notebook sequence number handed out. Updating this row serializes writers per
    -- experiment, so events commit in seq order and pollers never skip one.
    event_seq integer not null default 0
);
create index experiments_examples_idx on experiments (created_at desc) where is_example;

create table trials (
    id uuid primary key default gen_random_uuid(),
    experiment_id uuid not null references experiments (id) on delete cascade,
    role text not null check (role in ('fit', 'prove')),
    status text not null default 'uploaded'
        check (status in ('uploaded', 'ingesting', 'ready', 'failed')),
    original_filename text,
    size_bytes bigint,
    sha256 text,
    -- File paths are relative to the experiment folder on the Volume (experiments/{exp}/).
    paths jsonb not null default '{}',
    meta_file_ids jsonb not null default '{}',
    probe jsonb,
    timing jsonb,
    trim jsonb,
    n_frames integer,
    duration_s double precision,
    calibration jsonb,
    error text,
    created_at timestamptz not null default now()
);
create index trials_experiment_idx on trials (experiment_id, role, created_at);

create table tracks (
    trial_id uuid not null references trials (id) on delete cascade,
    object_key text not null,
    sam_prompt text,
    cache_key text,
    parquet_path text,
    stats jsonb not null default '{}',
    primary key (trial_id, object_key)
);

create table trajectories (
    trial_id uuid primary key references trials (id) on delete cascade,
    parquet_path text,
    summary jsonb not null default '{}'
);

create table models (
    experiment_id uuid not null references experiments (id) on delete cascade,
    id text not null,
    round integer not null default 0,
    source text not null default 'library' check (source in ('library', 'wonder')),
    library_ref text,
    code text,
    name text not null,
    mechanism text not null default '',
    equations_latex text not null default '',
    params_spec jsonb,
    prior_rank integer,
    refuting_signature text not null default '',
    status text not null default 'proposed',
    rejection_reason text,
    created_at timestamptz not null default now(),
    primary key (experiment_id, id)
);

create table fits (
    id uuid primary key default gen_random_uuid(),
    experiment_id uuid not null references experiments (id) on delete cascade,
    model_id text not null,
    trial_id uuid not null references trials (id) on delete cascade,
    split jsonb not null default '{}',
    params jsonb not null default '[]',
    train_rmse double precision,
    heldout_rmse double precision,
    heldout_ci jsonb,
    aic double precision,
    bic double precision,
    diagnostics jsonb not null default '{}',
    plot_paths jsonb not null default '{}',
    created_at timestamptz not null default now(),
    foreign key (experiment_id, model_id) references models (experiment_id, id) on delete cascade
);
create index fits_experiment_idx on fits (experiment_id, created_at);

create table reviews (
    id uuid primary key default gen_random_uuid(),
    experiment_id uuid not null references experiments (id) on delete cascade,
    round integer not null default 1,
    verdict text not null,
    issues jsonb not null default '[]',
    summary text not null default '',
    response jsonb,
    created_at timestamptz not null default now()
);
create index reviews_experiment_idx on reviews (experiment_id, round);

create table ghost_renders (
    id uuid primary key default gen_random_uuid(),
    experiment_id uuid not null references experiments (id) on delete cascade,
    trial_id uuid not null references trials (id) on delete cascade,
    model_id text,
    version integer not null,
    video_path text,
    track_path text,
    slow_video_path text,
    divergence_onset_s double precision,
    max_error_mm double precision,
    created_at timestamptz not null default now(),
    unique (experiment_id, version)
);

create table events (
    id bigserial primary key,
    experiment_id uuid not null references experiments (id) on delete cascade,
    seq integer not null,
    ts timestamptz not null default now(),
    phase text,
    actor text not null,
    kind text not null,
    title text not null,
    body_md text not null default '',
    data jsonb not null default '{}',
    attachments jsonb not null default '[]',
    refs jsonb not null default '[]',
    unique (experiment_id, seq)
);

-- The notebook is append-only (Part 20): events are never edited. Deletes happen only through
-- the cascade when an experiment is deleted.
create function events_reject_update() returns trigger language plpgsql as $$
begin
    raise exception 'events are append-only';
end;
$$;
create trigger events_append_only before update on events
    for each row execute function events_reject_update();

create table llm_calls (
    id bigserial primary key,
    ts timestamptz not null default now(),
    experiment_id uuid references experiments (id) on delete cascade,
    phase text,
    provider text not null,
    model text not null,
    input_tokens integer not null default 0,
    cached_tokens integer not null default 0,
    output_tokens integer not null default 0,
    frames integer not null default 0,
    cost_usd double precision not null default 0,
    latency_ms integer,
    cache_hit boolean not null default false,
    request_hash text
);
create index llm_calls_experiment_idx on llm_calls (experiment_id);
create index llm_calls_ts_idx on llm_calls (ts);

create table jobs (
    id uuid primary key default gen_random_uuid(),
    experiment_id uuid not null references experiments (id) on delete cascade,
    kind text not null check (kind in (
        'ingest_trial', 'analyze_experiment', 'prove_trial', 'revise', 'rerun', 'export_verify'
    )),
    status text not null default 'queued'
        check (status in ('queued', 'running', 'succeeded', 'failed')),
    phase text,
    progress double precision not null default 0 check (progress >= 0 and progress <= 1),
    params jsonb not null default '{}',
    attempt integer not null default 0,
    retry_of uuid references jobs (id) on delete set null,
    client_ip_hash text,
    call_id text,
    created_at timestamptz not null default now(),
    started_at timestamptz,
    heartbeat_at timestamptz not null default now(),
    finished_at timestamptz,
    error text,
    error_code text,
    retryable boolean not null default true
);
-- Only one active job per experiment. create_job also takes an advisory lock so the loser gets a
-- clean "job already running" error instead of a unique violation.
create unique index jobs_one_active_idx on jobs (experiment_id) where status in ('queued', 'running');
create index jobs_created_idx on jobs (created_at desc);
create index jobs_ip_idx on jobs (client_ip_hash, created_at) where client_ip_hash is not null;

create table benchmark (
    experiment_id uuid not null references experiments (id) on delete cascade,
    system text not null,
    quantity text not null,
    truth_value double precision not null,
    truth_sigma double precision,
    truth_method text not null default '',
    notes text not null default '',
    primary key (experiment_id, quantity)
);

create table uploads (
    id uuid primary key default gen_random_uuid(),
    experiment_id uuid not null references experiments (id) on delete cascade,
    role text not null check (role in ('fit', 'prove')),
    filename text not null,
    size_bytes bigint not null check (size_bytes > 0),
    chunk_size integer not null,
    n_chunks integer not null check (n_chunks > 0),
    received_chunks integer[] not null default '{}',
    status text not null default 'open' check (status in ('open', 'complete')),
    trial_id uuid references trials (id) on delete set null,
    created_at timestamptz not null default now(),
    completed_at timestamptz
);
create index uploads_experiment_idx on uploads (experiment_id);

-- Spend storage for the budget guard (Part 23). One row per charge. It survives experiment
-- deletion (experiment_id becomes null) so the daily cap still counts deleted experiments.
create table spend_ledger (
    id bigserial primary key,
    ts timestamptz not null default now(),
    day date not null default ((now() at time zone 'utc')::date),
    experiment_id uuid references experiments (id) on delete set null,
    provider text not null,
    category text not null default 'llm',
    amount_usd double precision not null check (amount_usd >= 0),
    ref text
);
create index spend_ledger_day_idx on spend_ledger (day);
create index spend_ledger_experiment_idx on spend_ledger (experiment_id);

-- The backend connects with a privileged role, which bypasses row level security. Turning it on
-- without policies hides every table from Supabase's public Data API (anon/authenticated roles).
alter table experiments enable row level security;
alter table trials enable row level security;
alter table tracks enable row level security;
alter table trajectories enable row level security;
alter table models enable row level security;
alter table fits enable row level security;
alter table reviews enable row level security;
alter table ghost_renders enable row level security;
alter table events enable row level security;
alter table llm_calls enable row level security;
alter table jobs enable row level security;
alter table benchmark enable row level security;
alter table uploads enable row level security;
alter table spend_ledger enable row level security;
