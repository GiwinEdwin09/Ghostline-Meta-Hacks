"""Repository over Postgres (build plan Part 22). One class, three groups of methods.

Every method is one SQL statement or a small transaction. `Repo.transaction()` returns a Repo
bound to a single connection, for callers that need several writes to commit together.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel

from ghostline.store.db import BoundDatabase, Database, J
from ghostline.store.records import (
    Actor,
    Attachment,
    BenchmarkEntry,
    BenchmarkRecord,
    DeclaredCapture,
    Event,
    EventKind,
    Experiment,
    FitRecord,
    GhostRender,
    Job,
    JobKind,
    LlmCall,
    ModelRecord,
    Ref,
    ReviewRecord,
    Setup,
    TrackRecord,
    TrajectoryRecord,
    Trial,
    Upload,
)

_EXPERIMENT_UPDATABLE = {
    "title",
    "description",
    "system_type",
    "status",
    "declared_capture",
    "setup",
    "scene",
    "winner_model_id",
    "conclusion",
    "confidence",
    "review_status",
    "is_example",
    "is_benchmark",
    "visibility",
    "error",
    "replay_check",
}
_EXPERIMENT_JSON = {"declared_capture", "setup", "scene", "conclusion", "replay_check"}
_TRIAL_UPDATABLE = {
    "status",
    "original_filename",
    "size_bytes",
    "sha256",
    "paths",
    "meta_file_ids",
    "probe",
    "timing",
    "trim",
    "n_frames",
    "duration_s",
    "calibration",
    "error",
}
_TRIAL_JSON = {"paths", "meta_file_ids", "probe", "timing", "trim", "calibration"}


class ActiveJobError(Exception):
    """Raised by create_job when the experiment already has a queued or running job."""

    def __init__(self, job: Job):
        super().__init__(f"experiment already has an active {job.kind} job")
        self.job = job


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


class Repo:
    def __init__(self, db: Database | BoundDatabase):
        self.db = db

    @contextmanager
    def transaction(self) -> Iterator["Repo"]:
        """Several repo calls in one transaction: commit on exit, roll back on an exception."""
        if isinstance(self.db, BoundDatabase):  # already inside one
            yield self
            return
        with self.db.connection() as conn:
            yield Repo(BoundDatabase(conn))

    # --- helpers ----------------------------------------------------------------------------
    def _update(
        self,
        table: str,
        where: dict[str, Any],
        fields: dict[str, Any],
        allowed: set[str],
        json_cols: set[str],
    ) -> dict[str, Any] | None:
        if not fields:
            raise ValueError("nothing to update")
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"cannot update {table} columns: {sorted(unknown)}")
        sets = ", ".join(f"{col} = %s" for col in fields)
        params = [
            J(_dump(v)) if col in json_cols and v is not None else v for col, v in fields.items()
        ]
        conds = " and ".join(f"{col} = %s" for col in where)
        sql = f"update {table} set {sets} where {conds} returning *"
        return self.db.fetch_one(sql, [*params, *where.values()])

    # --- experiments ------------------------------------------------------------------------
    def create_experiment(
        self,
        *,
        title: str,
        description: str = "",
        system_type: str = "auto",
        declared_capture: DeclaredCapture | None = None,
        setup: Setup | None = None,
        experiment_id: UUID | None = None,
    ) -> Experiment:
        row = self.db.fetch_one(
            "insert into experiments (id, title, description, system_type, declared_capture, setup)"
            " values (%s, %s, %s, %s, %s, %s) returning *",
            (
                experiment_id or uuid4(),
                title,
                description,
                system_type,
                J(declared_capture or DeclaredCapture()),
                J(setup or Setup()),
            ),
        )
        return Experiment.model_validate(row)

    def get_experiment(self, experiment_id: UUID) -> Experiment | None:
        row = self.db.fetch_one("select * from experiments where id = %s", (experiment_id,))
        return Experiment.model_validate(row) if row else None

    def get_visibility(self, experiment_id: UUID) -> str | None:
        """Just the visibility column: a cheap existence check for the media route."""
        row = self.db.fetch_one(
            "select visibility from experiments where id = %s", (experiment_id,)
        )
        return row["visibility"] if row else None

    def list_experiments(
        self, *, examples_only: bool = False, limit: int = 50, offset: int = 0
    ) -> list[Experiment]:
        where = "where is_example and visibility = 'public'" if examples_only else ""
        rows = self.db.fetch_all(
            f"select * from experiments {where} order by created_at desc limit %s offset %s",
            (limit, offset),
        )
        return [Experiment.model_validate(r) for r in rows]

    def update_experiment(self, experiment_id: UUID, **fields: Any) -> Experiment | None:
        row = self._update(
            "experiments",
            {"id": experiment_id},
            fields,
            _EXPERIMENT_UPDATABLE,
            _EXPERIMENT_JSON,
        )
        return Experiment.model_validate(row) if row else None

    def delete_experiment(self, experiment_id: UUID) -> bool:
        """Delete the experiment and, by cascade, everything that hangs off it."""
        row = self.db.fetch_one(
            "delete from experiments where id = %s returning id", (experiment_id,)
        )
        return row is not None

    # --- uploads ----------------------------------------------------------------------------
    def create_upload(
        self,
        *,
        experiment_id: UUID,
        role: str,
        filename: str,
        size_bytes: int,
        chunk_size: int,
        n_chunks: int,
    ) -> Upload:
        row = self.db.fetch_one(
            "insert into uploads (id, experiment_id, role, filename, size_bytes, chunk_size,"
            " n_chunks) values (%s, %s, %s, %s, %s, %s, %s) returning *",
            (uuid4(), experiment_id, role, filename, size_bytes, chunk_size, n_chunks),
        )
        return Upload.model_validate(row)

    def get_upload(self, upload_id: UUID) -> Upload | None:
        row = self.db.fetch_one("select * from uploads where id = %s", (upload_id,))
        return Upload.model_validate(row) if row else None

    def count_uploads(self, experiment_id: UUID) -> int:
        row = self.db.fetch_one(
            "select count(*) as n from uploads where experiment_id = %s", (experiment_id,)
        )
        return int(row["n"]) if row else 0

    def add_received_chunk(self, upload_id: UUID, n: int) -> Upload | None:
        """Record chunk n as received. Idempotent: the array stays a sorted set."""
        row = self.db.fetch_one(
            "update uploads set received_chunks = (select coalesce("
            "array_agg(distinct c order by c), '{}') from unnest(received_chunks || %s::int) as c)"
            " where id = %s and status = 'open' returning *",
            (n, upload_id),
        )
        return Upload.model_validate(row) if row else None

    def lock_upload(self, upload_id: UUID) -> Upload | None:
        """Read the upload row and hold it until the transaction ends, so two completes of the
        same upload run one after the other."""
        row = self.db.fetch_one("select * from uploads where id = %s for update", (upload_id,))
        return Upload.model_validate(row) if row else None

    def mark_upload_complete(self, upload_id: UUID, trial_id: UUID) -> None:
        self.db.execute(
            "update uploads set status = 'complete', trial_id = %s, completed_at = now()"
            " where id = %s",
            (trial_id, upload_id),
        )

    def stale_open_uploads(self, older_than_hours: float) -> list[Upload]:
        rows = self.db.fetch_all(
            "select * from uploads where status = 'open'"
            " and created_at < now() - make_interval(secs => %s)",
            (older_than_hours * 3600,),
        )
        return [Upload.model_validate(r) for r in rows]

    def delete_upload(self, upload_id: UUID) -> None:
        self.db.execute("delete from uploads where id = %s", (upload_id,))

    def upload_ids_for_experiment(self, experiment_id: UUID) -> list[UUID]:
        rows = self.db.fetch_all(
            "select id from uploads where experiment_id = %s", (experiment_id,)
        )
        return [r["id"] for r in rows]

    # --- trials -----------------------------------------------------------------------------
    def create_trial(
        self,
        *,
        experiment_id: UUID,
        role: str,
        original_filename: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        paths: dict[str, str] | None = None,
        trial_id: UUID | None = None,
    ) -> Trial:
        row = self.db.fetch_one(
            "insert into trials (id, experiment_id, role, original_filename, size_bytes, sha256,"
            " paths) values (%s, %s, %s, %s, %s, %s, %s) returning *",
            (
                trial_id or uuid4(),
                experiment_id,
                role,
                original_filename,
                size_bytes,
                sha256,
                J(paths or {}),
            ),
        )
        return Trial.model_validate(row)

    def get_trial(self, trial_id: UUID) -> Trial | None:
        row = self.db.fetch_one("select * from trials where id = %s", (trial_id,))
        return Trial.model_validate(row) if row else None

    def list_trials(self, experiment_id: UUID) -> list[Trial]:
        rows = self.db.fetch_all(
            "select * from trials where experiment_id = %s order by created_at", (experiment_id,)
        )
        return [Trial.model_validate(r) for r in rows]

    def latest_trial(
        self, experiment_id: UUID, role: str, *, status: str | None = None
    ) -> Trial | None:
        row = self.db.fetch_one(
            "select * from trials where experiment_id = %s and role = %s"
            " and (%s::text is null or status = %s) order by created_at desc limit 1",
            (experiment_id, role, status, status),
        )
        return Trial.model_validate(row) if row else None

    def update_trial(self, trial_id: UUID, **fields: Any) -> Trial | None:
        row = self._update("trials", {"id": trial_id}, fields, _TRIAL_UPDATABLE, _TRIAL_JSON)
        return Trial.model_validate(row) if row else None

    def merge_trial_paths(self, trial_id: UUID, paths: dict[str, str]) -> None:
        self.db.execute("update trials set paths = paths || %s where id = %s", (J(paths), trial_id))

    # --- tracks, trajectories ---------------------------------------------------------------
    def upsert_track(self, track: TrackRecord) -> None:
        self.db.execute(
            "insert into tracks (trial_id, object_key, sam_prompt, cache_key, parquet_path, stats)"
            " values (%s, %s, %s, %s, %s, %s) on conflict (trial_id, object_key) do update set"
            " sam_prompt = excluded.sam_prompt, cache_key = excluded.cache_key,"
            " parquet_path = excluded.parquet_path, stats = excluded.stats",
            (
                track.trial_id,
                track.object_key,
                track.sam_prompt,
                track.cache_key,
                track.parquet_path,
                J(track.stats),
            ),
        )

    def list_tracks(self, trial_ids: list[UUID]) -> list[TrackRecord]:
        rows = self.db.fetch_all(
            "select * from tracks where trial_id = any(%s) order by trial_id, object_key",
            (trial_ids,),
        )
        return [TrackRecord.model_validate(r) for r in rows]

    def upsert_trajectory(self, trajectory: TrajectoryRecord) -> None:
        self.db.execute(
            "insert into trajectories (trial_id, parquet_path, summary) values (%s, %s, %s)"
            " on conflict (trial_id) do update set parquet_path = excluded.parquet_path,"
            " summary = excluded.summary",
            (trajectory.trial_id, trajectory.parquet_path, J(trajectory.summary)),
        )

    def list_trajectories(self, trial_ids: list[UUID]) -> list[TrajectoryRecord]:
        rows = self.db.fetch_all(
            "select * from trajectories where trial_id = any(%s)", (trial_ids,)
        )
        return [TrajectoryRecord.model_validate(r) for r in rows]

    # --- models, fits, reviews, ghosts ------------------------------------------------------
    def upsert_model(self, model: ModelRecord) -> None:
        self.db.execute(
            "insert into models (experiment_id, id, round, source, library_ref, code, name,"
            " mechanism, equations_latex, params_spec, prior_rank, refuting_signature, status,"
            " rejection_reason) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            " on conflict (experiment_id, id) do update set round = excluded.round,"
            " source = excluded.source, library_ref = excluded.library_ref, code = excluded.code,"
            " name = excluded.name, mechanism = excluded.mechanism,"
            " equations_latex = excluded.equations_latex, params_spec = excluded.params_spec,"
            " prior_rank = excluded.prior_rank, refuting_signature = excluded.refuting_signature,"
            " status = excluded.status, rejection_reason = excluded.rejection_reason",
            (
                model.experiment_id,
                model.id,
                model.round,
                model.source,
                model.library_ref,
                model.code,
                model.name,
                model.mechanism,
                model.equations_latex,
                J(model.params_spec) if model.params_spec is not None else None,
                model.prior_rank,
                model.refuting_signature,
                model.status,
                model.rejection_reason,
            ),
        )

    def set_model_status(
        self, experiment_id: UUID, model_id: str, status: str, rejection_reason: str | None = None
    ) -> None:
        self.db.execute(
            "update models set status = %s, rejection_reason = %s"
            " where experiment_id = %s and id = %s",
            (status, rejection_reason, experiment_id, model_id),
        )

    def get_model(self, experiment_id: UUID, model_id: str) -> ModelRecord | None:
        row = self.db.fetch_one(
            "select * from models where experiment_id = %s and id = %s", (experiment_id, model_id)
        )
        return ModelRecord.model_validate(row) if row else None

    def list_models(self, experiment_id: UUID) -> list[ModelRecord]:
        rows = self.db.fetch_all(
            "select * from models where experiment_id = %s"
            " order by round, prior_rank nulls last, id",
            (experiment_id,),
        )
        return [ModelRecord.model_validate(r) for r in rows]

    def insert_fit(self, fit: FitRecord) -> FitRecord:
        row = self.db.fetch_one(
            "insert into fits (id, experiment_id, model_id, trial_id, split, params, train_rmse,"
            " heldout_rmse, heldout_ci, aic, bic, diagnostics, plot_paths)"
            " values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) returning *",
            (
                fit.id,
                fit.experiment_id,
                fit.model_id,
                fit.trial_id,
                J(fit.split),
                J(fit.params),
                fit.train_rmse,
                fit.heldout_rmse,
                J(fit.heldout_ci) if fit.heldout_ci is not None else None,
                fit.aic,
                fit.bic,
                J(fit.diagnostics),
                J(fit.plot_paths),
            ),
        )
        return FitRecord.model_validate(row)

    def list_fits(self, experiment_id: UUID) -> list[FitRecord]:
        rows = self.db.fetch_all(
            "select * from fits where experiment_id = %s order by created_at", (experiment_id,)
        )
        return [FitRecord.model_validate(r) for r in rows]

    def insert_review(self, review: ReviewRecord) -> ReviewRecord:
        row = self.db.fetch_one(
            "insert into reviews (id, experiment_id, round, verdict, issues, summary, response)"
            " values (%s, %s, %s, %s, %s, %s, %s) returning *",
            (
                review.id,
                review.experiment_id,
                review.round,
                review.verdict,
                J(review.issues),
                review.summary,
                J(review.response) if review.response is not None else None,
            ),
        )
        return ReviewRecord.model_validate(row)

    def set_review_response(self, review_id: UUID, response: dict[str, Any]) -> None:
        self.db.execute("update reviews set response = %s where id = %s", (J(response), review_id))

    def list_reviews(self, experiment_id: UUID) -> list[ReviewRecord]:
        rows = self.db.fetch_all(
            "select * from reviews where experiment_id = %s order by round, created_at",
            (experiment_id,),
        )
        return [ReviewRecord.model_validate(r) for r in rows]

    def add_ghost_render(self, ghost: GhostRender) -> GhostRender:
        """Insert a render. Version 0 means the next number for the experiment (v1, v2, ...)."""
        with self.transaction() as tx:
            tx.db.execute(
                "select pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"ghost:{ghost.experiment_id}",),
            )
            row = tx.db.fetch_one(
                "insert into ghost_renders (id, experiment_id, trial_id, model_id, version,"
                " video_path, track_path, slow_video_path, divergence_onset_s, max_error_mm)"
                " values (%s, %s, %s, %s, coalesce(%s, (select coalesce(max(version), 0) + 1"
                " from ghost_renders where experiment_id = %s)), %s, %s, %s, %s, %s) returning *",
                (
                    ghost.id,
                    ghost.experiment_id,
                    ghost.trial_id,
                    ghost.model_id,
                    ghost.version or None,
                    ghost.experiment_id,
                    ghost.video_path,
                    ghost.track_path,
                    ghost.slow_video_path,
                    ghost.divergence_onset_s,
                    ghost.max_error_mm,
                ),
            )
        return GhostRender.model_validate(row)

    def next_ghost_version(self, experiment_id: UUID) -> int:
        row = self.db.fetch_one(
            "select coalesce(max(version), 0) + 1 as v from ghost_renders where experiment_id = %s",
            (experiment_id,),
        )
        return int(row["v"]) if row else 1

    def list_ghosts(self, experiment_id: UUID) -> list[GhostRender]:
        rows = self.db.fetch_all(
            "select * from ghost_renders where experiment_id = %s order by version",
            (experiment_id,),
        )
        return [GhostRender.model_validate(r) for r in rows]

    # --- events (the notebook) --------------------------------------------------------------
    def append_event(
        self,
        experiment_id: UUID,
        *,
        actor: Actor,
        kind: EventKind,
        title: str,
        body_md: str = "",
        phase: str | None = None,
        data: dict[str, Any] | None = None,
        attachments: list[Attachment] | None = None,
        refs: list[Ref] | None = None,
    ) -> Event:
        """Append one event with the next seq. Bumping experiments.event_seq in the same
        transaction serializes writers, so seq is gapless and events commit in order."""
        with self.transaction() as tx:
            counter = tx.db.fetch_one(
                "update experiments set event_seq = event_seq + 1 where id = %s"
                " returning event_seq",
                (experiment_id,),
            )
            if counter is None:
                raise LookupError(f"experiment {experiment_id} not found")
            row = tx.db.fetch_one(
                "insert into events (experiment_id, seq, phase, actor, kind, title, body_md, data,"
                " attachments, refs) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) returning *",
                (
                    experiment_id,
                    counter["event_seq"],
                    phase,
                    actor,
                    kind,
                    title,
                    body_md,
                    J(data or {}),
                    J(attachments or []),
                    J(refs or []),
                ),
            )
        return Event.model_validate(row)

    def list_events(self, experiment_id: UUID, *, after: int = 0, limit: int = 500) -> list[Event]:
        rows = self.db.fetch_all(
            "select * from events where experiment_id = %s and seq > %s order by seq limit %s",
            (experiment_id, after, limit),
        )
        return [Event.model_validate(r) for r in rows]

    def last_event_seq(self, experiment_id: UUID) -> int:
        row = self.db.fetch_one("select event_seq from experiments where id = %s", (experiment_id,))
        return int(row["event_seq"]) if row else 0

    # --- benchmark --------------------------------------------------------------------------
    def upsert_benchmark(self, experiment_id: UUID, entry: BenchmarkEntry) -> BenchmarkRecord:
        with self.transaction() as tx:
            row = tx.db.fetch_one(
                "insert into benchmark (experiment_id, system, quantity, truth_value, truth_sigma,"
                " truth_method, notes) values (%s, %s, %s, %s, %s, %s, %s)"
                " on conflict (experiment_id, quantity) do update set system = excluded.system,"
                " truth_value = excluded.truth_value, truth_sigma = excluded.truth_sigma,"
                " truth_method = excluded.truth_method, notes = excluded.notes returning *",
                (
                    experiment_id,
                    entry.system,
                    entry.quantity,
                    entry.truth_value,
                    entry.truth_sigma,
                    entry.truth_method,
                    entry.notes,
                ),
            )
            tx.db.execute(
                "update experiments set is_benchmark = true where id = %s", (experiment_id,)
            )
        return BenchmarkRecord.model_validate(row)

    def list_benchmark(self) -> list[dict[str, Any]]:
        """Benchmark rows joined with the experiment's title, status and result."""
        return self.db.fetch_all(
            "select b.*, e.title, e.status, e.confidence, e.conclusion, e.visibility"
            " from benchmark b join experiments e on e.id = b.experiment_id"
            " order by b.system, e.created_at, b.quantity"
        )

    # --- model calls and spend --------------------------------------------------------------
    def insert_llm_call(self, call: LlmCall) -> None:
        self.db.execute(
            "insert into llm_calls (experiment_id, phase, provider, model, input_tokens,"
            " cached_tokens, output_tokens, frames, cost_usd, latency_ms, cache_hit, request_hash)"
            " values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                call.experiment_id,
                call.phase,
                call.provider,
                call.model,
                call.input_tokens,
                call.cached_tokens,
                call.output_tokens,
                call.frames,
                call.cost_usd,
                call.latency_ms,
                call.cache_hit,
                call.request_hash,
            ),
        )

    def llm_stats_today(self) -> dict[str, int]:
        row = self.db.fetch_one(
            "select count(*) as calls, count(*) filter (where cache_hit) as cache_hits"
            " from llm_calls where ts >= date_trunc('day', now() at time zone 'utc')"
            " at time zone 'utc'"
        )
        return {"calls": int(row["calls"]), "cache_hits": int(row["cache_hits"])} if row else {}

    # The spend store the budget guard (Part 23) plugs into: add_spend, spend_today and
    # spend_for_experiment.
    def add_spend(
        self,
        *,
        amount_usd: float,
        provider: str,
        experiment_id: UUID | None = None,
        category: Literal["llm", "sam", "reviewer", "other"] = "llm",
        ref: str | None = None,
    ) -> None:
        """Record a charge in the ledger and add it to experiments.cost_usd, atomically."""
        with self.transaction() as tx:
            tx.db.execute(
                "insert into spend_ledger (experiment_id, provider, category, amount_usd, ref)"
                " values (%s, %s, %s, %s, %s)",
                (experiment_id, provider, category, amount_usd, ref),
            )
            if experiment_id is not None:
                tx.db.execute(
                    "update experiments set cost_usd = cost_usd + %s where id = %s",
                    (amount_usd, experiment_id),
                )

    def spend_today(self) -> float:
        row = self.db.fetch_one(
            "select coalesce(sum(amount_usd), 0) as total from spend_ledger"
            " where day = (now() at time zone 'utc')::date"
        )
        return float(row["total"]) if row else 0.0

    def spend_for_experiment(self, experiment_id: UUID) -> float:
        row = self.db.fetch_one(
            "select coalesce(sum(amount_usd), 0) as total from spend_ledger"
            " where experiment_id = %s",
            (experiment_id,),
        )
        return float(row["total"]) if row else 0.0

    # --- jobs -------------------------------------------------------------------------------
    def create_job(
        self,
        *,
        experiment_id: UUID,
        kind: JobKind,
        params: dict[str, Any] | None = None,
        client_ip_hash: str | None = None,
        retry_of: UUID | None = None,
    ) -> Job:
        """Insert a queued job, unless the experiment already has an active one (ActiveJobError).
        The advisory lock makes check-then-insert atomic; it is transaction-scoped, so it also
        works through pgbouncer in transaction mode."""
        with self.transaction() as tx:
            tx.db.execute(
                "select pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"job:{experiment_id}",)
            )
            active = tx.db.fetch_one(
                "select * from jobs where experiment_id = %s and status in ('queued', 'running')",
                (experiment_id,),
            )
            if active:
                raise ActiveJobError(Job.model_validate(active))
            row = tx.db.fetch_one(
                "insert into jobs (id, experiment_id, kind, params, client_ip_hash, retry_of)"
                " values (%s, %s, %s, %s, %s, %s) returning *",
                (uuid4(), experiment_id, kind, J(params or {}), client_ip_hash, retry_of),
            )
        return Job.model_validate(row)

    def get_job(self, job_id: UUID) -> Job | None:
        row = self.db.fetch_one("select * from jobs where id = %s", (job_id,))
        return Job.model_validate(row) if row else None

    def list_jobs(self, experiment_id: UUID | None = None, *, limit: int = 20) -> list[Job]:
        rows = self.db.fetch_all(
            "select * from jobs where (%s::uuid is null or experiment_id = %s)"
            " order by created_at desc limit %s",
            (experiment_id, experiment_id, limit),
        )
        return [Job.model_validate(r) for r in rows]

    def latest_job_for_trial(self, trial_id: UUID) -> Job | None:
        row = self.db.fetch_one(
            "select * from jobs where params->>'trial_id' = %s order by created_at desc limit 1",
            (str(trial_id),),
        )
        return Job.model_validate(row) if row else None

    def active_job(self, experiment_id: UUID) -> Job | None:
        row = self.db.fetch_one(
            "select * from jobs where experiment_id = %s and status in ('queued', 'running')",
            (experiment_id,),
        )
        return Job.model_validate(row) if row else None

    def claim_job(self, job_id: UUID) -> Job | None:
        """Worker start: queued -> running. Returns None if someone else already ran it."""
        row = self.db.fetch_one(
            "update jobs set status = 'running', started_at = now(), heartbeat_at = now(),"
            " attempt = attempt + 1 where id = %s and status = 'queued' returning *",
            (job_id,),
        )
        return Job.model_validate(row) if row else None

    def set_job_call_id(self, job_id: UUID, call_id: str) -> None:
        self.db.execute("update jobs set call_id = %s where id = %s", (call_id, job_id))

    def heartbeat(self, job_id: UUID) -> bool:
        """Returns False once the job is no longer running (for example the reaper failed it)."""
        row = self.db.fetch_one(
            "update jobs set heartbeat_at = now() where id = %s and status = 'running'"
            " returning 1 as ok",
            (job_id,),
        )
        return row is not None

    def set_job_phase(self, job_id: UUID, phase: str, progress: float) -> bool:
        row = self.db.fetch_one(
            "update jobs set phase = %s, progress = greatest(progress, %s), heartbeat_at = now()"
            " where id = %s and status = 'running' returning 1 as ok",
            (phase, min(max(progress, 0.0), 1.0), job_id),
        )
        return row is not None

    def set_job_progress(self, job_id: UUID, progress: float) -> bool:
        row = self.db.fetch_one(
            "update jobs set progress = greatest(progress, %s), heartbeat_at = now()"
            " where id = %s and status = 'running' returning 1 as ok",
            (min(max(progress, 0.0), 1.0), job_id),
        )
        return row is not None

    def finish_job(
        self,
        job_id: UUID,
        *,
        status: Literal["succeeded", "failed"],
        error: str | None = None,
        error_code: str | None = None,
        retryable: bool = True,
    ) -> Job | None:
        """Move an active job to its final state. Returns None if it was already final, which
        makes this safe to call from both a worker and the reaper."""
        row = self.db.fetch_one(
            "update jobs set status = %s, finished_at = now(), error = %s, error_code = %s,"
            " retryable = %s, progress = case when %s = 'succeeded' then 1 else progress end"
            " where id = %s and status in ('queued', 'running') returning *",
            (status, error, error_code, retryable, status, job_id),
        )
        return Job.model_validate(row) if row else None

    def stale_jobs(
        self,
        *,
        running_timeout_s: float,
        queued_timeout_s: float,
        experiment_id: UUID | None = None,
    ) -> list[Job]:
        rows = self.db.fetch_all(
            "select * from jobs where (%s::uuid is null or experiment_id = %s) and ("
            " (status = 'running' and heartbeat_at < now() - make_interval(secs => %s))"
            " or (status = 'queued' and heartbeat_at < now() - make_interval(secs => %s)))",
            (experiment_id, experiment_id, running_timeout_s, queued_timeout_s),
        )
        return [Job.model_validate(r) for r in rows]

    def count_recent_analyses(self, client_ip_hash: str, *, hours: float = 24) -> int:
        """Fresh analyses started from this client in the window. Retries don't count."""
        row = self.db.fetch_one(
            "select count(*) as n from jobs where client_ip_hash = %s"
            " and kind = 'analyze_experiment' and retry_of is null"
            " and created_at > now() - make_interval(secs => %s)",
            (client_ip_hash, hours * 3600),
        )
        return int(row["n"]) if row else 0

    # --- admin ------------------------------------------------------------------------------
    def experiment_costs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return self.db.fetch_all(
            "select id, title, status, created_at, cost_usd from experiments"
            " order by cost_usd desc, created_at desc limit %s",
            (limit,),
        )
