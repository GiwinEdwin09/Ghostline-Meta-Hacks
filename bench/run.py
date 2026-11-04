"""The benchmark runner (build plan Part 25): every manifest clip through the deployed pipeline.

    export GHOSTLINE_API_BASE=https://<workspace>--ghostline-api.modal.run
    export GHOSTLINE_PASSCODE=...                  # APP_PASSCODE; never put it in a file
    export BENCH_LATITUDE=..  BENCH_ALTITUDE_M=..  # where the g clips were filmed (stays local)

    python -m bench.run --replay                   # back end in LLM_MODE=replay: free
    python -m bench.run --live --yes               # back end in LLM_MODE=live: about $1.50 a clip
    python -m bench.run --report-only              # re-score saved results; no API calls
    python -m bench.run --sync-truth               # PUT truths measured since, then re-score

--replay refuses to run against a back end that is not in replay mode, and --live refuses one that
is, so a flag can't spend money by accident. Per clip it creates the experiment, uploads in
chunks, waits for ingest, analyzes, waits, collects the stored result, and PUTs the ground truth.
Progress is saved per clip in bench/state/, so an interrupted run resumes where it stopped and
clips that already have results are skipped (--force starts them over, --retry-failed retries the
ones that failed). Outputs: bench/results.csv, results.md and results.json.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from bench.client import (
    ApiFailure,
    GhostlineClient,
    JobFailed,
    QuotaExceeded,
    WaitTimeout,
    settings_from_env,
)
from bench.metrics import ClipResult, Summary, blank_result, score_clip, summarize
from bench.report import write_all
from bench.truth import BENCH_DIR, ManifestRow, Truth, load_manifest, resolve_truth
from ghostline.ops.versions import prompt_version

COST_PER_CLIP_USD = 1.5  # Part 25's estimate, including Wonder
FAILURE_QUANTITY = "expect_low_confidence"  # the benchmark row that records a bad clip's purpose
EXIT_OK, EXIT_ERROR, EXIT_REFUSED, EXIT_STOPPED = 0, 1, 2, 3


@dataclass
class RunConfig:
    mode: str = "replay"  # live | replay
    manifest: Path = BENCH_DIR / "manifest.csv"
    clips_dir: Path = BENCH_DIR / "clips"
    state_dir: Path = BENCH_DIR / "state"
    out_dir: Path = BENCH_DIR
    only: list[str] | None = None
    force: bool = False
    retry_failed: bool = False
    publish: bool = False
    upload_truth: bool = True
    max_cost_usd: float | None = None
    latitude: float | None = None
    altitude_m: float | None = None
    workers: int = 1
    poll_s: float = 1.0
    ingest_timeout_s: float = 900.0
    analysis_timeout_s: float = 3900.0  # the worker's own limit is an hour
    job_retries: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


# --- per-clip progress, saved so an interrupted run can resume --------------------------------
class ClipState(BaseModel):
    clip_id: str
    stage: str = "new"  # new | created | uploaded | ingested | analyzing | done | failed
    failed_stage: str | None = None
    experiment_id: str | None = None
    upload_id: str | None = None
    trial_id: str | None = None
    ingest_job_id: str | None = None
    analyze_job_id: str | None = None
    status: str | None = None  # the final status once done or failed
    error: str | None = None
    duration_s: float | None = None
    snapshot: dict[str, Any] = Field(default_factory=dict)  # experiment + events, to re-score


class StateStore:
    def __init__(self, folder: Path):
        self.folder = folder
        self._lock = threading.Lock()

    def _path(self, clip_id: str) -> Path:
        return self.folder / f"{clip_id}.json"

    def load(self, clip_id: str) -> ClipState:
        try:
            return ClipState.model_validate_json(self._path(clip_id).read_text())
        except FileNotFoundError:
            return ClipState(clip_id=clip_id)

    def save(self, state: ClipState) -> None:
        with self._lock:
            self.folder.mkdir(parents=True, exist_ok=True)
            tmp = self._path(state.clip_id).with_suffix(".tmp")
            tmp.write_text(state.model_dump_json(indent=1))
            os.replace(tmp, self._path(state.clip_id))

    def delete(self, clip_id: str) -> None:
        self._path(clip_id).unlink(missing_ok=True)


class StopRun(Exception):
    """Stop starting clips: a quota was hit or the spend cap was reached."""


def truth_entry(row: ManifestRow, truth: Truth | None) -> dict[str, Any] | None:
    """The body of PUT /benchmark/{experiment}: a ground truth, or the 'expected low confidence'
    record of a bad clip. Story clips have neither."""
    if row.kind == "failure":
        return {
            "system": row.system,
            "quantity": FAILURE_QUANTITY,
            "truth_value": 1.0,
            "truth_sigma": None,
            "truth_method": f"expected low confidence: {row.failure_expected.replace('_', ' ')}",
            "notes": row.setup_notes,
        }
    if truth is None or not row.quantity:
        return None
    return {
        "system": row.system,
        "quantity": row.quantity,
        "truth_value": truth.value,
        "truth_sigma": truth.sigma,
        "truth_method": truth.method,
        "notes": row.setup_notes,
    }


class Benchmark:
    def __init__(
        self,
        cfg: RunConfig,
        client: GhostlineClient | None,
        rows: list[ManifestRow],
        *,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.client = client
        self.rows = rows
        self.log = log
        self._clock = clock
        self.store = StateStore(cfg.state_dir)
        self.stop_reason: str | None = None
        self._lock = threading.Lock()
        self._truths: dict[str, Truth | None] = {}

    # --- helpers ------------------------------------------------------------------------------
    def say(self, clip_id: str, message: str) -> None:
        with self._lock:
            self.log(f"[{clip_id}] {message}")

    def truth_for(self, row: ManifestRow) -> Truth | None:
        if row.clip_id not in self._truths:
            try:
                truth = resolve_truth(
                    row,
                    latitude=self.cfg.latitude,
                    altitude_m=self.cfg.altitude_m,
                    clips_dir=self.cfg.clips_dir,
                )
            except (OSError, ValueError) as exc:  # a missing audio file must not stop the run
                self.say(row.clip_id, f"ground truth unavailable: {exc}")
                truth = None
            self._truths[row.clip_id] = truth
        return self._truths[row.clip_id]

    def rescore(self, row: ManifestRow, state: ClipState) -> ClipResult:
        """The clip's result from what the API stored, against the manifest's current truth."""
        truth = self.truth_for(row)
        if state.stage not in ("done", "failed") or not state.snapshot:
            return blank_result(row, truth, "pending")
        return score_clip(
            row,
            truth,
            state.snapshot.get("experiment"),
            state.snapshot.get("events", []),
            status=state.status,
            error=state.error,
            duration_s=state.duration_s,
        )

    def _check_cost(self) -> None:
        cap = self.cfg.max_cost_usd
        if cap is None:
            return
        spent = 0.0
        for row in self.rows:
            snap = self.store.load(row.clip_id).snapshot.get("experiment") or {}
            spent += float(snap.get("cost_usd") or 0.0)
        if spent >= cap:
            raise StopRun(f"model spend ${spent:.2f} reached the cap of ${cap:.2f}")

    # --- one clip -----------------------------------------------------------------------------
    def process_clip(self, row: ManifestRow) -> ClipResult:
        state = self.store.load(row.clip_id)
        if state.stage == "failed" and self.cfg.retry_failed:
            # Go back to the stage that failed: the experiment and its upload are kept, so a
            # transient failure costs a re-upload at most, never a second experiment.
            resume = {"analyzing": "ingested", "uploaded": "created", "created": "created"}
            state.stage = resume.get(state.failed_stage or "", "new")
            if state.stage == "created":
                state.upload_id = None
            state.status = state.error = state.failed_stage = None
        if self.cfg.force:
            self.store.delete(row.clip_id)
            state = ClipState(clip_id=row.clip_id)
        if state.stage in ("done", "failed"):
            self.say(row.clip_id, f"already has a result ({state.status}); skipping")
            return self.rescore(row, state)
        video = self.cfg.clips_dir / row.file
        if state.stage == "new" and not video.is_file():
            self.say(row.clip_id, f"no video at {video}; skipping")
            return blank_result(row, self.truth_for(row), "missing_file")
        if self.stop_reason:
            return blank_result(row, self.truth_for(row), "pending")
        assert self.client is not None, "a run needs an API client"
        client = self.client
        started = self._clock()
        try:
            self._run_stages(row, state, video)
        except QuotaExceeded as exc:
            self.stop_reason = f"{exc.code}: {exc.detail}"
            self.say(row.clip_id, f"stopping the run: {self.stop_reason}")
            return blank_result(row, self.truth_for(row), "pending")
        except (JobFailed, WaitTimeout, ApiFailure, httpx.HTTPError) as exc:
            # This clip failed; the others still run. --retry-failed picks it up again.
            state.failed_stage, state.stage = state.stage, "failed"
            state.status = "timeout" if isinstance(exc, WaitTimeout) else "failed"
            state.error = str(exc)
            self.say(row.clip_id, f"{state.status}: {exc}")
        else:
            state.stage, state.status = "done", "complete"
            state.error = None
        state.duration_s = round(self._clock() - started, 1)
        if state.experiment_id:  # keep what the API stored, whatever happened
            try:
                exp = client.get_experiment(state.experiment_id)
                events = client.all_events(state.experiment_id)
                state.snapshot = {"experiment": exp, "events": events}
            except (ApiFailure, httpx.HTTPError) as exc:
                state.error = state.error or f"couldn't read the result back: {exc}"
        self.store.save(state)
        result = self.rescore(row, state)
        self.say(
            row.clip_id,
            f"{state.status} in {state.duration_s:.0f} s: "
            f"{result.model_selected or '-'} {result.confidence or '-'}",
        )
        return result

    def _run_stages(self, row: ManifestRow, state: ClipState, video: Path) -> None:
        client, cfg = self.client, self.cfg
        assert client is not None
        say = lambda msg: self.say(row.clip_id, msg)  # noqa: E731

        def save(**fields: Any) -> None:
            for key, value in fields.items():
                setattr(state, key, value)
            self.store.save(state)

        if state.stage == "new":
            setup: dict[str, Any] = {}
            if row.plane_offset_cm is not None:
                setup["plane_offset_cm"] = row.plane_offset_cm
            if row.pendulum_length_cm is not None:
                setup["pendulum_length_cm"] = row.pendulum_length_cm
            created = client.create_experiment(
                title=f"Benchmark {row.clip_id}: {row.object or row.system}",
                system_hint=row.system,
                description=row.setup_notes,
                declared_capture=row.declared_capture,
                setup=setup,
            )
            save(experiment_id=created["id"], stage="created")
            say(f"experiment {created['id']}")
            self._put_truth(row, state)
            if cfg.publish:
                client.patch_experiment(created["id"], visibility="public")

        if state.stage == "created":
            say(f"uploading {video.name} ({video.stat().st_size / 1e6:.1f} MB)")
            done = client.upload_file(
                state.experiment_id,
                video,
                upload_id=state.upload_id,
                on_upload=lambda upload_id: save(upload_id=upload_id),
            )
            save(trial_id=done["trial_id"], ingest_job_id=done["job_id"], stage="uploaded")

        if state.stage == "uploaded":
            say("preparing the clip")
            job_id = self._wait(row.clip_id, state.ingest_job_id, cfg.ingest_timeout_s)
            save(ingest_job_id=job_id, stage="ingested")

        if state.stage == "ingested":
            if row.trim_start_s is not None or row.trim_end_s is not None:
                client.set_trim(state.trial_id, row.trim_start_s, row.trim_end_s)
                say(f"trimmed to {row.trim_start_s} to {row.trim_end_s} s")
            job = client.analyze(state.experiment_id, state.trial_id)
            save(analyze_job_id=job["id"], stage="analyzing")
            say("analyzing")

        if state.stage == "analyzing":
            job_id = self._wait(row.clip_id, state.analyze_job_id, cfg.analysis_timeout_s)
            save(analyze_job_id=job_id)

    def _wait(self, clip_id: str, job_id: str | None, timeout_s: float) -> str:
        """Wait for a job, retrying a failed one the API says is retryable (the Retry button).
        Returns the id of the job that succeeded, which differs from the first after a retry."""
        client = self.client
        assert client is not None and job_id
        for attempt in range(self.cfg.job_retries + 1):
            try:
                client.wait_for_job(job_id, timeout_s=timeout_s, poll_s=self.cfg.poll_s)
                return job_id
            except JobFailed as exc:
                if attempt == self.cfg.job_retries or not exc.retryable:
                    raise
                job_id = client.retry_job(job_id)["id"]
                self.say(clip_id, f"job failed ({exc}); retrying as {job_id}")
        raise AssertionError("unreachable")

    def _put_truth(self, row: ManifestRow, state: ClipState) -> bool:
        if not self.cfg.upload_truth or not state.experiment_id or self.client is None:
            return False
        entry = truth_entry(row, self.truth_for(row))
        if entry is None:
            return False
        self.client.put_benchmark(state.experiment_id, entry)
        return True

    # --- the whole run --------------------------------------------------------------------------
    def selected_rows(self) -> list[ManifestRow]:
        if not self.cfg.only:
            return self.rows
        unknown = sorted(set(self.cfg.only) - {r.clip_id for r in self.rows})
        if unknown:
            raise SystemExit(f"not in the manifest: {', '.join(unknown)}")
        return [r for r in self.rows if r.clip_id in self.cfg.only]

    def run(self) -> list[ClipResult]:
        selected = self.selected_rows()

        def work(row: ManifestRow) -> None:
            if self.stop_reason:
                return
            try:
                self._check_cost()
                self.process_clip(row)
            except StopRun as stop:
                self.stop_reason = str(stop)

        if self.cfg.workers > 1:
            with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
                list(pool.map(work, selected))
        else:
            for row in selected:
                work(row)
        return self.results()

    def results(self) -> list[ClipResult]:
        return [self.rescore(r, self.store.load(r.clip_id)) for r in self.rows]

    def sync_truth(self) -> int:
        """PUT the ground truth of every clip that has an experiment. Idempotent."""
        count = 0
        for row in self.rows:
            state = self.store.load(row.clip_id)
            if self._put_truth(row, state):
                count += 1
        return count


# --- command line --------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m bench.run",
        description="Run the benchmark clips through a deployed Ghostline back end.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="the back end spends model money")
    mode.add_argument("--replay", action="store_true", help="the back end replays recorded calls")
    mode.add_argument("--report-only", action="store_true", help="re-score saved results, no API")
    mode.add_argument("--sync-truth", action="store_true", help="PUT truths, then re-score")
    parser.add_argument("--yes", action="store_true", help="don't ask before spending (--live)")
    parser.add_argument("--only", help="comma-separated clip ids")
    parser.add_argument("--force", action="store_true", help="redo clips that have results")
    parser.add_argument("--retry-failed", action="store_true", help="retry clips that failed")
    parser.add_argument("--publish", action="store_true", help="make the experiments public")
    parser.add_argument("--no-upload-truth", action="store_true", help="skip PUT /benchmark")
    parser.add_argument("--max-cost", type=float, help="stop once model spend reaches this (USD)")
    parser.add_argument("--workers", type=int, default=1, help="clips in flight at once")
    parser.add_argument("--poll-s", type=float, default=1.0, help="seconds between job polls")
    parser.add_argument("--manifest", type=Path, default=BENCH_DIR / "manifest.csv")
    parser.add_argument("--clips-dir", type=Path, default=BENCH_DIR / "clips")
    parser.add_argument("--state-dir", type=Path, default=BENCH_DIR / "state")
    parser.add_argument("--out-dir", type=Path, default=BENCH_DIR)
    parser.add_argument("--latitude", type=float, help="default for g clips (BENCH_LATITUDE)")
    parser.add_argument("--altitude", type=float, help="metres (BENCH_ALTITUDE_M)")
    args = parser.parse_args(argv)
    if not (args.live or args.replay or args.report_only or args.sync_truth):
        parser.error("choose --live, --replay, --report-only or --sync-truth")
    return args


def _env_float(environ: dict[str, str], name: str) -> float | None:
    raw = environ.get(name, "").strip()
    return float(raw) if raw else None


def main(
    argv: list[str] | None = None,
    *,
    http: httpx.Client | None = None,
    environ: dict[str, str] | None = None,
    confirm: Callable[[str], bool] | None = None,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """`http`, `environ`, `confirm` and `sleep` are for tests; a real run uses the real ones."""
    args = parse_args(argv)
    env = dict(os.environ if environ is None else environ)
    cfg = RunConfig(
        mode="live" if args.live else "replay",
        manifest=args.manifest,
        clips_dir=args.clips_dir,
        state_dir=args.state_dir,
        out_dir=args.out_dir,
        only=[c.strip() for c in args.only.split(",") if c.strip()] if args.only else None,
        force=args.force,
        retry_failed=args.retry_failed,
        publish=args.publish,
        upload_truth=not args.no_upload_truth,
        max_cost_usd=args.max_cost,
        latitude=args.latitude if args.latitude is not None else _env_float(env, "BENCH_LATITUDE"),
        altitude_m=args.altitude
        if args.altitude is not None
        else _env_float(env, "BENCH_ALTITUDE_M"),
        workers=max(1, args.workers),
        poll_s=args.poll_s,
    )
    rows = load_manifest(cfg.manifest)
    offline = args.report_only
    client: GhostlineClient | None = None
    backend: dict[str, Any] = {}
    if not offline:
        base, passcode = settings_from_env(env)
        client = GhostlineClient(base, passcode, http=http, sleep=sleep)
        backend = client.health()
    bench = Benchmark(cfg, client, rows, log=log)
    exit_code = EXIT_OK

    if args.live or args.replay:
        llm_mode = backend.get("llm_mode")
        if args.replay and llm_mode != "replay":
            log(
                f"Refusing: the back end is in {llm_mode!r} mode, so this run would spend model "
                "money. Set LLM_MODE=replay (docs/runbook.md) or pass --live."
            )
            return EXIT_REFUSED
        if args.live and llm_mode == "replay":
            log("Refusing: the back end is in replay mode, so a --live run would only replay.")
            return EXIT_REFUSED
        if args.live and not args.yes:
            todo = [r for r in bench.selected_rows() if bench.store.load(r.clip_id).stage != "done"]
            message = (
                f"{len(todo)} clip(s) at about ${COST_PER_CLIP_USD:.2f} is roughly "
                f"${len(todo) * COST_PER_CLIP_USD:.0f} of Meta Model API spend. Continue?"
            )
            ask = confirm or (lambda m: sys.stdin.isatty() and input(f"{m} [y/N] ").lower() == "y")
            if not ask(message):
                log("Not confirmed; nothing was run. Pass --yes to skip this question.")
                return EXIT_REFUSED
        bench.run()
        if bench.stop_reason:
            log(f"Stopped early: {bench.stop_reason}. Run again to continue from here.")
            exit_code = EXIT_STOPPED
        if cfg.upload_truth:
            bench.sync_truth()
    elif args.sync_truth:
        log(f"Uploaded {bench.sync_truth()} ground truth row(s).")

    results = bench.results()
    summary: Summary = summarize(results)
    meta = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "mode": backend.get("llm_mode") or "saved results",
        "api_version": backend.get("version"),
        "prompt_version": prompt_version().version,
    }
    paths = write_all(results, summary, cfg.out_dir, meta)
    log(
        f"{summary.n_complete} of {summary.n_run} run clips complete; "
        f"coverage {summary.n_covered}/{summary.n_scored}; wrote {paths['md']}"
    )
    unexpected = [r for r in results if r.ran and r.status != "complete" and not r.failure_flagged]
    if exit_code == EXIT_OK and unexpected:
        log(f"{len(unexpected)} clip(s) failed: {', '.join(r.clip_id for r in unexpected)}")
        exit_code = EXIT_ERROR
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
