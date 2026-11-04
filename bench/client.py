"""A small client for the Ghostline API (build plan Part 21), for the benchmark and the evals.

It speaks only the public routes: it does not import the back end's routes or stores, so it also
works against a deployment whose code differs from this checkout. The base URL and the passcode
come from the environment, never from code:

    GHOSTLINE_API_BASE   https://<workspace>--ghostline-api.modal.run   (with or without /api)
    GHOSTLINE_PASSCODE   the value of APP_PASSCODE in the Modal secret

Transient failures (429 from the platform, 5xx, a dropped connection) are retried with
exponential backoff and jitter, honouring Retry-After. A quota the API enforces on purpose (the
daily budget, ten analyses per IP per day) is not transient, so it raises QuotaExceeded at once.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

PASSCODE_HEADER = "X-Ghostline-Key"
ENV_BASE = "GHOSTLINE_API_BASE"
ENV_PASSCODE = "GHOSTLINE_PASSCODE"

# 429s that mean "come back tomorrow" or "stop guessing", not "try again in a second".
QUOTA_CODES = {"rate_limited", "daily_budget_reached", "too_many_attempts", "too_many_uploads"}
RETRY_STATUSES = {429, 500, 502, 503, 504}


class ApiFailure(Exception):
    """The API answered with an error status."""

    def __init__(self, status: int, code: str | None, detail: str, where: str):
        super().__init__(f"{where} -> {status} {code or ''}: {detail}".replace("  ", " "))
        self.status = status
        self.code = code
        self.detail = detail
        self.where = where


class QuotaExceeded(ApiFailure):
    """A limit that waiting a few seconds won't lift. Stop and say what to do."""


class JobFailed(Exception):
    def __init__(self, job: Mapping[str, Any]):
        super().__init__(job.get("error") or "the job failed")
        self.job = dict(job)
        self.retryable = bool(job.get("retryable", True))


class WaitTimeout(Exception):
    pass


def api_root(base_url: str) -> str:
    """The URL that routes hang off: .../api, whether or not the caller included it."""
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError(f"set {ENV_BASE} to the API's URL")
    return base if base.endswith("/api") else base + "/api"


def settings_from_env(environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    env = os.environ if environ is None else environ
    base, passcode = env.get(ENV_BASE, ""), env.get(ENV_PASSCODE, "")
    missing = [name for name, value in ((ENV_BASE, base), (ENV_PASSCODE, passcode)) if not value]
    if missing:
        raise SystemExit(
            f"set {' and '.join(missing)} (the passcode is APP_PASSCODE in the secret)"
        )
    return base, passcode


class GhostlineClient:
    def __init__(
        self,
        base_url: str,
        passcode: str,
        *,
        http: httpx.Client | None = None,
        retries: int = 4,
        backoff_s: float = 0.5,
        timeout_s: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.root = api_root(base_url)
        self._passcode = passcode
        self._http = http or httpx.Client(timeout=timeout_s)
        self._retries = retries
        self._backoff = backoff_s
        self._sleep = sleep
        self._clock = clock

    def __repr__(self) -> str:  # never show the passcode
        return f"GhostlineClient({self.root!r})"

    def close(self) -> None:
        self._http.close()

    # --- plumbing ---------------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        content: bytes | None = None,
        params: Mapping[str, Any] | None = None,
        auth: bool = True,
        replay_safe: bool = True,
    ) -> Any:
        """`replay_safe=False` for a request that must not be sent twice if the first one may have
        reached the server (creating an experiment): only failures before it was processed are
        retried."""
        url = f"{self.root}{path}"
        headers = {PASSCODE_HEADER: self._passcode} if auth else {}
        if content is not None:
            headers["Content-Type"] = "application/octet-stream"
        where = f"{method} {path}"
        for attempt in range(self._retries + 1):
            try:
                response = self._http.request(
                    method, url, json=json, content=content, params=params, headers=headers
                )
            except httpx.TransportError as exc:
                connect_failed = isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout)
                if attempt == self._retries or not (replay_safe or connect_failed):
                    raise
                self._sleep(self._delay(attempt, None))
                continue
            if response.is_success:
                return response.json() if response.content else None
            code, detail = _error_body(response)
            if code in QUOTA_CODES:
                raise QuotaExceeded(response.status_code, code, detail, where)
            retryable = response.status_code in RETRY_STATUSES and (
                replay_safe or response.status_code in (429, 503)
            )
            if not retryable or attempt == self._retries:
                raise ApiFailure(response.status_code, code, detail, where)
            self._sleep(self._delay(attempt, response.headers.get("retry-after")))
        raise AssertionError("unreachable")

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        delay = self._backoff * 2**attempt * (0.5 + random.random())
        if retry_after:
            try:
                delay = max(delay, min(float(retry_after), 120.0))
            except ValueError:
                pass
        return delay

    # --- routes -------------------------------------------------------------------------------
    def health(self) -> dict:
        return self._request("GET", "/health", auth=False)

    def create_experiment(
        self,
        *,
        title: str,
        system_hint: str = "auto",
        description: str = "",
        declared_capture: Mapping[str, Any] | None = None,
        setup: Mapping[str, Any] | None = None,
    ) -> dict:
        body: dict[str, Any] = {"title": title, "system_hint": system_hint}
        body["description"] = description[:2000]
        if declared_capture:
            body["declared_capture"] = dict(declared_capture)
        if setup:
            body["setup"] = dict(setup)
        return self._request("POST", "/experiments", json=body, replay_safe=False)

    def get_experiment(self, experiment_id: str) -> dict:
        return self._request("GET", f"/experiments/{experiment_id}")

    def patch_experiment(self, experiment_id: str, **fields: Any) -> dict:
        return self._request("PATCH", f"/experiments/{experiment_id}", json=fields)

    def events(self, experiment_id: str, after: int = 0, limit: int = 500) -> dict:
        return self._request(
            "GET", f"/experiments/{experiment_id}/events", params={"after": after, "limit": limit}
        )

    def all_events(self, experiment_id: str) -> list[dict]:
        events: list[dict] = []
        after = 0
        while True:
            page = self.events(experiment_id, after)
            events += page["events"]
            if len(page["events"]) < 500:
                return events
            after = page["last_seq"]

    def start_upload(self, experiment_id: str, filename: str, size: int, role: str = "fit") -> dict:
        body = {"experiment_id": experiment_id, "role": role, "filename": filename, "size": size}
        return self._request("POST", "/uploads", json=body)

    def get_upload(self, upload_id: str) -> dict:
        return self._request("GET", f"/uploads/{upload_id}", auth=False)

    def put_chunk(self, upload_id: str, n: int, data: bytes) -> dict:
        return self._request("PUT", f"/uploads/{upload_id}/chunks/{n}", content=data)

    def complete_upload(self, upload_id: str) -> dict:
        return self._request("POST", f"/uploads/{upload_id}/complete")

    def upload_file(
        self,
        experiment_id: str,
        path: Path | str,
        *,
        role: str = "fit",
        upload_id: str | None = None,
        on_upload: Callable[[str], None] | None = None,
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> dict:
        """Chunked upload, resumable: pass the upload_id of an earlier attempt and only the
        chunks the server doesn't have are sent. Returns the /complete response.
        `on_upload(upload_id)` is called once the id is known, so a caller can save it."""
        path = Path(path)
        size = path.stat().st_size
        info = self.get_upload(upload_id) if upload_id else None
        if info is None or info["status"] == "open" and info["size"] != size:
            info = self.start_upload(experiment_id, path.name, size, role)
        if on_upload:
            on_upload(info["upload_id"])
        if info["status"] != "complete":
            done = set(info["received_chunks"])
            chunk = info["chunk_size"]
            with path.open("rb") as handle:
                for n in range(info["n_chunks"]):
                    if n in done:
                        continue
                    handle.seek(n * chunk)
                    self.put_chunk(info["upload_id"], n, handle.read(chunk))
                    if on_chunk:
                        on_chunk(n + 1, info["n_chunks"])
        return self.complete_upload(info["upload_id"])

    def set_trim(
        self, trial_id: str, start_s: float | None = None, end_s: float | None = None
    ) -> dict:
        body = {k: v for k, v in (("start_s", start_s), ("end_s", end_s)) if v is not None}
        return self._request("POST", f"/trials/{trial_id}/trim", json=body)

    def analyze(self, experiment_id: str, trial_id: str | None = None) -> dict:
        body = {"trial_id": trial_id} if trial_id else None
        return self._request("POST", f"/experiments/{experiment_id}/analyze", json=body)

    def get_job(self, job_id: str) -> dict:
        return self._request("GET", f"/jobs/{job_id}", auth=False)

    def retry_job(self, job_id: str) -> dict:
        return self._request("POST", f"/jobs/{job_id}/retry")

    def put_benchmark(self, experiment_id: str, entry: Mapping[str, Any]) -> dict:
        return self._request("PUT", f"/benchmark/{experiment_id}", json=dict(entry))

    def get_benchmark(self) -> dict:
        return self._request("GET", "/benchmark", auth=False)

    # --- waiting ------------------------------------------------------------------------------
    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_s: float,
        poll_s: float = 1.0,
        on_poll: Callable[[dict], None] | None = None,
    ) -> dict:
        """Poll until the job succeeds (returns it) or fails (raises JobFailed). The API marks a
        job failed itself when its worker stops sending heartbeats, so this only needs a
        timeout as a backstop."""
        deadline = self._clock() + timeout_s
        while True:
            job = self.get_job(job_id)
            if on_poll:
                on_poll(job)
            if job["status"] == "succeeded":
                return job
            if job["status"] == "failed":
                raise JobFailed(job)
            if self._clock() >= deadline:
                where = f"{job['status']} ({job.get('phase')})"
                raise WaitTimeout(f"job {job_id} still {where} after {timeout_s:.0f} s")
            self._sleep(poll_s)


def _error_body(response: httpx.Response) -> tuple[str | None, str]:
    try:
        body = response.json()
    except ValueError:
        return None, response.text[:200] or response.reason_phrase
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, list):  # FastAPI's own validation errors
            detail = "; ".join(str(d.get("msg", d)) for d in detail)[:300]
        return body.get("code"), str(detail or response.reason_phrase)
    return None, str(body)[:200]
