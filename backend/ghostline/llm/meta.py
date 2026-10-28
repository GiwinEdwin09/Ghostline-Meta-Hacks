"""The Meta Model API client (build plan Part 23): Files, Responses (Wonder) and SAM 3.1.

    runtime = Runtime.create()
    meta = MetaClient(runtime)
    reply = meta.respond(input="...", text_format=SceneAnalysis, reasoning_effort="medium")

Every call goes through `Runtime.run`, so it is retried, metered, budget-checked and, by
`LLM_MODE`, recorded or replayed. The client is synchronous; run it in a worker thread from
async code.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic import Field as PField

from .cache import sha256_file
from .errors import PermanentLLMError, StructuredOutputError
from .items import as_items, input_image, input_video, user_message
from .pricing import price_for_model, sam_image_cost, sam_video_cost
from .retry import RetryableError, call_with_retry
from .runtime import CallSummary, Runtime
from .structured import parse_reply, text_format_for, validation_message

log = logging.getLogger("ghostline.llm")

# Stream errors that are the server's fault and worth another attempt.
_RETRYABLE_CODES = {
    "server_error",
    "service_overloaded",
    "backend_unavailable",
    "server_shutting_down",
    "rate_limit_exceeded",
}
_TERMINAL = {"response.completed", "response.incomplete"}
FILE_EXPIRY_S = 7 * 24 * 3600


class Usage(BaseModel):
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0  # includes reasoning tokens
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class FunctionCall(BaseModel):
    call_id: str
    name: str
    arguments: str  # the raw JSON text, as the model wrote it

    def args(self) -> dict[str, Any]:
        """The arguments as a dict. Raises ValueError when they aren't a JSON object."""
        value = json.loads(self.arguments or "{}")
        if not isinstance(value, dict):
            raise ValueError("tool arguments must be a JSON object")
        return value


class ModelResponse(BaseModel):
    """A Responses API reply, reduced to plain data (what is recorded and replayed).

    `output` holds the output items (reasoning, messages, function calls) as dicts: pass them
    back in `input`, followed by any `function_call_output` items, to continue the
    conversation without `previous_response_id`. `cost_usd` is what the usage is worth at list
    price, also on a cache hit, where nothing is actually charged.
    """

    id: str | None = None
    model: str
    status: str = "completed"
    incomplete_reason: str | None = None
    output: list[dict[str, Any]] = PField(default_factory=list)
    output_text: str = ""
    usage: Usage = PField(default_factory=Usage)
    cost_usd: float = 0.0
    cache_hit: bool = False
    request_hash: str = ""
    repairs: int = 0
    parsed: Any = PField(default=None, exclude=True)

    @property
    def function_calls(self) -> list[FunctionCall]:
        return [
            FunctionCall(**{k: item[k] for k in ("call_id", "name", "arguments")})
            for item in self.output
            if item.get("type") == "function_call"
        ]

    @property
    def truncated(self) -> bool:
        return self.status == "incomplete"


@dataclass(frozen=True)
class UploadedFile:
    id: str
    sha256: str
    size_bytes: int
    filename: str


@dataclass
class SamResult:
    """A SAM response: the raw event stream and the output text the parser reads."""

    events: list[dict[str, Any]]
    output_text: str
    usage: dict[str, Any]
    frames_processed: int
    cost_usd: float
    cache_hit: bool = False
    request_hash: str = ""
    kind: str = "sam_video"
    extras: dict[str, Any] = field(default_factory=dict)

    def write_events_jsonl(self, path: str | Path) -> Path:
        """Write the raw event stream, one event per line, for the browser to replay."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")
        return path


def usage_from(raw: dict[str, Any] | None) -> Usage:
    raw = raw or {}
    return Usage(
        input_tokens=raw.get("input_tokens", 0) or 0,
        cached_tokens=(raw.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0,
        output_tokens=raw.get("output_tokens", 0) or 0,
        reasoning_tokens=(raw.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
    )


def _output_text(output: list[dict[str, Any]]) -> str:
    parts = []
    for item in output:
        if item.get("type") == "message":
            parts.extend(
                part.get("text", "")
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
    return "".join(parts)


def _failure(error: Any) -> Exception:
    """Map a streamed or non-streamed failure to a retryable or permanent error."""
    error = error if isinstance(error, dict) else getattr(error, "model_dump", lambda: {})()
    code, message = error.get("code"), error.get("message") or "model call failed"
    if code in _RETRYABLE_CODES:
        return RetryableError(f"{code}: {message}")
    return PermanentLLMError(f"{code}: {message}", code=code)


class MetaClient:
    """Wonder (Responses API), SAM 3.1 and the Files API on `META_BASE_URL`."""

    def __init__(self, runtime: Runtime | None = None, *, sdk: Any = None) -> None:
        self.runtime = runtime or Runtime.create()
        self._sdk = sdk
        self.rate_limits: dict[str, str] = {}

    @property
    def settings(self):
        return self.runtime.settings

    @property
    def sdk(self):
        """The OpenAI SDK client, created on first use so replay mode needs no key."""
        if self._sdk is None:
            import httpx2
            from openai import OpenAI

            self._sdk = OpenAI(
                base_url=self.settings.meta_base_url,
                api_key=self.settings.model_api_key,
                max_retries=0,  # retries are ours
                timeout=httpx2.Timeout(600.0, connect=15.0),
                http_client=httpx2.Client(
                    event_hooks={"response": [self._note_rate_limits]},
                    timeout=httpx2.Timeout(600.0, connect=15.0),
                ),
            )
        return self._sdk

    def _note_rate_limits(self, response: Any) -> None:
        found = {k: v for k, v in response.headers.items() if k.lower().startswith("x-ratelimit-")}
        if found:
            self.rate_limits.update(found)
            log.debug("rate limits: %s", found)

    # Files API

    def upload_file(self, path: str | Path, *, expires_s: int = FILE_EXPIRY_S) -> UploadedFile:
        """Upload with `purpose="user_data"` and a 7-day expiry, and remember the file's hash so
        request keys don't depend on the ID. In replay mode nothing is uploaded: the ID is
        derived from the content."""
        path = Path(path)
        digest = sha256_file(path)
        if self.runtime.mode == "replay":
            uploaded = UploadedFile(
                f"file-replay-{digest[:24]}", digest, path.stat().st_size, path.name
            )
        else:

            def attempt():
                with path.open("rb") as handle:
                    return self.sdk.files.create(
                        file=(path.name, handle),
                        purpose="user_data",
                        expires_after={"anchor": "created_at", "seconds": expires_s},
                    )

            created = call_with_retry(
                attempt,
                attempts=self.runtime.attempts,
                sleep=self.runtime.sleep,
                rng=self.runtime.rng,
            )
            uploaded = UploadedFile(created.id, digest, path.stat().st_size, path.name)
        self.runtime.file_hashes[uploaded.id] = digest
        return uploaded

    def delete_file(self, file_id: str) -> None:
        """Delete an uploaded file (experiment deletion, Part 22). A no-op in replay mode."""
        if self.runtime.mode != "replay":
            self.sdk.files.delete(file_id)

    # Responses API (Muse Wonder)

    def respond(
        self,
        *,
        input: str | list[dict[str, Any]],
        model: str | None = None,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        text_format: type[BaseModel] | dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int = 8000,
        stream: bool = True,
        on_text: Callable[[str], None] | None = None,
        on_event: Callable[[Any], None] | None = None,
        experiment_id: str | None = None,
        phase: str | None = None,
    ) -> ModelResponse:
        """One Responses API call.

        - `input`: a string, or a list of items (messages, and for a continuation the previous
          `ModelResponse.output` plus `function_call_output` items).
        - `text_format`: a Pydantic model class (strict JSON schema; `.parsed` holds the
          validated instance, with one repair call on a validation failure) or a raw
          `{"type": "json_schema", ...}` dict.
        - `reasoning_effort`: minimal, low, medium, high, xhigh or max. Reasoning tokens share
          `max_output_tokens`, so leave it generous.
        - `stream`: stream the reply (always, in the pipeline). `on_text` gets each text delta
          and `on_event` every event; a retried attempt starts the stream over.
        """
        items = as_items(input)
        pydantic_format = isinstance(text_format, type) and issubclass(text_format, BaseModel)

        def ask(messages: list[dict[str, Any]]) -> ModelResponse:
            return self._respond_once(
                messages,
                model=model,
                instructions=instructions,
                tools=tools,
                text_format=text_format,
                effort=reasoning_effort,
                max_tokens=max_output_tokens,
                stream=stream,
                on_text=on_text,
                on_event=on_event,
                experiment_id=experiment_id,
                phase=phase,
            )

        response = ask(items)
        if not pydantic_format:
            return response
        try:
            response.parsed = parse_reply(text_format, response.output_text)
            return response
        except ValueError as error:  # JSONDecodeError and pydantic's ValidationError
            problem = (
                validation_message(error) if isinstance(error, ValidationError) else str(error)
            )
            raw = response.output_text
        log.warning("structured reply failed validation, repairing once: %s", problem)
        repair = user_message(
            f"Your reply did not validate against the schema: {problem}\n"
            "Reply again with only the corrected JSON."
        )
        second = ask([*items, *response.output, repair])
        second.usage = response.usage + second.usage
        second.cost_usd = round(response.cost_usd + second.cost_usd, 8)
        second.repairs = 1
        try:
            second.parsed = parse_reply(text_format, second.output_text)
        except ValueError as error:
            raise StructuredOutputError(
                f"reply still failed validation after a repair: {error}", raw_text=raw
            ) from error
        return second

    def _respond_once(
        self,
        items: list[dict[str, Any]],
        *,
        model: str | None,
        instructions: str | None,
        tools: list[dict[str, Any]] | None,
        text_format: type[BaseModel] | dict[str, Any] | None,
        effort: str | None,
        max_tokens: int,
        stream: bool,
        on_text: Callable[[str], None] | None,
        on_event: Callable[[Any], None] | None,
        experiment_id: str | None,
        phase: str | None,
    ) -> ModelResponse:
        model = model or self.settings.physicist_model
        if isinstance(text_format, type) and issubclass(text_format, BaseModel):
            text = {"format": text_format_for(text_format)}
        elif text_format is not None:
            text = {"format": text_format}
        else:
            text = None
        request: dict[str, Any] = {"model": model, "input": items, "max_output_tokens": max_tokens}
        for key, value in (
            ("instructions", instructions),
            ("tools", tools),
            ("text", text),
            ("reasoning", {"effort": effort} if effort else None),
        ):
            if value:
                request[key] = value

        def live() -> dict[str, Any]:
            return self._live_response(request, stream, on_text, on_event)

        def decode(entry: dict[str, Any]) -> ModelResponse:
            return ModelResponse.model_validate(entry["response"])

        def finish(raw: dict[str, Any]) -> ModelResponse:
            output = raw.get("output") or []
            usage = usage_from(raw.get("usage"))
            incomplete = (raw.get("incomplete_details") or {}).get("reason")
            return ModelResponse(
                id=raw.get("id"),
                model=raw.get("model") or model,
                status=raw.get("status") or "completed",
                incomplete_reason=incomplete,
                output=output,
                output_text=_output_text(output),
                usage=usage,
                cost_usd=price_for_model(model).cost(
                    usage.input_tokens, usage.cached_tokens, usage.output_tokens
                ),
            )

        def summarize(result: ModelResponse) -> CallSummary:
            u = result.usage
            return CallSummary(
                u.input_tokens,
                u.cached_tokens,
                u.output_tokens,
                u.reasoning_tokens,
                cost_usd=result.cost_usd,
            )

        result, key, hit = self.runtime.run(
            "respond",
            request,
            live=lambda: finish(live()),
            encode=lambda r: {
                "response": r.model_dump(mode="json", exclude={"cache_hit", "request_hash"})
            },
            decode=decode,
            summarize=summarize,
            model=model,
            experiment_id=experiment_id,
            phase=phase,
        )
        result.request_hash, result.cache_hit = key, hit
        return result

    def _live_response(self, request, stream, on_text, on_event) -> dict[str, Any]:
        if not stream:
            response = self.sdk.responses.create(**request)
            raw = response.model_dump(mode="json", exclude_none=True)
            if raw.get("status") == "failed":
                raise _failure(raw.get("error"))
            return raw
        final = None
        for event in self.sdk.responses.create(stream=True, **request):
            if on_event:
                on_event(event)
            kind = event.type
            if kind == "response.output_text.delta" and on_text:
                on_text(event.delta)
            elif kind in _TERMINAL:
                final = event.response
            elif kind == "response.failed":
                raise _failure(event.response.error)
            elif kind == "error":
                raise _failure({"code": getattr(event, "code", None), "message": event.message})
        if final is None:
            raise RetryableError("the response stream ended without a terminal event")
        return final.model_dump(mode="json", exclude_none=True)

    # SAM 3.1

    def sam_video(
        self,
        file_id: str,
        prompt: str,
        *,
        metadata: dict[str, str] | None = None,
        media_sha256: str | None = None,
        frames_hint: int | None = None,
        experiment_id: str | None = None,
        phase: str | None = "track",
    ) -> SamResult:
        """Segment a video uploaded to the Files API.

        `prompt` is a noun phrase, or the text built by `meta_sam_parser.build_object_prompt`
        for box and point prompts. `metadata` adds to `include_confidence="true"`, for example
        `{"propagation_direction": "both"}` or `{"mask_encoding": "one_bit"}`. Pass the master's
        `media_sha256` so the cache key survives re-uploads. `frames_hint` (the clip's frame
        count) prices the call when the response reports no frame count.
        """
        if media_sha256:
            self.runtime.file_hashes[file_id] = media_sha256
        return self._sam(
            "sam_video", input_video(file_id), prompt, metadata, frames_hint, experiment_id, phase
        )

    def sam_image(
        self,
        image: bytes | str | Path,
        prompt: str,
        *,
        mime: str | None = None,
        metadata: dict[str, str] | None = None,
        experiment_id: str | None = None,
        phase: str | None = "sheet",
    ) -> SamResult:
        """Segment one image (bytes or a file) with a concept prompt."""
        return self._sam(
            "sam_image", input_image(image, mime), prompt, metadata, None, experiment_id, phase
        )

    def _sam(self, kind, media, prompt, metadata, frames_hint, experiment_id, phase) -> SamResult:
        model = self.settings.sam_model
        meta = {"include_confidence": "true", **(metadata or {})}
        request = {"model": model, "input": [user_message(prompt, media)], "metadata": meta}

        def live() -> SamResult:
            events, texts, usage = [], [], {}
            for event in self.sdk.responses.create(stream=True, **request):
                raw = event.model_dump(mode="json", exclude_none=True)
                events.append(raw)
                if event.type == "response.output_text.delta":
                    texts.append(event.delta)
                elif event.type in _TERMINAL:
                    usage = raw.get("response", {}).get("usage") or {}
                elif event.type == "response.failed":
                    raise _failure(event.response.error)
                elif event.type == "error":
                    raise _failure({"code": getattr(event, "code", None), "message": event.message})
            if not any(e["type"] in _TERMINAL for e in events):
                raise RetryableError("the SAM stream ended without a terminal event")
            return self._sam_result(kind, events, "".join(texts), usage, frames_hint)

        def encode(result: SamResult) -> dict[str, Any]:
            return {
                "response": {
                    "output_text": result.output_text,
                    "usage": result.usage,
                    "frames_processed": result.frames_processed,
                },
                "events": result.events,
            }

        def decode(entry: dict[str, Any]) -> SamResult:
            saved = entry["response"]
            return SamResult(
                events=entry.get("events", []),
                output_text=saved["output_text"],
                usage=saved["usage"],
                frames_processed=saved["frames_processed"],
                cost_usd=self._sam_cost(kind, saved["frames_processed"]),
                kind=kind,
            )

        def summarize(result: SamResult) -> CallSummary:
            frames = result.frames_processed if kind == "sam_video" else 0
            return CallSummary(frames=frames, cost_usd=result.cost_usd)

        result, key, hit = self.runtime.run(
            kind,
            request,
            live=live,
            encode=encode,
            decode=decode,
            summarize=summarize,
            model=model,
            experiment_id=experiment_id,
            phase=phase,
        )
        result.request_hash, result.cache_hit = key, hit
        return result

    @staticmethod
    def _sam_cost(kind: str, frames: int) -> float:
        return sam_video_cost(frames) if kind == "sam_video" else sam_image_cost()

    def _sam_result(self, kind, events, text, usage, frames_hint) -> SamResult:
        frames = usage.get("video_frames_processed") if isinstance(usage, dict) else None
        if frames is None:
            frames = frames_hint or 0
        if not text:  # fall back to the finalized text part, if no deltas were sent
            for event in events:
                part = event.get("part") or {}
                if (
                    event["type"] == "response.content_part.done"
                    and part.get("type") == "output_text"
                ):
                    text += part.get("text", "")
        return SamResult(
            events=events,
            output_text=text,
            usage=usage,
            frames_processed=int(frames),
            cost_usd=self._sam_cost(kind, int(frames)),
            kind=kind,
        )
