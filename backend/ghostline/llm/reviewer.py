"""The Muse Spark Open client (build plan Parts 14 and 23): Chat Completions on a Modal Shared
Endpoint, with the routing-affinity header so a conversation's prompt prefix stays cached in
one container across review rounds.

    reviewer = ReviewerClient(runtime)
    result = reviewer.chat(messages, experiment_id="exp_1", schema=Review)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from .errors import StructuredOutputError
from .pricing import SPARK
from .runtime import CallSummary, Runtime
from .structured import parse_reply, response_format_for, validation_message

log = logging.getLogger("ghostline.llm")

AFFINITY_HEADER = "Modal-Routing-Affinity-Key"


class ChatResult(BaseModel):
    """A Chat Completions reply, reduced to plain data."""

    model: str
    content: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    request_hash: str = ""
    repairs: int = 0
    finish_reason: str | None = None
    parsed: Any = None

    model_config = {"arbitrary_types_allowed": True}


class ReviewerClient:
    """Spark Open on Modal. `REVIEWER_BASE_URL` is the endpoint, `REVIEWER_MODEL` the model."""

    def __init__(self, runtime: Runtime | None = None, *, sdk: Any = None) -> None:
        self.runtime = runtime or Runtime.create()
        self._sdk = sdk

    @property
    def settings(self):
        return self.runtime.settings

    @property
    def sdk(self):
        if self._sdk is None:
            import httpx2
            from openai import OpenAI

            base_url = self.settings.reviewer_base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            self._sdk = OpenAI(
                base_url=base_url,
                api_key=self.settings.reviewer_api_key,
                max_retries=0,
                timeout=httpx2.Timeout(300.0, connect=15.0),
            )
        return self._sdk

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        experiment_id: str,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 4000,
        temperature: float | None = None,
        phase: str | None = "review",
    ) -> ChatResult:
        """One review turn. `messages` are Chat Completions messages (images as `image_url`
        parts with data URLs). With `schema`, the endpoint enforces the JSON schema
        (`response_format`, verified in the Day 0 smoke test) and `.parsed` holds the validated
        instance, with one repair call on a validation failure. `experiment_id` becomes the
        affinity key."""
        result = self._once(messages, experiment_id, schema, max_tokens, temperature, phase)
        if schema is None:
            return result
        try:
            result.parsed = parse_reply(schema, result.content)
            return result
        except ValueError as error:
            problem = (
                validation_message(error) if isinstance(error, ValidationError) else str(error)
            )
            raw = result.content
        log.warning("reviewer reply failed validation, repairing once: %s", problem)
        repair = [
            *messages,
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": f"Your reply did not validate against the schema: {problem}\n"
                "Reply again with only the corrected JSON.",
            },
        ]
        second = self._once(repair, experiment_id, schema, max_tokens, temperature, phase)
        second.input_tokens += result.input_tokens
        second.cached_tokens += result.cached_tokens
        second.output_tokens += result.output_tokens
        second.cost_usd = round(second.cost_usd + result.cost_usd, 8)
        second.repairs = 1
        try:
            second.parsed = parse_reply(schema, second.content)
        except ValueError as error:
            raise StructuredOutputError(
                f"reviewer reply still failed validation after a repair: {error}", raw_text=raw
            ) from error
        return second

    def _once(self, messages, experiment_id, schema, max_tokens, temperature, phase) -> ChatResult:
        model = self.settings.reviewer_model
        request: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            request["temperature"] = temperature
        if schema is not None:
            request["response_format"] = response_format_for(schema)

        def live() -> ChatResult:
            response = self.sdk.chat.completions.create(
                extra_headers={AFFINITY_HEADER: experiment_id}, **request
            )
            usage = response.usage
            details = getattr(usage, "prompt_tokens_details", None)
            cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
            choice = response.choices[0]
            prompt_tokens = usage.prompt_tokens or 0
            completion_tokens = usage.completion_tokens or 0
            return ChatResult(
                model=response.model or model,
                content=choice.message.content or "",
                input_tokens=prompt_tokens,
                cached_tokens=cached,
                output_tokens=completion_tokens,
                cost_usd=SPARK.cost(prompt_tokens, cached, completion_tokens),
                finish_reason=choice.finish_reason,
            )

        def summarize(result: ChatResult) -> CallSummary:
            return CallSummary(
                result.input_tokens,
                result.cached_tokens,
                result.output_tokens,
                cost_usd=result.cost_usd,
            )

        result, key, hit = self.runtime.run(
            "chat",
            request,
            live=live,
            encode=lambda r: {
                "response": r.model_dump(exclude={"parsed", "cache_hit", "request_hash"})
            },
            decode=lambda entry: ChatResult.model_validate(entry["response"]),
            summarize=summarize,
            model=model,
            provider="modal",
            experiment_id=experiment_id,
            phase=phase,
        )
        result.request_hash, result.cache_hit = key, hit
        return result
