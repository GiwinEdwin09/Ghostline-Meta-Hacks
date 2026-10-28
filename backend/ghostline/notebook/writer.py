"""The lab notebook's event writer (build plan Part 20).

The orchestrator writes events, never the model: numbers in `data` come straight from the harness,
and a model's narrative is attached as `body_md`. Events are append-only. This class has no update
or delete, and the database rejects UPDATE on the events table.
"""

from collections.abc import Callable
from typing import Any
from uuid import UUID

import orjson
from pydantic import BaseModel, Field, field_validator

from ghostline.store.records import Actor, Attachment, Event, EventKind, Ref
from ghostline.store.repo import Repo

MAX_TITLE = 300
MAX_BODY = 50_000
MAX_DATA_BYTES = 1_000_000


class NewEvent(BaseModel):
    """A validated event, before it gets an id and a sequence number."""

    actor: Actor
    kind: EventKind
    title: str = Field(min_length=1, max_length=MAX_TITLE)
    body_md: str = Field(default="", max_length=MAX_BODY)
    phase: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    attachments: list[Attachment] = Field(default_factory=list)
    refs: list[Ref] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be blank")
        return value

    @field_validator("data")
    @classmethod
    def _data_small(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            size = len(orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY))
        except TypeError as exc:
            raise ValueError(f"data is not JSON serializable: {exc}") from exc
        if size > MAX_DATA_BYTES:
            raise ValueError(f"data is {size} bytes, over the {MAX_DATA_BYTES} byte limit")
        return value

    @field_validator("attachments")
    @classmethod
    def _attachments_relative(cls, value: list[Attachment]) -> list[Attachment]:
        for item in value:
            parts = item.path.split("/")
            if item.path.startswith("/") or "\\" in item.path or ".." in parts:
                raise ValueError(f"attachment path must stay inside the experiment: {item.path!r}")
        return value


class NotebookWriter:
    """Appends events for one experiment. `current_phase` lets a job stamp each event with the
    phase it is in without every call site repeating it."""

    def __init__(
        self,
        repo: Repo,
        experiment_id: UUID,
        *,
        current_phase: Callable[[], str | None] | None = None,
    ):
        self._repo = repo
        self.experiment_id = experiment_id
        self._current_phase = current_phase

    def append(
        self,
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
        if phase is None and self._current_phase is not None:
            phase = self._current_phase()
        event = NewEvent(
            actor=actor,
            kind=kind,
            title=title,
            body_md=body_md,
            phase=phase,
            data=data or {},
            attachments=attachments or [],
            refs=refs or [],
        )
        return self._repo.append_event(
            self.experiment_id,
            actor=event.actor,
            kind=event.kind,
            title=event.title,
            body_md=event.body_md,
            phase=event.phase,
            data=event.data,
            attachments=event.attachments,
            refs=event.refs,
        )

    def warning(self, title: str, body_md: str = "", **kwargs: Any) -> Event:
        """A system warning, for problems the user should see in the notebook."""
        return self.append(actor="system", kind="warning", title=title, body_md=body_md, **kwargs)
