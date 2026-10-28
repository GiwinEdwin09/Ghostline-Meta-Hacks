import threading

import numpy as np
import psycopg
import pytest
from pydantic import ValidationError

from ghostline.notebook import NewEvent, NotebookWriter
from ghostline.notebook.writer import MAX_DATA_BYTES
from ghostline.store.records import Attachment, Ref


# --- validation (no database) ---------------------------------------------------------------
def test_title_is_trimmed_and_required():
    assert NewEvent(actor="system", kind="warning", title="  hello  ").title == "hello"

    with pytest.raises(ValidationError):
        NewEvent(actor="system", kind="warning", title="   ")
    with pytest.raises(ValidationError):
        NewEvent(actor="system", kind="warning", title="")


def test_actor_and_kind_must_come_from_the_plan_lists():
    with pytest.raises(ValidationError):
        NewEvent(actor="the_model", kind="warning", title="x")
    with pytest.raises(ValidationError):
        NewEvent(actor="system", kind="gossip", title="x")


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "../other/plot.png", "fits/../../x.png", "a\\b.png"]
)
def test_attachments_must_stay_inside_the_experiment(path):
    with pytest.raises(ValidationError, match="inside the experiment"):
        NewEvent(actor="harness", kind="fit", title="x", attachments=[Attachment(path=path)])


def test_oversized_data_is_rejected():
    big = {"blob": "x" * (MAX_DATA_BYTES + 1)}

    with pytest.raises(ValidationError, match="byte limit"):
        NewEvent(actor="harness", kind="fit", title="x", data=big)


# --- writing (database) ---------------------------------------------------------------------
def test_events_get_gapless_increasing_seq(repo):
    exp = repo.create_experiment(title="t")
    writer = NotebookWriter(repo, exp.id)

    events = [writer.append(actor="system", kind="observation", title=f"e{i}") for i in range(4)]

    assert [e.seq for e in events] == [1, 2, 3, 4]
    assert repo.last_event_seq(exp.id) == 4


def test_seq_is_per_experiment(repo):
    a, b = repo.create_experiment(title="a"), repo.create_experiment(title="b")

    NotebookWriter(repo, a.id).append(actor="system", kind="observation", title="a1")
    NotebookWriter(repo, a.id).append(actor="system", kind="observation", title="a2")
    first_b = NotebookWriter(repo, b.id).append(actor="system", kind="observation", title="b1")

    assert first_b.seq == 1


def test_the_whole_row_round_trips(repo):
    exp = repo.create_experiment(title="t")
    writer = NotebookWriter(repo, exp.id)

    event = writer.append(
        actor="harness",
        kind="rejection",
        title="Rejected P0",
        body_md="Held-out error is $5\\times$ larger.",
        phase="fitting",
        data={"heldout_rmse": 0.011, "array": np.array([1.0, np.nan])},
        attachments=[Attachment(path="fits/abc/fit.png", caption="fit")],
        refs=[Ref(kind="model", id="P0")],
    )

    [loaded] = repo.list_events(exp.id)
    assert loaded == event
    assert loaded.body_md == "Held-out error is $5\\times$ larger."
    assert loaded.data == {"heldout_rmse": 0.011, "array": [1.0, None]}  # NaN becomes null
    assert loaded.attachments[0].path == "fits/abc/fit.png"
    assert loaded.refs[0].id == "P0"
    assert loaded.phase == "fitting"


def test_phase_defaults_to_the_current_phase(repo):
    exp = repo.create_experiment(title="t")
    current = {"phase": "tracking"}
    writer = NotebookWriter(repo, exp.id, current_phase=lambda: current["phase"])

    inherited = writer.append(actor="sam", kind="observation", title="a")
    explicit = writer.append(actor="sam", kind="observation", title="b", phase="calibrating")
    current["phase"] = None
    none = writer.append(actor="sam", kind="observation", title="c")

    assert (inherited.phase, explicit.phase, none.phase) == ("tracking", "calibrating", None)


def test_warning_helper(repo):
    exp = repo.create_experiment(title="t")

    event = NotebookWriter(repo, exp.id).warning("Clip looks re-timed", "Check the fps.")

    assert (event.actor, event.kind, event.body_md) == ("system", "warning", "Check the fps.")


def test_writing_to_a_missing_experiment_fails(repo):
    from uuid import uuid4

    with pytest.raises(LookupError):
        NotebookWriter(repo, uuid4()).append(actor="system", kind="warning", title="x")


def test_events_are_append_only(repo, db):
    exp = repo.create_experiment(title="t")
    NotebookWriter(repo, exp.id).append(actor="system", kind="observation", title="first")

    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        db.execute("update events set title = 'edited' where experiment_id = %s", (exp.id,))

    assert repo.list_events(exp.id)[0].title == "first"
    assert not hasattr(NotebookWriter, "update") and not hasattr(NotebookWriter, "delete")


def test_polling_with_after_returns_only_newer_events(repo):
    exp = repo.create_experiment(title="t")
    writer = NotebookWriter(repo, exp.id)
    for i in range(5):
        writer.append(actor="system", kind="observation", title=f"e{i}")

    assert [e.seq for e in repo.list_events(exp.id, after=3)] == [4, 5]
    assert repo.list_events(exp.id, after=5) == []
    assert [e.seq for e in repo.list_events(exp.id, limit=2)] == [1, 2]


def test_concurrent_writers_never_clash_or_leave_gaps(repo):
    exp = repo.create_experiment(title="t")
    errors = []

    def write(i):
        try:
            NotebookWriter(repo, exp.id).append(actor="system", kind="observation", title=f"e{i}")
        except Exception as exc:  # pragma: no cover - the assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert [e.seq for e in repo.list_events(exp.id)] == list(range(1, 13))
