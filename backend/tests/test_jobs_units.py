import inspect
import json
import logging
from typing import get_args
from unittest.mock import MagicMock

import pytest

from ghostline.jobs import stages
from ghostline.jobs.context import PHASE_PLANS, PHASES, phase_span
from ghostline.jobs.errors import StageError, TransientStageError
from ghostline.jobs.log import JsonFormatter
from ghostline.jobs.retry import call_with_retries
from ghostline.jobs.spawn import FUNCTION_FOR_KIND, InlineSpawner, ModalSpawner, ThreadSpawner
from ghostline.store.records import JobKind, Phase


def test_every_job_kind_has_a_plan_and_a_modal_function():
    kinds = set(get_args(JobKind))

    assert set(PHASE_PLANS) == kinds
    assert set(FUNCTION_FOR_KIND) == kinds
    assert set(FUNCTION_FOR_KIND.values()) == {"ingest_trial", "analyze_experiment", "prove_trial"}


@pytest.mark.parametrize("kind", sorted(PHASE_PLANS))
def test_phase_weights_fill_the_progress_bar_and_use_known_phases(kind):
    plan = PHASE_PLANS[kind]

    assert sum(weight for _, weight in plan) == pytest.approx(1.0)
    assert all(name in PHASES for name, _ in plan)


def test_every_phase_is_used_by_some_plan():
    used = {name for plan in PHASE_PLANS.values() for name, _ in plan}

    assert used == set(get_args(Phase))


def test_the_analysis_stepper_follows_the_plan_order():
    names = [name for name, _ in PHASE_PLANS["analyze_experiment"]]

    assert names == [
        "watching",
        "tracking",
        "calibrating",
        "hypothesizing",
        "fitting",
        "peer_review",
    ]
    assert [n for n, _ in PHASE_PLANS["ingest_trial"]] == ["normalizing"]


def test_phase_span_gives_start_and_weight():
    start, weight = phase_span("analyze_experiment", "tracking")

    assert start == pytest.approx(0.08)
    assert weight == pytest.approx(0.27)
    assert phase_span("analyze_experiment", "rendering") == (0.0, 0.0)  # not in this plan


# --- retries --------------------------------------------------------------------------------
def test_transient_errors_are_retried_with_doubling_backoff():
    attempts, sleeps = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise TransientStageError("429")
        return "ok"

    assert call_with_retries(flaky, attempts=3, base_delay_s=0.5, sleep=sleeps.append) == "ok"
    assert sleeps == [0.5, 1.0]


def test_the_last_transient_error_propagates():
    def always():
        raise TransientStageError("503")

    with pytest.raises(TransientStageError):
        call_with_retries(always, attempts=2, sleep=lambda _: None)


def test_other_errors_are_not_retried():
    calls = []

    def broken():
        calls.append(1)
        raise StageError("bad clip", retryable=False)

    with pytest.raises(StageError):
        call_with_retries(broken, attempts=5, sleep=lambda _: None)
    assert len(calls) == 1


def test_stage_error_carries_message_code_and_retryability():
    error = StageError("Nothing to track", code="cannot_measure", retryable=False)

    assert (error.message, error.code, error.retryable) == (
        "Nothing to track",
        "cannot_measure",
        False,
    )
    assert TransientStageError("x").retryable is True


# --- spawners -------------------------------------------------------------------------------
def test_modal_spawner_picks_the_function_for_the_job_kind():
    functions = {
        name: MagicMock() for name in ("ingest_trial", "analyze_experiment", "prove_trial")
    }
    functions["prove_trial"].spawn.return_value.object_id = "fc-123"
    job = MagicMock(kind="revise", id="job-1")

    call_id = ModalSpawner(functions).spawn(job)

    functions["prove_trial"].spawn.assert_called_once_with("job-1")
    functions["analyze_experiment"].spawn.assert_not_called()
    assert call_id == "fc-123"


def test_inline_and_thread_spawners_run_the_callback():
    import threading

    seen, done = [], threading.Event()
    job = MagicMock(id="job-1")

    InlineSpawner(seen.append).spawn(job)
    assert seen == ["job-1"]

    ThreadSpawner(lambda job_id: (seen.append(job_id), done.set())).spawn(job)
    assert done.wait(2)
    assert seen == ["job-1", "job-1"]


# --- the stage stubs are the wave 2 wiring point ----------------------------------------------
STAGE_FUNCTIONS = [
    name
    for name, fn in inspect.getmembers(stages, inspect.isfunction)
    if fn.__module__ == stages.__name__
]


def test_there_is_a_stub_for_every_stage():
    assert set(STAGE_FUNCTIONS) == {
        "ingest",
        "analyze_scene",
        "track_objects",
        "calibrate",
        "extract_trajectory",
        "run_physicist",
        "run_review",
        "predict_ghost",
        "diagnose_revision",
        "rerun_model",
        "verify_export",
        "build_export",
        "simulate_custom_model",
    }


@pytest.mark.parametrize("name", STAGE_FUNCTIONS)
def test_stubs_raise_not_implemented_with_the_wave_2_message(name):
    fn = getattr(stages, name)
    args = [MagicMock() for _ in inspect.signature(fn).parameters]

    with pytest.raises(NotImplementedError, match="wired in wave 2"):
        fn(*args)


# --- logs -----------------------------------------------------------------------------------
def test_json_logs_carry_experiment_job_and_phase():
    record = logging.LogRecord("ghostline.jobs", logging.INFO, __file__, 1, "phase", None, None)
    record.experiment_id = "exp-1"
    record.job_id = "job-1"
    record.phase = "tracking"

    line = json.loads(JsonFormatter().format(record))

    assert line["message"] == "phase"
    assert line["level"] == "INFO"
    assert (line["experiment_id"], line["job_id"], line["phase"]) == ("exp-1", "job-1", "tracking")
    assert "ts" in line


def test_json_logs_include_exceptions():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "ghostline", logging.ERROR, __file__, 1, "crashed", None, sys.exc_info()
        )

    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["exc"]
