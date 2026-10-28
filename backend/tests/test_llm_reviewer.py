import json

import pytest
from fixtures.fakes import FakeSDK, chat_dict, status_error
from pydantic import BaseModel

from ghostline.config import get_settings
from ghostline.llm import (
    InMemorySpendStore,
    PermanentLLMError,
    ReplayMissError,
    ReviewerClient,
    Runtime,
    StructuredOutputError,
    TransientLLMError,
)
from ghostline.llm.pricing import SPARK


class Verdict(BaseModel):
    verdict: str
    summary: str


GOOD = json.dumps({"verdict": "accept", "summary": "fine"})
MESSAGES = [{"role": "user", "content": "review this"}]


def make(chats, tmp_path, mode="live", monkeypatch=None):
    runtime = Runtime.create(
        get_settings(),
        mode=mode,
        cache_dir=tmp_path,
        store=InMemorySpendStore(),
        sleep=lambda s: None,
    )
    return ReviewerClient(runtime, sdk=FakeSDK(chats=chats)), runtime


@pytest.fixture(autouse=True)
def reviewer_env(monkeypatch):
    monkeypatch.setenv("REVIEWER_MODEL", "spark-open-test")


def test_the_affinity_header_is_the_experiment_id(tmp_path):
    reviewer, _ = make([chat_dict(GOOD)], tmp_path)

    reviewer.chat(MESSAGES, experiment_id="exp-42")

    (call,) = reviewer.sdk.chat.completions.calls
    assert call["extra_headers"] == {"Modal-Routing-Affinity-Key": "exp-42"}
    assert call["model"] == "spark-open-test" and call["messages"] == MESSAGES


def test_structured_output_uses_response_format_and_parses(tmp_path):
    reviewer, _ = make([chat_dict(GOOD)], tmp_path)

    result = reviewer.chat(MESSAGES, experiment_id="e", schema=Verdict)

    assert result.parsed == Verdict(verdict="accept", summary="fine")
    fmt = reviewer.sdk.chat.completions.calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Verdict" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["required"] == ["verdict", "summary"]


def test_cost_uses_spark_prices_with_cached_input(tmp_path):
    reviewer, runtime = make([chat_dict("ok", prompt=2000, completion=300, cached=1500)], tmp_path)

    result = reviewer.chat(MESSAGES, experiment_id="e")

    expected = SPARK.cost(2000, 1500, 300)
    assert result.cost_usd == pytest.approx(expected)
    (row,) = runtime.store.records("e")
    assert (row.provider, row.kind, row.cached_tokens) == ("modal", "chat", 1500)
    assert row.cost_usd == pytest.approx(expected)


def test_one_repair_call_on_an_invalid_reply(tmp_path):
    reviewer, _ = make([chat_dict('{"verdict": "accept"}'), chat_dict(GOOD)], tmp_path)

    result = reviewer.chat(MESSAGES, experiment_id="e", schema=Verdict)

    assert result.parsed.summary == "fine" and result.repairs == 1
    repair = reviewer.sdk.chat.completions.calls[1]["messages"]
    assert [m["role"] for m in repair] == ["user", "assistant", "user"]
    assert "summary" in repair[-1]["content"]
    assert result.input_tokens == 400  # both calls counted


def test_still_invalid_after_the_repair_raises(tmp_path):
    reviewer, _ = make([chat_dict("{}"), chat_dict("nope")], tmp_path)

    with pytest.raises(StructuredOutputError):
        reviewer.chat(MESSAGES, experiment_id="e", schema=Verdict)


def test_retries_and_permanent_errors(tmp_path):
    reviewer, _ = make([status_error(503), chat_dict(GOOD)], tmp_path)
    assert reviewer.chat(MESSAGES, experiment_id="e").content == GOOD

    reviewer, _ = make([status_error(400)], tmp_path)
    with pytest.raises(PermanentLLMError):
        reviewer.chat(MESSAGES, experiment_id="e")

    reviewer, _ = make([status_error(503)] * 4, tmp_path)
    with pytest.raises(TransientLLMError):  # the analysis then reports "not reviewed"
        reviewer.chat(MESSAGES, experiment_id="e")


def test_record_and_replay(tmp_path):
    recorder, _ = make([chat_dict(GOOD)], tmp_path, mode="record")
    recorded = recorder.chat(MESSAGES, experiment_id="e", schema=Verdict)

    replayer, runtime = make([], tmp_path, mode="replay")
    replayed = replayer.chat(MESSAGES, experiment_id="e", schema=Verdict)

    assert replayed.cache_hit and replayed.parsed == recorded.parsed
    assert replayed.content == recorded.content
    assert runtime.store.experiment_total("e") == 0
    with pytest.raises(ReplayMissError):
        replayer.chat([{"role": "user", "content": "other"}], experiment_id="e")
