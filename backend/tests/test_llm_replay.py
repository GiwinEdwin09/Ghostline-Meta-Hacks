"""Replay of the calls recorded in fixtures/cache (made once with `fixtures/record.py`)."""

import os
import re

import pytest
from fixtures import scenarios

from ghostline.llm import ReplayMissError
from ghostline.llm.pricing import WONDER


@pytest.fixture
def clients(monkeypatch):
    return scenarios.replay_clients(monkeypatch)


def test_structured_output_replays(clients):
    meta, _, runtime = clients

    response = scenarios.gravity(meta)

    assert response.parsed.planet == "Mars" and 3.5 < response.parsed.g_m_s2 < 3.9
    assert response.cache_hit and response.repairs == 0
    assert response.cost_usd == pytest.approx(WONDER.cost(22, 0, 304))


def test_a_tool_round_trip_replays_with_an_image_output(clients):
    meta, _, _ = clients

    first, second = scenarios.tool_roundtrip(meta)

    (call,) = first.function_calls
    assert call.name == "get_plot" and call.args() == {"name": "frame"}
    assert any(item["type"] == "function_call" for item in first.output)
    assert second.output_text  # Wonder described the image it got back from the tool


def test_a_full_set_of_calls_replays_with_byte_identical_results(monkeypatch):
    def run_everything():
        meta, reviewer, runtime = scenarios.replay_clients(monkeypatch)
        first, second = scenarios.tool_roundtrip(meta)
        scene = scenarios.scene(meta)
        ball = scenarios.sam_video(meta, scenarios.BALL_PROMPT)
        outputs = [
            scenarios.gravity(meta).model_dump_json(),
            first.model_dump_json(),
            second.model_dump_json(),
            scene.analysis.model_dump_json(),
            scene.response.model_dump_json(),
            ball.output_text,
            "".join(str(e) for e in ball.events),
            scenarios.sam_sheet(meta).output_text,
            str(scenarios.rephrase(meta)),
            str(scenarios.box(meta)),
            scenarios.reviewer_reading(reviewer).model_dump_json(),
        ]
        return outputs, runtime

    first_run, first_runtime = run_everything()
    second_run, _ = run_everything()

    assert first_run == second_run
    rows = first_runtime.store.records()
    assert len(rows) == 9 and all(r.cache_hit for r in rows)
    assert first_runtime.store.day_total(rows[0].day) == 0  # replay costs nothing


def test_replay_needs_no_network_and_no_keys(clients):
    meta, reviewer, runtime = clients

    assert runtime.mode == "replay"
    assert not os.environ.get("MODEL_API_KEY") and runtime.settings.model_api_key == ""
    scenarios.gravity(meta)  # the SDK stand-in fails the test if it is touched


def test_an_unrecorded_request_fails_loudly(clients):
    meta, _, _ = clients

    with pytest.raises(ReplayMissError):
        meta.respond(input="a question nobody recorded")


def test_the_cache_holds_no_secrets():
    pattern = re.compile(r"Bearer |\bsk-\w|\bwk-\w+\.ws-|api[_-]key|authorization", re.IGNORECASE)
    for path in scenarios.CACHE_DIR.rglob("*.json"):
        text = path.read_text()
        assert not pattern.search(text), f"a key or auth header may be in {path.name}"
        assert "base64," not in text  # images are stored as hashes
