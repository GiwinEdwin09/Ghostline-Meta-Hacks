"""Wonder's diagnosis of an undamped ghost on a damped pendulum, replayed from a live recording."""

import pytest
from fixtures.ghost.replay import CACHE, recorded_inputs, replay_client

from ghostline.ghost.diagnose import DiagnosisInputs, ask_wonder, load_prompt
from ghostline.llm import ReplayMissError


def test_the_recorded_diagnosis_names_damping_and_proposes_the_damped_pendulum():
    reply = ask_wonder(replay_client(), recorded_inputs())
    said = " ".join(
        [
            reply.diagnosis,
            reply.missing_physics,
            reply.revised_model.name,
            reply.revised_model.mechanism,
        ]
    ).lower()
    assert "damping" in said or "drag" in said
    assert reply.revised_model.source == "library" and reply.revised_model.library_ref == "P2"
    assert reply.predicted_signature and "damping" in reply.missing_physics.lower()


def test_what_wonder_was_shown_is_the_plot_four_frames_and_the_numbers():
    inputs = recorded_inputs()
    assert len(inputs.images) == 5 and inputs.images[0][:4] == b"\x89PNG"
    assert all(image[:2] == b"\xff\xd8" for image in inputs.images[1:])
    for needle in ("P1", "Threshold tau", "Library models you may choose", "initial_state"):
        assert needle in inputs.text
    assert sum(len(i) for i in inputs.images) < 400_000  # small enough to commit


def test_a_changed_request_is_not_replayed():
    inputs = recorded_inputs()
    changed = DiagnosisInputs(inputs.text + " extra", inputs.images, inputs.labels)
    with pytest.raises(ReplayMissError):
        ask_wonder(replay_client(), changed)


def test_the_prompt_file_is_what_was_recorded_and_the_cache_holds_no_secrets():
    text, version = load_prompt()
    assert "ghost" in text and len(version) == 12
    for path in CACHE.rglob("*.json"):
        body = path.read_text().lower()
        assert "authorization" not in body and "bearer" not in body and "api_key" not in body
