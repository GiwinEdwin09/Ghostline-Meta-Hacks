import pytest
from fixtures.fakes import FakeSDK, response_dict, sam_stream, stream_events

from ghostline.config import get_settings
from ghostline.llm import InMemorySpendStore, MetaClient, Runtime
from ghostline.llm.pricing import (
    SPARK,
    WONDER,
    price_for_model,
    sam_image_cost,
    sam_video_cost,
)


def test_wonder_prices_per_million_tokens():
    cost = WONDER.cost(input_tokens=1000, cached_tokens=400, output_tokens=500)

    assert cost == pytest.approx(600 * 4.0e-6 + 400 * 0.2e-6 + 500 * 20.0e-6)


def test_reasoning_tokens_are_billed_as_output():
    # Day 0 smoke, step 1: 22 input and 418 output tokens, 390 of them reasoning, cost $0.0084.
    assert WONDER.cost(22, 0, 418) == pytest.approx(0.008448)


def test_spark_prices():
    assert SPARK.cost(1_000_000, 0, 0) == pytest.approx(1.25)
    assert SPARK.cost(1_000_000, 1_000_000, 0) == pytest.approx(0.15)
    assert SPARK.cost(0, 0, 1_000_000) == pytest.approx(4.25)


def test_models_price_by_name():
    assert price_for_model("muse-spark-1.3") is SPARK
    assert price_for_model("muse-wonder-1.0") is WONDER
    assert price_for_model("something-new") is WONDER  # unknown models price at the dearer rate


def test_sam_video_is_billed_per_frame_with_a_50_frame_minimum():
    assert sam_video_cost(600) == pytest.approx(0.12)
    assert sam_video_cost(90) == pytest.approx(0.018)
    assert sam_video_cost(10) == pytest.approx(0.01)
    assert sam_video_cost(0) == pytest.approx(0.01)


def test_sam_image_price():
    assert sam_image_cost() == pytest.approx(0.0025)


def client(sdk, tmp_path, store=None):
    runtime = Runtime.create(
        get_settings(),
        mode="live",
        cache_dir=tmp_path,
        store=store or InMemorySpendStore(),
        sleep=lambda s: None,
    )
    return MetaClient(runtime, sdk=sdk)


def test_a_wonder_call_is_metered_and_logged(tmp_path):
    reply = response_dict("hi", input_tokens=1000, cached=400, output_tokens=500, reasoning=450)
    store = InMemorySpendStore()
    meta = client(FakeSDK([stream_events(reply)]), tmp_path, store)

    response = meta.respond(input="hello", experiment_id="exp1", phase="hypothesize")

    expected = WONDER.cost(1000, 400, 500)
    assert response.cost_usd == pytest.approx(expected)
    (row,) = store.records()
    assert (row.input_tokens, row.cached_tokens, row.output_tokens) == (1000, 400, 500)
    assert row.reasoning_tokens == 450
    assert row.cost_usd == pytest.approx(expected)
    assert (row.experiment_id, row.phase, row.kind, row.provider) == (
        "exp1",
        "hypothesize",
        "respond",
        "meta",
    )
    assert store.experiment_total("exp1") == pytest.approx(expected)


def test_sam_video_cost_uses_the_reported_frame_count(tmp_path):
    store = InMemorySpendStore()
    sdk = FakeSDK([sam_stream(["<0f>0"], frames=600)])
    meta = client(sdk, tmp_path, store)

    result = meta.sam_video("file-1", "orange ball", experiment_id="exp1")

    assert result.frames_processed == 600
    assert result.cost_usd == pytest.approx(0.12)
    (row,) = store.records()
    assert row.frames == 600 and row.kind == "sam_video"
    assert row.cost_usd == pytest.approx(0.12)


def test_a_short_sam_clip_is_billed_the_minimum(tmp_path):
    meta = client(FakeSDK([sam_stream(["<0f>0"], frames=12)]), tmp_path)

    assert meta.sam_video("file-1", "orange ball").cost_usd == pytest.approx(0.01)


def test_sam_image_cost(tmp_path):
    store = InMemorySpendStore()
    meta = client(FakeSDK([sam_stream(["<0f>0"], frames=1, kind="image")]), tmp_path, store)

    result = meta.sam_image(b"png-bytes", "white sheet of paper")

    assert result.cost_usd == pytest.approx(0.0025)
    assert store.records()[0].kind == "sam_image"
