from dataclasses import replace
from datetime import timedelta

import pytest
from fixtures.fakes import FakeSDK, response_dict, stream_events

from ghostline.config import get_settings
from ghostline.llm import (
    BudgetGuard,
    CallRecord,
    DailyBudgetExceeded,
    ExperimentBudgetExceeded,
    InMemorySpendStore,
    JsonFileSpendStore,
    MetaClient,
    Runtime,
)
from ghostline.llm.ledger import utc_now


def record(cost, experiment_id="exp1", ts=None, cache_hit=False):
    row = CallRecord(
        provider="meta",
        kind="respond",
        model="m",
        cost_usd=cost,
        request_hash="h",
        experiment_id=experiment_id,
        cache_hit=cache_hit,
    )
    return replace(row, ts=ts.isoformat()) if ts else row


def test_the_daily_cap_blocks_a_new_analysis():
    store = InMemorySpendStore()
    guard = BudgetGuard(store, daily_usd=20, per_experiment_usd=3)
    guard.check_new_analysis()  # fine at first

    store.add(record(12.0, "exp1"))
    store.add(record(8.5, "exp2"))

    with pytest.raises(DailyBudgetExceeded) as caught:
        guard.check_new_analysis()
    assert "budget" in str(caught.value) and "example" in str(caught.value)


def test_yesterdays_spend_does_not_count_today():
    store = InMemorySpendStore()
    store.add(record(50.0, ts=utc_now() - timedelta(days=1)))
    guard = BudgetGuard(store, daily_usd=20, per_experiment_usd=3)

    guard.check_new_analysis()
    assert guard.spent_today() == 0


def test_the_daily_cap_does_not_cut_off_an_analysis_already_running():
    store = InMemorySpendStore()
    store.add(record(25.0, "other"))
    guard = BudgetGuard(store, daily_usd=20, per_experiment_usd=3)

    guard.check_call("exp1")  # no error: only a new analysis is refused


def test_the_experiment_cap_stops_further_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("BUDGET_USD_PER_EXPERIMENT", "0.01")
    # 1000 input and 500 output tokens cost $0.014, over the cap in one call.
    reply = response_dict("ok", input_tokens=1000, output_tokens=500)
    sdk = FakeSDK([stream_events(reply), stream_events(reply)])
    runtime = Runtime.create(get_settings(), mode="live", cache_dir=tmp_path)
    meta = MetaClient(runtime, sdk=sdk)

    meta.respond(input="one", experiment_id="exp1")
    with pytest.raises(ExperimentBudgetExceeded) as caught:
        meta.respond(input="two", experiment_id="exp1")

    assert len(sdk.responses.calls) == 1  # the second call never reached the API
    assert caught.value.cap == 0.01 and caught.value.spent > 0.01
    assert runtime.guard.experiment_exhausted("exp1")


def test_other_experiments_and_unscoped_calls_are_not_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("BUDGET_USD_PER_EXPERIMENT", "0.01")
    reply = response_dict("ok", input_tokens=1000, output_tokens=500)
    sdk = FakeSDK([stream_events(reply)] * 3)
    runtime = Runtime.create(get_settings(), mode="live", cache_dir=tmp_path)
    meta = MetaClient(runtime, sdk=sdk)

    meta.respond(input="a", experiment_id="exp1")
    meta.respond(input="b", experiment_id="exp2")
    meta.respond(input="c")  # no experiment: only the daily total sees it

    assert len(sdk.responses.calls) == 3


def test_remaining_budget(tmp_path):
    store = InMemorySpendStore()
    store.add(record(1.25, "exp1"))
    guard = BudgetGuard(store, daily_usd=20, per_experiment_usd=3)

    assert guard.experiment_remaining("exp1") == pytest.approx(1.75)
    assert guard.experiment_remaining("untouched") == pytest.approx(3.0)


def test_budget_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("BUDGET_USD_DAILY", "7.5")
    monkeypatch.setenv("BUDGET_USD_PER_EXPERIMENT", "1.5")

    guard = BudgetGuard.from_settings(InMemorySpendStore(), get_settings())

    assert (guard.daily_usd, guard.per_experiment_usd) == (7.5, 1.5)


def test_json_file_store_persists_spend_across_instances(tmp_path):
    path = tmp_path / "spend" / "calls.jsonl"
    first = JsonFileSpendStore(path)
    first.add(record(0.5, "exp1"))
    first.add(record(0.25, "exp2"))

    second = JsonFileSpendStore(path)  # a new process reading the same file
    second.add(record(0.125, "exp1"))

    assert second.experiment_total("exp1") == pytest.approx(0.625)
    assert second.day_total(utc_now().date()) == pytest.approx(0.875)
    assert len(second.records("exp2")) == 1
    guard = BudgetGuard(second, daily_usd=0.8, per_experiment_usd=3)
    with pytest.raises(DailyBudgetExceeded):
        guard.check_new_analysis()


def test_an_empty_json_file_store_reads_as_zero(tmp_path):
    store = JsonFileSpendStore(tmp_path / "none.jsonl")

    assert store.records() == [] and store.experiment_total("x") == 0


def test_cache_hits_cost_nothing_and_never_trip_a_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("BUDGET_USD_PER_EXPERIMENT", "0.01")
    reply = response_dict("ok", input_tokens=1000, output_tokens=500)
    recorder = Runtime.create(get_settings(), mode="record", cache_dir=tmp_path)
    MetaClient(recorder, sdk=FakeSDK([stream_events(reply)])).respond(input="q")

    replayer = Runtime.create(get_settings(), mode="replay", cache_dir=tmp_path)
    meta = MetaClient(replayer, sdk=None)
    for _ in range(3):
        response = meta.respond(input="q", experiment_id="exp1")
        assert response.cache_hit and response.cost_usd > 0  # list price of the usage

    rows = replayer.store.records("exp1")
    assert [r.cache_hit for r in rows] == [True] * 3
    assert replayer.store.experiment_total("exp1") == 0
