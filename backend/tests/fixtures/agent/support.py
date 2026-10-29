"""Shared setup for the agent and review tests: a synthetic experiment in a throwaway database,
a JobContext for it, a scripted Wonder, and runners that record and replay harness results.

Synthetic data comes from `ghostline.physics.synthetic` (known truth), written the way the
trajectory stage writes it: a trajectory.parquet, a summary dict, a calibration dict.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ghostline.jobs.context import JobContext
from ghostline.jobs.stage_types import CalibrationResult, TrajectoryResult
from ghostline.physics.harness.schema import FitJob, FitResult
from ghostline.physics.synthetic import SyntheticCase, make_case
from ghostline.store.core_schemas import SceneAnalysis, TrackedObject
from ghostline.store.records import Experiment, TrajectoryRecord, Trial
from ghostline.store.volume import NullVolume

HERE = Path(__file__).parent
RECORDING = bool(os.environ.get("GHOSTLINE_RECORD"))


def spin_case(
    model_id: str = "S2",
    params: dict[str, float] | None = None,
    duration: float = 10.0,
    fps: float = 60.0,
    sigma: float = 0.03,
    seed: int = 7,
) -> SyntheticCase:
    params = params or dict(phi0=0.0, omega0=40.0, b=0.15, c=1.5)
    t = np.arange(0.0, duration, 1.0 / fps)
    return make_case("spin_down", model_id, params, {}, t, sigma, np.random.default_rng(seed))


def summary_for(case: SyntheticCase, **overrides: Any) -> dict[str, Any]:
    sigma = next(iter(case.sigma.values()))
    summary: dict[str, Any] = {
        "noise": {"sigma_angle_rad": sigma, "sigma_pos_m": sigma / 100},
        "uncertainty": {"scale_rel_at_object": 0.004, "timing_rel": 0.001, "out_of_plane_rel": 0.0},
        "out_of_plane": {"flag": False},
        "retime_check": {"checked": False},
        "gravity": {"source": "camera_level", "sigma_deg": 1.0},
        "events": {"impacts_t": []},
        "notices": [],
        "spinner": {"direction": "ccw", "marker_radius_m": 0.04},
    }
    summary.update(overrides)
    return summary


CALIBRATION = {
    "mm_per_px_center": 0.62,
    "offset": {"d_m": 0.02, "z_m": 1.2},
    "camera_motion": {"max_shift_px": 0.4, "status": "ok"},
    "uncertainty": {"sigma_scale_rel": 0.004},
}
TIMING = {"classification": "cfr", "real_fps": 60.0, "sigma_time_rel": 0.001, "retimed": False}


def spin_scene() -> SceneAnalysis:
    return SceneAnalysis(
        system_type="spin_down",
        description="A spinner with a red marker slows down and stops.",
        tracked_objects=[
            TrackedObject(key="marker", sam_prompt="red marker", role="marker", why="the marker")
        ],
        calibration={"sam_prompt": "white sheet of paper", "found": True},
        object_properties={"shape": "spinner"},
        initial_hypotheses=[
            {
                "name": "Viscous friction",
                "mechanism": "friction proportional to spin rate",
                "observable_signature": "exponential decay",
            }
        ],
        quality_issues=[],
        proceed=True,
    )


@dataclass
class World:
    experiment: Experiment
    trial: Trial
    ctx: JobContext
    scene: SceneAnalysis
    trajectory: TrajectoryResult
    calibration: CalibrationResult
    case: SyntheticCase


def make_world(
    repo,
    paths,
    settings,
    *,
    case: SyntheticCase | None = None,
    summary: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
    title: str = "Spin-down",
    description: str = "",
) -> World:
    case = case or spin_case()
    exp = repo.create_experiment(title=title, description=description, system_type="spin_down")
    scene = spin_scene()
    repo.update_experiment(exp.id, scene=scene)
    trial = repo.create_trial(experiment_id=exp.id, role="fit")
    rel = f"trials/{trial.id}/trajectory.parquet"
    target = paths.experiment_dir(exp.id) / rel
    target.parent.mkdir(parents=True)
    n = case.t.size
    pd.DataFrame(
        {
            "t": case.t,
            "phi": case.observed["phi"],
            "weight": np.ones(n),
            "flags": np.zeros(n, dtype=int),
            "segment_id": np.zeros(n, dtype=int),
        }
    ).to_parquet(target, index=False)
    summary = summary if summary is not None else summary_for(case)
    repo.update_trial(
        trial.id,
        status="ready",
        timing=timing or TIMING,
        calibration=CALIBRATION,
        n_frames=n,
        duration_s=float(case.t[-1]),
    )
    repo.upsert_trajectory(TrajectoryRecord(trial_id=trial.id, parquet_path=rel, summary=summary))
    job = repo.create_job(experiment_id=exp.id, kind="analyze_experiment")
    claimed = repo.claim_job(job.id)
    ctx = JobContext(job=claimed, repo=repo, paths=paths, volume=NullVolume(), settings=settings)
    return World(
        experiment=repo.get_experiment(exp.id),
        trial=repo.get_trial(trial.id),
        ctx=ctx,
        scene=scene,
        trajectory=TrajectoryResult(parquet_path=rel, summary=summary),
        calibration=CalibrationResult(calibration=CALIBRATION),
        case=case,
    )


# --- recorded harness results --------------------------------------------------------------
class RecordedRunner:
    """Replays harness results saved by `RecordingRunner`, keyed by job id and model ids, so
    a replayed run sees byte-identical numbers and plots on every machine."""

    def __init__(self, folder: Path, plots_dir: Path) -> None:
        self.folder, self.plots_dir = folder, plots_dir

    @staticmethod
    def key(job: FitJob) -> str:
        return f"{job.job_id}__{'_'.join(m.id for m in job.models)}"

    def run(self, job: FitJob) -> FitResult:
        path = self.folder / f"{self.key(job)}.json"
        if not path.exists():
            raise AssertionError(f"no recorded harness result for {path.name}")
        result = FitResult.model_validate_json(path.read_text())
        for model in result.models:
            for ref in model.plots:
                target = self.plots_dir / f"{result.job_id}_{model.id}_{ref.id}.png"
                shutil.copyfile(self.folder / "plots" / ref.path, target)
                ref.path = str(target)
        return result


class RecordingRunner:
    """Runs the real harness in-process and saves each result and its plots."""

    def __init__(self, folder: Path, plots_dir: Path) -> None:
        from ghostline.physics.runner import InProcessRunner

        self.folder, self.inner = folder, InProcessRunner(plots_dir)
        (folder / "plots").mkdir(parents=True, exist_ok=True)

    def run(self, job: FitJob) -> FitResult:
        result = self.inner.run(job)
        saved = result.model_copy(deep=True)
        for model in saved.models:
            for ref in model.plots:
                name = f"{saved.job_id}_{model.id}_{ref.id}.png"
                shutil.copyfile(ref.path, self.folder / "plots" / name)
                ref.path = name
        (self.folder / f"{RecordedRunner.key(job)}.json").write_text(saved.model_dump_json())
        return result


def runner_factory(folder: Path):
    """The factory `use_runner_factory` takes: record when recording, else replay."""
    return (
        (lambda plots: RecordingRunner(folder, plots))
        if RECORDING
        else (lambda plots: RecordedRunner(folder, plots))
    )


# --- a scripted Wonder -------------------------------------------------------------------
def call_item(name: str, args: dict[str, Any], n: int) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{n}",
        "call_id": f"call_{n}",
        "name": name,
        "arguments": json.dumps(args),
        "status": "completed",
    }


def last_tool_json(items: list[dict[str, Any]], name: str | None = None) -> dict[str, Any]:
    """The JSON object in the latest function_call_output (text before any guidance)."""
    for item in reversed(items):
        if item.get("type") == "function_call_output":
            out = item["output"]
            text = out if isinstance(out, str) else next(p["text"] for p in out if "text" in p)
            return json.loads(text.split("\n\n")[0])
    raise AssertionError("no tool output yet")


class PolicySDK:
    """A `responses` SDK stand-in whose replies come from `policy(items, n) -> output items`,
    so a scripted Wonder can read the numbers the harness printed and quote them back."""

    def __init__(self, policy) -> None:
        self.policy = policy
        self.calls: list[dict[str, Any]] = []
        self.responses = self

    def create(self, **kwargs: Any):
        from fixtures.fakes import Obj, response_dict, stream_events

        self.calls.append(kwargs)
        output = self.policy(kwargs["input"], len(self.calls))
        events = stream_events(response_dict(output=output))
        return iter(Obj(e) for e in events)


def message_output(text: str) -> dict[str, Any]:
    from fixtures.fakes import message_item

    return message_item(text)


def scripted_clients(policy, tmp_path):
    """(meta, reviewer, sdk) in live mode on a scripted Wonder; no network, no keys."""
    from ghostline.config import get_settings
    from ghostline.llm import InMemorySpendStore, MetaClient, ReviewerClient, Runtime

    runtime = Runtime.create(
        get_settings(),
        mode="live",
        cache_dir=tmp_path / "llm-cache",
        store=InMemorySpendStore(),
        sleep=lambda s: None,
    )
    sdk = PolicySDK(policy)
    return MetaClient(runtime, sdk=sdk), ReviewerClient(runtime, sdk=None), sdk


def tool_texts(items: list[dict[str, Any]]) -> list[str]:
    """The text of every tool output so far."""
    texts = []
    for item in items:
        if item.get("type") == "function_call_output":
            out = item["output"]
            texts.append(
                out if isinstance(out, str) else next(p["text"] for p in out if "text" in p)
            )
    return texts


def find_outputs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every tool output so far, parsed (the JSON before any trailing guidance)."""
    found = []
    for text in tool_texts(items):
        try:
            found.append(json.loads(text.split("\n\n")[0]))
        except json.JSONDecodeError:
            found.append({"text": text})
    return found


def conclude_message_json(items: list[dict[str, Any]]) -> dict[str, Any]:
    """The final numbers the orchestrator put in the CONCLUDE message."""
    for item in reversed(items):
        if item.get("type") == "message" and item.get("role") == "user":
            for part in item["content"]:
                text = part.get("text", "")
                if "Final numbers:\n" in text:
                    body = text.split("Final numbers:\n", 1)[1].split("\n\nCall submit", 1)[0]
                    return json.loads(body)
    raise AssertionError("no final numbers in the conversation")


REVIEWER_MODEL = "meta-models/Muse-Spark-Open-816B-A42B"  # the model name in the recorded requests


def recorded_clients(folder: Path):
    """(meta, reviewer, runtime, store) on the recordings in `folder/cache`: live calls that are
    saved when GHOSTLINE_RECORD is set (run with `uv run --env-file .env`), replay otherwise."""
    import dataclasses

    from fixtures.scenarios import NoNetwork

    from ghostline.config import get_settings
    from ghostline.llm import InMemorySpendStore, MetaClient, ReviewerClient, Runtime

    store = InMemorySpendStore()
    if RECORDING:
        runtime = Runtime.create(
            get_settings(), mode="record", cache_dir=folder / "cache", store=store
        )
        return MetaClient(runtime), ReviewerClient(runtime), runtime, store
    settings = dataclasses.replace(
        get_settings(),
        model_api_key="",
        reviewer_api_key="",
        physicist_model="muse-wonder-1.0",
        reviewer_model=REVIEWER_MODEL,
    )
    runtime = Runtime.create(settings, mode="replay", cache_dir=folder / "cache", store=store)
    return (
        MetaClient(runtime, sdk=NoNetwork()),
        ReviewerClient(runtime, sdk=NoNetwork()),
        runtime,
        store,
    )


def pin_training_plot(monkeypatch, folder: Path) -> None:
    """The first prompt carries a matplotlib PNG, and the request key hashes its bytes. Save the
    plot when recording and serve the saved bytes on replay, so replay doesn't depend on the
    machine's matplotlib output."""
    import ghostline.agent.loop as loop

    real, path = loop.training_plot, folder / "training_plot.png"

    def pinned(inputs):
        if RECORDING:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(real(inputs))
        return path.read_bytes()

    monkeypatch.setattr(loop, "training_plot", pinned)
