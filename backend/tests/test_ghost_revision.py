"""Divergence diagnosis and revision (build plan Part 17): an undamped v1 on a damped pendulum."""

import dataclasses
import json
from uuid import uuid4

import numpy as np
import pytest
from fixtures.ghost.replay import recorded_reply
from fixtures.ghost.rig import pendulum_rig

from ghostline.config import get_settings
from ghostline.ghost.diagnose import DiagnosisReply, ParamDef, RevisedModel
from ghostline.ghost.revise import paired_rmse_difference, spec_from_reply
from ghostline.ghost.stage import diagnose_revision, predict_ghost
from ghostline.jobs.errors import StageError
from ghostline.physics.harness.schema import ModelSpec
from ghostline.store.core_schemas import Review
from ghostline.store.records import GhostRender
from ghostline.store.volume import VolumePaths

GAMMA = 0.15  # 1/s: the true damping, which an undamped v1 knows nothing about
D1, D2 = 5.0, 10.0


def damping_reply(source="library", ref="P2") -> DiagnosisReply:
    return DiagnosisReply(
        diagnosis="The real swing shrinks while the ghost keeps its amplitude: damping is missing.",
        missing_physics="viscous damping",
        revised_model=RevisedModel(
            source=source, library_ref=ref, id="", name="Viscous damping",
            mechanism="Drag proportional to speed removes energy every swing.",
            equations_latex=r"\ddot\theta = -\tfrac{g}{L}\sin\theta - \gamma\dot\theta",
            code="", state=[], params=[],
        ),
        predicted_signature="The amplitude decays exponentially with the same loss every swing.",
    )  # fmt: skip


@pytest.fixture(scope="module")
def damped(tmp_path_factory):
    root = tmp_path_factory.mktemp("ghost-revise")
    paths = VolumePaths(root / "data")
    settings = dataclasses.replace(get_settings(), volume_root=str(root / "data"))
    rig = pendulum_rig(paths, settings, gamma=GAMMA, theta1=0.5, theta2=0.4, d1=D1, d2=D2)
    v1 = predict_ghost(
        rig.ctx, rig.experiment, rig.fit_trial, rig.prove_trial, rig.scene, rig.trajectory
    )
    rig.repo.ghost_count = 1
    ghost = GhostRender(
        id=uuid4(), experiment_id=rig.experiment.id, trial_id=rig.prove_trial.id, model_id="P1",
        version=1, video_path=v1.video_path, track_path=v1.track_path,
    )  # fmt: skip
    rig.experiment = rig.experiment.model_copy(update={"scene": rig.scene})
    rig.repo.latest_trial = lambda *a, **k: rig.fit_trial
    asked = []

    def ask(inputs):
        asked.append(inputs)
        return recorded_reply()  # what Wonder said when it looked at this drift

    result = diagnose_revision(rig.ctx, rig.experiment, ghost, ask=ask)
    return rig, v1, ghost, result, asked


def test_an_undamped_v1_drifts_on_a_damped_trial(damped):
    _, v1, _, _, _ = damped
    assert v1.divergence_onset_s is not None
    assert 1.0 < v1.divergence_onset_s < 4.0  # after the launch window, long before the end
    assert v1.max_error_mm > 3 * v1.threshold_mm


def test_the_v2_ghost_tracks_noticeably_longer(damped):
    rig, v1, _, result, _ = damped
    assert result.adopted and result.ghost is not None
    root = rig.paths.experiment_dir(rig.experiment.id)
    onset_v1 = v1.divergence_onset_s
    onset_v2 = result.ghost.divergence_onset_s
    assert onset_v2 is None or onset_v2 > onset_v1 + 3.0  # tracks at least 3 s longer
    assert result.ghost.max_error_mm < v1.max_error_mm / 3
    assert result.ghost.version == 2 and result.ghost.video_path == "ghost/v2.mp4"
    assert (root / result.ghost.video_path).stat().st_size > 10_000
    track = json.loads((root / result.ghost.track_path).read_text())
    assert track["version"] == 2 and track["model_id"] == "P2-rev"


def test_wonder_sees_the_error_plot_four_frames_and_the_numbers(damped):
    _, v1, _, _, asked = damped
    (inputs,) = asked
    assert len(inputs.images) == 5 and inputs.images[0][:4] == b"\x89PNG"
    assert inputs.images[1][:2] == b"\xff\xd8"
    assert "P1" in inputs.text and "9.8" in inputs.text  # the model and its frozen g
    assert f"{v1.threshold_mm:.2f}" in inputs.text  # tau
    assert "P2: " in inputs.text  # the library catalog it may choose from
    assert "initial_state" in inputs.text  # the custom model contract


def test_physical_parameters_of_the_revision_come_from_trial_1_only(damped):
    rig, _, _, _, _ = damped
    (event,) = [
        e for e in rig.repo.events if e["kind"] == "fit" and "tested on trial 2" in e["title"]
    ]
    fit = event["data"]
    assert fit["frozen_v2"]["gamma"] == pytest.approx(GAMMA, rel=0.1)  # trial 1's damping
    assert fit["rmse_difference_ci95_mm"][0] > 0
    assert fit["rmse_v2_mm"] < fit["rmse_v1_mm"] / 3


def test_the_result_carries_the_diagnosis_model_row_and_notebook_entries(damped):
    rig, _, _, result, _ = damped
    assert "damping" in result.diagnosis_md and "Adopted" in result.diagnosis_md
    model = result.revised_model
    assert model.id == "P2-rev" and model.library_ref == "P2" and model.status == "proposed"
    assert model.round == 1
    kinds = [e["kind"] for e in rig.repo.events]
    assert "revision" in kinds and kinds.count("fit") == 1
    diagnosis = next(e for e in rig.repo.events if e["kind"] == "revision")
    assert diagnosis["actor"] == "wonder" and len(diagnosis["attachments"]) == 5
    assert {"fitting", "rendering"} <= set(rig.repo.phases)  # the handler enters "diagnosing"


def test_a_revision_that_does_not_help_is_recorded_as_such(tmp_path):
    """The same law on both trials: v1 is already right, so P2 adds nothing beyond noise."""
    paths = VolumePaths(tmp_path / "data")
    settings = dataclasses.replace(get_settings(), volume_root=str(tmp_path / "data"))
    rig = pendulum_rig(paths, settings, gamma=0.0, d1=6.0, d2=6.0)
    predict_ghost(
        rig.ctx, rig.experiment, rig.fit_trial, rig.prove_trial, rig.scene, rig.trajectory
    )
    rig.repo.ghost_count = 1
    rig.experiment = rig.experiment.model_copy(update={"scene": rig.scene})
    rig.repo.latest_trial = lambda *a, **k: rig.fit_trial
    ghost = GhostRender(
        id=uuid4(), experiment_id=rig.experiment.id, trial_id=rig.prove_trial.id, model_id="P1",
        version=1,
    )  # fmt: skip
    result = diagnose_revision(rig.ctx, rig.experiment, ghost, ask=lambda i: damping_reply())
    assert not result.adopted and result.ghost is None
    assert "didn't help" in result.diagnosis_md
    assert result.revised_model.status == "rejected" and result.revised_model.rejection_reason


def test_proposing_the_model_that_drifted_or_a_non_library_id_is_refused(tmp_path):
    paths = VolumePaths(tmp_path / "data")
    settings = dataclasses.replace(get_settings(), volume_root=str(tmp_path / "data"))
    rig = pendulum_rig(paths, settings, gamma=GAMMA, d1=4.0, d2=4.0, video=True)
    rig.repo.ghost_count = 1
    rig.experiment = rig.experiment.model_copy(update={"scene": rig.scene})
    rig.repo.latest_trial = lambda *a, **k: rig.fit_trial
    ghost = GhostRender(
        id=uuid4(), experiment_id=rig.experiment.id, trial_id=rig.prove_trial.id, model_id="P1",
        version=1,
    )  # fmt: skip
    with pytest.raises(StageError, match="again"):
        diagnose_revision(rig.ctx, rig.experiment, ghost, ask=lambda i: damping_reply(ref="P1"))
    with pytest.raises(StageError, match="not a library model"):
        diagnose_revision(rig.ctx, rig.experiment, ghost, ask=lambda i: damping_reply(ref="R1"))


def test_the_optional_review_hook_sees_the_dossier_and_its_verdict_is_kept(damped, tmp_path):
    rig, _, ghost, _, _ = damped
    seen = []

    def review(dossier):
        seen.append(dossier)
        return Review(verdict="accept", issues=[], summary="The damping story holds up.")

    result = diagnose_revision(
        rig.ctx, rig.experiment, ghost, ask=lambda i: damping_reply(), review=review, n_draws=20
    )
    (dossier,) = seen
    assert dossier.adopted and dossier.v2_model_id == "P2-rev" and dossier.rmse_v2_mm < 5
    assert "Reviewer (accept)" in result.diagnosis_md


# --- pieces ---------------------------------------------------------------------------------
def test_the_adoption_rule_needs_the_whole_95_percent_interval_above_zero():
    rng = np.random.default_rng(0)
    base = np.abs(rng.normal(5, 1, 200))
    r1, r2, (lo, hi) = paired_rmse_difference(base * 3, base)
    assert r1 > r2 and lo > 0
    _, _, (lo, hi) = paired_rmse_difference(base, base * (1 + rng.normal(0, 0.02, 200)))
    assert lo < 0 < hi  # an indistinguishable pair straddles zero


def test_a_custom_reply_becomes_a_spec_with_its_parameters_and_a_fresh_id():
    code = "def initial_state(p):\n    return [p['theta0'], p['omega0']]\n"
    ic = ParamDef(name="theta0", init=0.1, lower=-3, upper=3, unit="rad", kind="initial_condition")
    law = ParamDef(name="g", init=9.8, lower=1, upper=30, unit="m/s^2", kind="physical")
    reply = damping_reply(source="custom", ref="").revised_model.model_copy(
        update={"id": "Damped pendulum!", "code": code, "state": ["theta"], "params": [ic, law]}
    )
    spec = spec_from_reply(reply, "pendulum", ("theta",), taken={"Dampedpendulum"})
    assert isinstance(spec, ModelSpec) and spec.id == "Dampedpendulumx"
    assert spec.observables == ["theta"] and spec.params["g"].kind == "physical"
    with pytest.raises(Exception, match="needs code"):
        spec_from_reply(reply.model_copy(update={"code": ""}), "pendulum", ("theta",), set())
