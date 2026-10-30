"""The Ghostline JSON export and the PyBullet snippet text, from experiments fitted on synthetic
data (build plan Part 19). No test here needs PyBullet."""

import ast
import math

import export_helpers as h
import pytest

from ghostline.export import conversions as conv
from ghostline.export.mapping import export_formats
from ghostline.export.measured import measure
from ghostline.export.snippet import render_pybullet
from ghostline.export.stage import available_formats, build_export
from ghostline.jobs.errors import StageError
from ghostline.store.core_schemas import SimExport


@pytest.fixture(scope="module")
def ramp():
    repo, experiment, result, a_true = h.fitted_ramp_block()
    return repo, experiment


@pytest.fixture(scope="module")
def bounce():
    repo, experiment, result = h.fitted_bounce()
    return repo, experiment


def pendulum(model="P2", param="gamma", value=0.12, **kwargs):
    length, radius = 0.62, 0.012
    return h.build(
        "pendulum",
        model,
        [h.quantity("g", 9.79, 0.01), h.quantity(param, value, 0.04)],
        summary={
            "pendulum": {"length_m": length, "bob_radius_m": radius},
            "samples": {"theta": [0.66, 0.5, 0.4, 0.3, 0.25, 0.2]},
            "ranges": {"theta": [-0.66, 0.66], "speed_max_m_s": 1.5},
        },
        mover="steel nut",
        surface=None,
        size_cm=2.4,
        **kwargs,
    )


def spinner(model="S2"):
    quantities = [h.quantity("b", 0.05), h.quantity("c", 1.2)]
    if model == "S3":
        quantities = [h.quantity("c", 1.2), h.quantity("k", 0.004)]
    return h.build("spin_down", model, quantities, mover="fidget spinner", surface=None)


# --- the JSON -------------------------------------------------------------------------------
def test_ramp_json_carries_the_fitted_friction_with_its_interval(ramp):
    repo, experiment = ramp
    q = next(q for q in experiment.conclusion.quantities if q.name == "mu_k")

    export = build_export(repo, experiment, "json", 200.0)

    assert isinstance(export, SimExport)
    assert export.schema_version == "ghostline.sim/1"
    (param,) = export.parameters
    assert (param.name, param.from_model) == ("friction.kinetic", "R1")
    assert (param.value, param.ci95) == (q.value, (q.ci_low, q.ci_high))
    assert export.randomization == {"friction.kinetic": (q.ci_low, q.ci_high)}
    assert export.mass_kg == pytest.approx(0.2)
    assert export.confidence == "high"
    assert export.object.startswith("wooden block") and export.surface == "painted board"
    assert export.experiment_url.endswith(f"/e/{experiment.id}")


def test_the_json_says_where_the_numbers_apply(ramp):
    repo, experiment = ramp

    export = build_export(repo, experiment, "json", None)

    notes = " ".join(export.valid_for["notes"])
    assert "painted board" in notes  # the surface pair
    assert "extrapolation" in notes and "ramp angle 25 deg" in notes  # the measured ranges
    assert "measurement uncertainty only" in notes  # what the randomization ranges cover
    assert export.valid_for["angle_deg"] == [25.0, 25.0]
    assert export.valid_for["caveats"] == ["Measured on one clip."]
    assert export.mass_kg is None


def test_bounce_json_exports_restitution(bounce):
    repo, experiment = bounce
    e = next(q for q in experiment.conclusion.quantities if q.name == "e")

    export = build_export(repo, experiment, "json", None)

    (param,) = export.parameters
    assert (param.name, param.value) == ("restitution", e.value)
    g = measure(repo, experiment).gravity  # the fitted g, since the bounce fit measures it
    assert export.valid_for["impact_speed_m_s"][1] == pytest.approx(math.sqrt(2 * g * 1.0))


def test_a_speed_dependent_bounce_exports_e_at_the_median_speed_and_keeps_both_coefficients():
    summary = {
        "ranges": {"y": [0.0, 1.0], "speed_max_m_s": 4.4},
        "events": {"impacts_t": [0.45, 1.2, 1.8, 2.2]},
    }
    repo, experiment = h.build(
        "bounce",
        "B1",
        [h.quantity("e0", 0.9, 0.01), h.quantity("e1", 0.02, 0.1), h.quantity("g", 9.81, 0.01)],
        summary=summary,
        size_cm=4.0,
    )
    m = measure(repo, experiment)
    speeds = sorted(m.impact_speeds)
    median = (speeds[1] + speeds[2]) / 2 if len(speeds) == 4 else speeds[len(speeds) // 2]

    export = build_export(repo, experiment, "json", None)

    by_name = {p.name: p for p in export.parameters}
    assert list(by_name) == ["restitution", "restitution.e0", "restitution.e1"]
    assert by_name["restitution"].value == pytest.approx(
        conv.restitution_at_speed(0.9, 0.02, median)
    )
    low, high = by_name["restitution"].ci95
    assert low < by_name["restitution"].value < high
    assert export.valid_for["median_impact_speed_m_s"] == pytest.approx(median)
    assert any("restitution.e0" in n for n in export.valid_for["notes"])


def test_pendulum_json_adds_the_inertia_when_the_mass_is_known():
    repo, experiment = pendulum()

    with_mass = build_export(repo, experiment, "json", 50.0)
    without = build_export(repo, experiment, "json", None)

    (param,) = with_mass.parameters
    assert (param.name, param.unit) == ("joint.damping", "1/s")  # still a rate per unit of inertia
    assert with_mass.valid_for["geometry"]["inertia_kg_m2"] == pytest.approx(
        conv.pendulum_inertia(0.05, 0.62, 0.012)
    )
    assert "inertia_kg_m2" not in without.valid_for["geometry"]
    assert with_mass.valid_for["amplitude_deg"][1] == pytest.approx(math.degrees(0.66))


def test_a_spinner_exports_rates_per_unit_of_inertia_and_takes_an_inertia_if_given():
    repo, experiment = spinner("S2")

    plain = build_export(repo, experiment, "json", None)
    with_inertia = build_export(repo, experiment, "json", None, inertia_kg_m2=2.5e-5)

    assert [p.name for p in plain.parameters] == ["joint.damping", "joint.coulomb_torque"]
    assert [p.unit for p in plain.parameters] == ["1/s", "rad/s^2"]
    assert "geometry" not in plain.valid_for
    assert with_inertia.valid_for["geometry"]["inertia_kg_m2"] == 2.5e-5
    assert export_formats("S2") == ["json"]


def test_a_law_with_nothing_for_a_simulator_cannot_be_exported():
    repo, experiment = h.build("ramp", "R2", [h.quantity("beta", 0.4)], size_cm=4.0)
    wonder_repo, wonder = h.build("pendulum", "P2", [h.quantity("gamma", 0.1)], source="wonder")

    with pytest.raises(StageError) as raised:
        build_export(repo, experiment, "json", None)
    with pytest.raises(StageError):
        build_export(wonder_repo, wonder, "json", None)

    assert raised.value.code == "not_exportable"
    assert available_formats(repo, experiment) == []
    assert export_formats("R1") == ["json", "pybullet"]
    assert export_formats("P0") == []


def test_a_fit_that_lacks_the_parameter_cannot_be_exported():
    repo, experiment = h.build("ramp", "R1", [h.quantity("acceleration", 2.0)])

    with pytest.raises(StageError) as raised:
        build_export(repo, experiment, "json", None)

    assert raised.value.code == "missing_quantity"


def test_an_unknown_format_is_refused(ramp):
    repo, experiment = ramp

    with pytest.raises(StageError):
        build_export(repo, experiment, "mujoco", None)  # type: ignore[arg-type]


# --- the PyBullet snippet -----------------------------------------------------------------
def test_ramp_snippet_sets_the_friction_and_zeroes_the_default_damping(ramp):
    repo, experiment = ramp
    q = next(q for q in experiment.conclusion.quantities if q.name == "mu_k")

    text = build_export(repo, experiment, "pybullet", 200.0)

    ast.parse(text)
    assert f"FRICTION_KINETIC = {q.value:.6g}" in text
    assert "lateralFriction=FRICTION_KINETIC" in text
    assert "p.changeDynamics(board, -1, lateralFriction=1.0" in text  # the surface's value is 1
    assert "multiplies the friction" in text  # and the comment says why
    assert "linearDamping=0.0" in text and "angularDamping=0.0" in text
    assert "MASS_KG = 0.2  # as entered" in text
    assert "95% interval" in text and "RANDOMIZATION" in text
    assert "outside these ranges is extrapolation" in " ".join(text.split())  # wrapped


def test_bounce_snippet_sets_the_floor_to_one_and_explains_it(bounce):
    repo, experiment = bounce

    text = build_export(repo, experiment, "pybullet", None)

    ast.parse(text)
    assert "p.changeDynamics(floor, -1, restitution=1.0)" in text
    assert "multiplies the restitution" in text
    assert "restitution=RESTITUTION" in text
    assert "MASS_KG = 0.1  # placeholder; set your object's mass" in text


def test_every_supported_law_gives_a_snippet_that_parses():
    cases = [
        pendulum("P2", "gamma", 0.12),
        pendulum("P3", "kappa", 0.15),
        pendulum("P4", "mu_p", 0.25),
        h.build("free_fall", "FF1", [h.quantity("b", 0.8)], mover="foam ball", surface=None),
        h.build("free_fall", "FF2", [h.quantity("k", 0.35)], mover="foam ball", surface=None),
        h.build(
            "bounce",
            "B2",
            [h.quantity("e", 0.8), h.quantity("k", 0.02)],
            summary={"ranges": {"y": [0, 1.0]}, "events": {"impacts_t": [0.4, 1.0]}},
        ),
        h.build("ramp", "R3", [h.quantity("c_rr", 0.012)], size_cm=4.0),
    ]
    for repo, experiment in cases:
        ast.parse(build_export(repo, experiment, "pybullet", 50.0))


def test_free_fall_snippet_converts_the_rate_with_the_bullet_formula():
    repo, experiment = h.build(
        "free_fall", "FF1", [h.quantity("b", 0.8)], mover="foam ball", surface=None
    )

    text = build_export(repo, experiment, "pybullet", None)

    assert "linearDamping=1.0 - math.exp(-DAMPING_LINEAR_RATE)" in text
    assert "useMaximalCoordinates=True" in text  # the formula is exact for rigid bodies


def test_quadratic_drag_is_applied_as_a_force_every_step():
    repo, experiment = h.build(
        "free_fall", "FF2", [h.quantity("k", 0.35)], mover="foam ball", surface=None
    )

    text = build_export(repo, experiment, "pybullet", 50.0)

    assert "def apply_air_drag(body)" in text
    assert "p.applyExternalForce" in text
    assert "-MASS_KG * DRAG_QUADRATIC * speed * v" in text
    assert "MASS_KG = 0.05  # as entered" in text


def test_pendulum_snippet_converts_with_the_inertia_and_turns_the_default_motor_off():
    repo, experiment = pendulum("P2", "gamma", 0.12)

    text = build_export(repo, experiment, "pybullet", 50.0)

    assert "INERTIA_KG_M2 = MASS_KG * (PENDULUM_LENGTH_M**2 + 0.4 * BOB_RADIUS_M**2)" in text
    assert "jointDamping=JOINT_DAMPING * INERTIA_KG_M2" in text
    assert (
        "p.setJointMotorControl2(pendulum, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=0)"
        in text
    )
    assert "linearDamping=0.0" in text
    assert "MASS_KG = 0.05" in text and "PENDULUM_LENGTH_M = 0.62" in text


def test_joint_friction_is_a_velocity_motor_with_target_zero():
    repo, experiment = pendulum("P4", "mu_p", 0.25)

    text = build_export(repo, experiment, "pybullet", 50.0)

    assert "force=JOINT_COULOMB_TORQUE * INERTIA_KG_M2" in text
    assert "targetVelocity=0" in text
    assert "JOINT_COULOMB_TORQUE = 0.25" in text


def test_rolling_resistance_is_multiplied_by_the_ball_radius():
    repo, experiment = h.build("ramp", "R3", [h.quantity("c_rr", 0.012)], size_cm=4.0)

    text = build_export(repo, experiment, "pybullet", None)

    assert "rollingFriction=FRICTION_ROLLING * BALL_RADIUS_M" in text
    assert "BALL_RADIUS_M = 0.02" in text


def test_a_pendulum_snippet_needs_the_mass_and_a_spinner_snippet_needs_the_inertia():
    pend_repo, pend = pendulum()
    spin_repo, spin = spinner("S1")
    spin1 = h.build("spin_down", "S1", [h.quantity("c", 1.2)], mover="fidget spinner")

    with pytest.raises(StageError) as no_mass:
        build_export(pend_repo, pend, "pybullet", None)
    with pytest.raises(StageError) as no_inertia:
        build_export(spin_repo, spin, "pybullet", None)
    text = build_export(*spin1, "pybullet", None, inertia_kg_m2=2e-5)

    assert no_mass.value.code == "mass_required"
    assert no_inertia.value.code == "inertia_required"
    ast.parse(text)
    assert "INERTIA_KG_M2 = 2e-05" in text
    assert "force=JOINT_COULOMB_TORQUE * INERTIA_KG_M2" in text


def test_the_snippet_runs_when_pybullet_is_installed(ramp, bounce):
    pytest.importorskip("pybullet")
    namespace = {"__name__": "ghostline_export"}

    exec(compile(build_export(*ramp, "pybullet", None), "ramp", "exec"), namespace)  # noqa: S102
    accel = namespace["simulate"]()["acceleration"]
    exec(compile(build_export(*bounce, "pybullet", None), "bounce", "exec"), namespace)  # noqa: S102
    ratio = namespace["simulate"]()["height_ratio"]

    assert accel == pytest.approx(
        h.G * (math.sin(math.radians(25)) - 0.2 * math.cos(math.radians(25))), rel=0.02
    )
    assert ratio == pytest.approx(0.64, rel=0.05)
