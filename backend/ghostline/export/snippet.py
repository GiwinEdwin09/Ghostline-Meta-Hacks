"""The PyBullet snippet (build plan Part 19).

`render_pybullet` writes a Python script that applies the measured numbers with `changeDynamics`
and `setJointMotorControl2` and then replays the experiment once, so it runs as it is after
`pip install pybullet`. The same text is what the replay check runs in the Sandbox: what the
user downloads is what was verified.

The script has two halves. Up to `apply_ghostline_dynamics` is meant to be pasted into the
user's own scene; `simulate` below it rebuilds the experiment (a block on a ramp, a ball drop, a
falling ball, a pendulum) from the measured geometry and reports what the simulation reproduces.
"""

from __future__ import annotations

import inspect
import re
import textwrap
from string import Template

from ghostline.jobs.errors import StageError

from . import replay_math
from .build import applicability_notes, exported_parameters, moment_of_inertia
from .conversions import BULLET_DEFAULT_DAMPING
from .expectations import PENDULUM_PARAM, TIMESTEP_HZ, pendulum_duration_s, restitution_triplet
from .mapping import REPLAY_SCENARIOS
from .measured import Measurement

PLACEHOLDER_MASS_KG = 0.1  # friction, restitution and linear damping don't depend on mass
REFERENCE_MASS_KG = 1.0  # the replay's pendulum: rates per unit of inertia make the mass moot
MAX_RUN_S = 30.0

_HEADER = Template('''\
"""Ghostline simulator export for PyBullet.

Object:   $object
Surface:  $surface
Source:   $url   (confidence: $confidence)

Every value below was measured by filming the real thing, with its 95% interval beside it.
$notes

Run this file as it is (pip install pybullet) to replay the experiment: it rebuilds the setup
from the measured geometry, steps PyBullet at 240 Hz and prints what the simulation reproduces.
The part down to apply_ghostline_dynamics() is the part to paste into your own scene.
"""
$imports

DT = 1.0 / $hz  # the replay check steps at $hz Hz
''')

_RAMP = Template('''\
GRAVITY = $gravity  # m/s^2, the value the ramp fit used
RAMP_ANGLE_DEG = $angle  # the ramp's measured angle
BLOCK_HALF_SIZE_M = $half  # half the block's side
MASS_KG = $mass  # $mass_note


def apply_ghostline_dynamics(block, board):
    """Give the block its measured sliding friction. Call once, after creating both bodies."""
    # Bullet multiplies the friction values of the two bodies in contact. The measurement is for
    # the pair (this object on this surface), so the object carries the measured value and the
    # surface is set to 1, which leaves the product unchanged.
    p.changeDynamics(board, -1, lateralFriction=1.0, restitution=0.0)
    p.changeDynamics(
        block,
        -1,
        lateralFriction=FRICTION_KINETIC,
        # PyBullet gives every body $damping of linear and angular damping. Zero it, so the
        # block loses energy only to the friction that was measured.
        linearDamping=0.0,
        angularDamping=0.0,
        rollingFriction=0.0,
        spinningFriction=0.0,
        restitution=0.0,
    )


def simulate():
    """Slide the block down a ramp at the measured angle; return its acceleration (m/s^2)."""
    p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -GRAVITY)
        p.setTimeStep(DT)
        alpha = math.radians(RAMP_ANGLE_DEG)
        tilt = p.getQuaternionFromEuler([0.0, alpha, 0.0])
        down = [math.cos(alpha), 0.0, -math.sin(alpha)]  # along the slope, downhill
        up_out = [math.sin(alpha), 0.0, math.cos(alpha)]  # out of the slope
        board = p.createMultiBody(
            0,
            p.createCollisionShape(p.GEOM_BOX, halfExtents=[4.0, 1.0, 0.02]),
            baseOrientation=tilt,
        )
        lift = 0.02 + BLOCK_HALF_SIZE_M
        start = [up_out[i] * lift - 2.0 * down[i] for i in range(3)]
        block = p.createMultiBody(
            MASS_KG,
            p.createCollisionShape(p.GEOM_BOX, halfExtents=[BLOCK_HALF_SIZE_M] * 3),
            basePosition=start,
            baseOrientation=tilt,
        )
        apply_ghostline_dynamics(block, board)
        speeds = []
        for _ in range(int(1.5 / DT) + 1):
            p.stepSimulation()
            velocity = p.getBaseVelocity(block)[0]
            speeds.append(sum(velocity[i] * down[i] for i in range(3)))
        # Skip the first 0.6 s: the block settles onto the board before it slides steadily.
        first, last = int(0.6 / DT), int(1.5 / DT)
        return {"acceleration": (speeds[last] - speeds[first]) / ((last - first) * DT)}
    finally:
        p.disconnect()
''')

_ROLLING = Template('''\
BALL_RADIUS_M = $half  # the ball's radius


def apply_ghostline_dynamics(ball, surface):
    """Give the ball its measured rolling resistance. Call once, after creating both bodies."""
    # Bullet multiplies the friction values of the two bodies in contact, so the surface is set
    # to 1 and the ball carries the whole value.
    p.changeDynamics(surface, -1, lateralFriction=1.0)
    p.changeDynamics(
        ball,
        -1,
        # Bullet's rollingFriction is a torque per unit of normal force, so it has the units of a
        # length: the dimensionless c_rr times the ball's radius.
        rollingFriction=FRICTION_ROLLING * BALL_RADIUS_M,
        # The ramp fit assumed the ball rolls without slipping: keep sliding friction high.
        lateralFriction=1.0,
        linearDamping=0.0,  # PyBullet's default is $damping
        angularDamping=0.0,
        spinningFriction=0.0,
        restitution=0.0,
    )
''')

_BOUNCE = Template('''\
GRAVITY = $gravity  # m/s^2
DROP_HEIGHT_M = $height  # the measured drop height of the ball's centre
BALL_RADIUS_M = $half  # $radius_note
MASS_KG = $mass  # $mass_note
DURATION_S = $duration  # long enough for the first bounces


def apply_ghostline_dynamics(ball, floor):
    """Give the ball its measured restitution. Call once, after creating both bodies."""
    # Bullet multiplies the restitution of the two bodies in contact. The measurement is for the
    # pair (this ball on this floor), so the ball carries the measured value and the floor is
    # set to 1, which leaves the product unchanged.
    p.changeDynamics(floor, -1, restitution=1.0)
    # Bullet's contact error reduction pushes a penetrating ball back out and adds energy to
    # every bounce. Zero it, so the restitution alone sets how much speed the ball keeps.
    p.setPhysicsEngineParameter(contactERP=0.0)
    p.changeDynamics(
        ball,
        -1,
        restitution=RESTITUTION,
        # PyBullet gives every body $damping of linear and angular damping. Zero it, so the ball
        # loses energy only in the bounces.
        linearDamping=0.0,
        angularDamping=0.0,
        rollingFriction=0.0,
        spinningFriction=0.0,
    )
$drag

def simulate():
    """Drop the ball from the measured height; return the ratio of successive bounce heights."""
    p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -GRAVITY)
        p.setTimeStep(DT)
        # Rigid bodies (maximal coordinates): the contact formulas used here are exact for them.
        # PyBullet's default reduced-coordinate bodies follow different ones.
        floor = p.createMultiBody(
            0, p.createCollisionShape(p.GEOM_PLANE), useMaximalCoordinates=True
        )
        ball = p.createMultiBody(
            MASS_KG,
            p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS_M),
            basePosition=[0.0, 0.0, DROP_HEIGHT_M + BALL_RADIUS_M],
            useMaximalCoordinates=True,
        )
        apply_ghostline_dynamics(ball, floor)
        heights_of_centre = []
        for _ in range(int(DURATION_S / DT)):
            $step_drag
            p.stepSimulation()
            heights_of_centre.append(p.getBasePositionAndOrientation(ball)[0][2])
        apexes = rebound_heights(heights_of_centre)
        return {"height_ratio": geometric_ratio(apexes), "bounce_heights_m": apexes[:3]}
    finally:
        p.disconnect()
''')

_FALL = Template('''\
GRAVITY = $gravity  # m/s^2
BALL_RADIUS_M = $half  # $radius_note
MASS_KG = $mass  # $mass_note
DURATION_S = $duration  # long enough for the speed to near its limit


def apply_ghostline_dynamics(ball):
    """Give the ball its measured air drag. Call once, after creating it."""
$fall_apply

$drag
def simulate():
    """Let the ball fall in the air; return the terminal velocity the drag leads to (m/s)."""
    p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -GRAVITY)
        p.setTimeStep(DT)
        # A rigid body (maximal coordinates): the damping formula in apply_ghostline_dynamics()
        # is exact for it. PyBullet's default reduced-coordinate bodies damp by a different law.
        ball = p.createMultiBody(
            MASS_KG,
            p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS_M),
            basePosition=[0.0, 0.0, 0.0],
            useMaximalCoordinates=True,
        )
        apply_ghostline_dynamics(ball)
        for _ in range(int(DURATION_S / DT)):
            $step_drag
            p.stepSimulation()
        speed = -p.getBaseVelocity(ball)[0][2]
        return {"terminal_velocity": speed / $approach}
    finally:
        p.disconnect()
''')

_PENDULUM = Template('''\
GRAVITY = $gravity  # m/s^2, as fitted from this video
PENDULUM_LENGTH_M = $length  # pivot to the bob's centre, measured
BOB_RADIUS_M = $half  # the bob's radius
MASS_KG = $mass  # $mass_note
AMPLITUDE_RAD = $amplitude  # the starting angle of the measured swing
DURATION_S = $duration  # long enough to see the swing fall to half

# Moment of inertia about the pivot: a point mass at the length plus the bob's own spin.
INERTIA_KG_M2 = MASS_KG * (PENDULUM_LENGTH_M**2 + 0.4 * BOB_RADIUS_M**2)


def create_pendulum():
    """A bob of MASS_KG on a revolute joint, PENDULUM_LENGTH_M from the pivot."""
    bob_inertia = 0.4 * MASS_KG * BOB_RADIUS_M**2  # solid sphere about its own centre
    urdf = f"""<?xml version="1.0"?>
<robot name="ghostline_pendulum">
  <link name="pivot">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <link name="bob">
    <inertial>
      <origin xyz="0 0 -{PENDULUM_LENGTH_M}"/>
      <mass value="{MASS_KG}"/>
      <inertia ixx="{bob_inertia}" ixy="0" ixz="0" iyy="{bob_inertia}" iyz="0" izz="{bob_inertia}"/>
    </inertial>
  </link>
  <joint name="swing" type="revolute">
    <parent link="pivot"/>
    <child link="bob"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="-1" effort="1000" velocity="100"/>
    <dynamics damping="0" friction="0"/>
  </joint>
</robot>
"""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "pendulum.urdf")
        with open(path, "w") as handle:
            handle.write(urdf)
        return p.loadURDF(path, useFixedBase=True)


def apply_ghostline_dynamics(pendulum):
    """Give the pendulum its measured damping. Call once, after creating it."""
    # PyBullet gives every body $damping of linear and angular damping, and every revolute
    # joint starts with a velocity motor that holds it still. Switch both off, so the swing
    # loses energy only where the measurement says it does.
    p.changeDynamics(pendulum, -1, linearDamping=0.0, angularDamping=0.0)
    p.setJointMotorControl2(pendulum, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
    # The measured value is a rate per unit of inertia; PyBullet wants absolute torques.
$joint_apply
$drag
def simulate():
    """Swing the pendulum from the measured amplitude; return its half-life and period."""
    p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -GRAVITY)
        p.setTimeStep(DT)
        pendulum = create_pendulum()
        apply_ghostline_dynamics(pendulum)
        p.resetJointState(pendulum, 0, AMPLITUDE_RAD)
        times, angles = [], []
        for step in range(int(round(DURATION_S / DT)) + 1):
            times.append(step * DT)
            angles.append(p.getJointState(pendulum, 0)[0])
            $step_drag
            p.stepSimulation()
        return {"half_life_s": half_life(times, angles), "period_s": mean_period(times, angles)}
    finally:
        p.disconnect()
''')

_SPINNER = Template('''\
INERTIA_KG_M2 = $inertia  # the spinner's moment of inertia, as entered
JOINT_AXIS = (0, 0, 1)  # the axis the spinner turns about, in its link frame; change to yours


def apply_ghostline_dynamics(body, joint=0):
    """Give the spinner's joint its measured friction. Call once, after creating the body."""
    # PyBullet gives every body $damping of linear and angular damping, and every revolute
    # joint starts with a velocity motor that holds it still. Switch both off, so the spinner
    # loses energy only where the measurement says it does.
    p.changeDynamics(body, -1, linearDamping=0.0, angularDamping=0.0)
    p.setJointMotorControl2(body, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
    # The measured values are rates per unit of inertia; PyBullet wants absolute torques.
$joint_apply
$drag''')

_TAIL = Template('''
# Ranges to randomize over in training: the 95% intervals above, in the same units. They cover
# measurement uncertainty only; part-to-part variation needs wider ranges.
RANDOMIZATION = {
$randomization
}

$main
''')

_MAIN_REPLAY = '''if __name__ == "__main__":
    print(simulate())'''
_MAIN_APPLY = '''if __name__ == "__main__":
    print("Call apply_ghostline_dynamics() on your bodies after creating them.")'''

_DRAG_BODY = Template('''

def apply_air_drag($args):
    """$doc"""
$body
''')


def _num(x: float) -> str:
    return f"{x:.6g}"


def _first_line(text: str) -> str:
    return text.split(". ")[0].rstrip(".")


def _constants(m: Measurement) -> str:
    lines = []
    for param in exported_parameters(m):
        name = param.name.upper().replace(".", "_")
        unit = f" ({param.unit})" if param.unit else ""
        lines.append(
            f"# {_first_line(param.definition)}{unit}\n"
            f"# 95% interval [{_num(param.ci95[0])}, {_num(param.ci95[1])}]\n"
            f"{name} = {_num(param.value)}"
        )
    return "\n".join(lines)


def _randomization(m: Measurement) -> str:
    return "\n".join(
        f'    "{p.name}": ({_num(p.ci95[0])}, {_num(p.ci95[1])}),' for p in exported_parameters(m)
    )


def _helpers(*functions) -> str:
    return "\n\n".join(inspect.getsource(f).rstrip() for f in functions)


def _resolve_mass(
    m: Measurement, mass_kg: float | None, inertia_kg_m2: float | None, replay: bool
) -> tuple[float | None, str]:
    """The mass to write into the script, and the comment that says where it came from."""
    if m.system == "spin_down":
        if not inertia_kg_m2:
            raise StageError(
                "A spinner's mass distribution can't be seen on video. Enter its moment of "
                "inertia to export it to PyBullet; the JSON export has the rates as they are.",
                code="inertia_required",
            )
        return None, ""
    if m.system == "pendulum":
        if mass_kg:
            return mass_kg, "as entered"
        if replay:
            return REFERENCE_MASS_KG, "reference mass: the replay uses rates per unit of inertia"
        raise StageError(
            "Enter the bob's mass in grams to export the pendulum to PyBullet.",
            code="mass_required",
        )
    if mass_kg:
        return mass_kg, "as entered"
    return PLACEHOLDER_MASS_KG, "placeholder; set your object's mass"


def _radius_note(m: Measurement) -> str:
    return "from the scene analysis" if m.half_size_known else "assumed; set your ball's radius"


def _drag_function(m: Measurement, kind: str) -> tuple[str, str]:
    """(the apply_air_drag function, the call to put in the step loop) for quadratic drag."""
    if kind == "force":  # a body falling or bouncing
        body = _DRAG_BODY.substitute(
            args="body",
            doc=(
                "Quadratic drag has no PyBullet field: apply the force every step, before "
                "p.stepSimulation()."
            ),
            body=(
                "    velocity = p.getBaseVelocity(body)[0]\n"
                "    speed = math.sqrt(sum(v * v for v in velocity))\n"
                "    force = [-MASS_KG * DRAG_QUADRATIC * speed * v for v in velocity]\n"
                "    position = p.getBasePositionAndOrientation(body)[0]\n"
                "    p.applyExternalForce(body, -1, force, position, p.WORLD_FRAME)"
            ),
        )
        return body, "apply_air_drag(ball)"
    body = _DRAG_BODY.substitute(
        args="pendulum",
        doc=(
            "Quadratic drag has no PyBullet field: apply the torque every step, before "
            "p.stepSimulation()."
        ),
        body=(
            "    omega = p.getJointState(pendulum, 0)[1]\n"
            "    torque = -INERTIA_KG_M2 * DRAG_QUADRATIC * abs(omega) * omega\n"
            "    p.setJointMotorControl2(pendulum, 0, p.TORQUE_CONTROL, force=torque)"
        ),
    )
    return body, "apply_air_drag(pendulum)"


def _bounce_duration(m: Measurement, e: float, height: float) -> float:
    fall = (2 * height / m.gravity) ** 0.5
    bounces = sum(e**k for k in range(1, 6))
    return round(min(max(fall * (1 + 2 * bounces) + 0.3, 1.0), 20.0), 3)


def _join(lines: list[str], indent: int) -> str:
    return ("\n" + " " * indent).join(lines)


def _scenario(m: Measurement, mass: float | None, mass_note: str, inertia: float | None) -> str:
    """Everything after the constants: the apply function and, where there is one, simulate()."""
    damping = f"{BULLET_DEFAULT_DAMPING:g}"
    model = m.model_id
    if m.system == "ramp" and model == "R1":
        return _RAMP.substitute(
            gravity=_num(m.gravity),
            angle=_num(m.geometry["angle_deg"]),
            half=_num(m.half_size_m),
            mass=_num(mass or PLACEHOLDER_MASS_KG),
            mass_note=mass_note,
            damping=damping,
        )
    if model == "R3":
        return _ROLLING.substitute(half=_num(m.half_size_m), damping=damping)
    if m.system == "bounce":
        height = m.geometry.get("drop_height_m")
        if not height:
            raise StageError("The drop height was not measured.", code="missing_geometry")
        e = restitution_triplet(m)[0]
        drag, step = "", ""
        if model == "B2":
            drag, step = _drag_function(m, "force")
        return _BOUNCE.substitute(
            gravity=_num(m.gravity),
            height=_num(height),
            half=_num(m.half_size_m),
            radius_note=_radius_note(m),
            mass=_num(mass or PLACEHOLDER_MASS_KG),
            mass_note=mass_note,
            duration=_num(_bounce_duration(m, e, height)),
            damping=damping,
            drag=drag,
            step_drag=step,
        )
    if m.system == "free_fall":
        rate_name = "DAMPING_LINEAR_RATE" if model == "FF1" else "DRAG_QUADRATIC"
        rate_q = m.quantity("b" if model == "FF1" else "k")
        rate = rate_q.value if rate_q else 1.0
        if model == "FF1":
            duration = min(8.0 / rate, MAX_RUN_S)
            approach = f"(1.0 - math.exp(-{rate_name} * DURATION_S))"
            apply_lines = [
                "    # Bullet scales a rigid body's velocity by (1 - linearDamping) ** dt every step,",
                "    # so a measured decay rate lambda needs linearDamping = 1 - exp(-lambda).",
                "    p.changeDynamics(",
                "        ball, -1, linearDamping=1.0 - math.exp(-DAMPING_LINEAR_RATE),",
                "        angularDamping=0.0,  # PyBullet's default is " + damping,
                "    )",
            ]
            drag, step = "", ""
        else:
            duration = min(6.0 / (m.gravity * rate) ** 0.5, MAX_RUN_S)
            approach = f"math.tanh(math.sqrt(GRAVITY * {rate_name}) * DURATION_S)"
            apply_lines = [
                "    # PyBullet gives every body " + damping + " of linear and angular damping.",
                "    # Zero it: the drag below is the only loss the measurement asks for.",
                "    p.changeDynamics(ball, -1, linearDamping=0.0, angularDamping=0.0)",
            ]
            drag, step = _drag_function(m, "force")
        return _FALL.substitute(
            gravity=_num(m.gravity),
            half=_num(m.half_size_m),
            radius_note=_radius_note(m),
            mass=_num(mass or PLACEHOLDER_MASS_KG),
            mass_note=mass_note,
            duration=_num(round(duration, 3)),
            fall_apply="\n".join(apply_lines),
            drag=drag,
            step_drag=step,
            approach=approach,
        )
    if m.system == "pendulum":
        length = m.geometry.get("length_m")
        if not length:
            raise StageError("The pendulum's length was not measured.", code="missing_geometry")
        param = PENDULUM_PARAM[model]
        joint = {
            "gamma": [
                "    p.changeDynamics(pendulum, 0, jointDamping=JOINT_DAMPING * INERTIA_KG_M2)",
            ],
            "mu_p": [
                "    # Dry friction: a velocity motor with target 0 and a maximum force pushes",
                "    # against any motion with up to that torque, and holds the joint when the",
                "    # load is below it.",
                "    p.setJointMotorControl2(",
                "        pendulum, 0, p.VELOCITY_CONTROL, targetVelocity=0,",
                "        force=JOINT_COULOMB_TORQUE * INERTIA_KG_M2,",
                "    )",
            ],
            "kappa": [
                "    # Quadratic drag has no field of its own: see apply_air_drag() below.",
            ],
        }[param]
        drag, step = "", ""
        if model == "P3":
            drag, step = _drag_function(m, "torque")
        return _PENDULUM.substitute(
            gravity=_num(m.gravity),
            length=_num(length),
            half=_num(m.geometry.get("bob_radius_m", m.half_size_m)),
            mass=_num(mass or REFERENCE_MASS_KG),
            mass_note=mass_note,
            amplitude=_num(m.geometry.get("amplitude_rad", 0.5)),
            duration=_num(round(pendulum_duration_s(m), 3)),
            damping=damping,
            joint_apply="\n".join(joint),
            drag=drag,
            step_drag=step,
        )
    # A spinner.
    apply_lines = []
    if model in ("S0", "S2"):
        apply_lines.append(
            "    p.changeDynamics(body, joint, jointDamping=JOINT_DAMPING * INERTIA_KG_M2)"
        )
    if model in ("S1", "S2", "S3"):
        apply_lines += [
            "    # Dry friction: a velocity motor with target 0 and a maximum force.",
            "    p.setJointMotorControl2(",
            "        body, joint, p.VELOCITY_CONTROL, targetVelocity=0,",
            "        force=JOINT_COULOMB_TORQUE * INERTIA_KG_M2,",
            "    )",
        ]
    drag = ""
    if model == "S3":
        drag = (
            "\n\ndef apply_air_drag(body, joint=0):\n"
            '    """Quadratic drag has no PyBullet field: apply the torque every step."""\n'
            "    omega = p.getJointState(body, joint)[1]\n"
            "    torque = -INERTIA_KG_M2 * DRAG_QUADRATIC * omega * abs(omega)\n"
            "    p.applyExternalTorque(\n"
            "        body, joint, [c * torque for c in JOINT_AXIS], p.LINK_FRAME\n"
            "    )\n"
        )
    return _SPINNER.substitute(
        inertia=_num(inertia or 0.0),
        damping=damping,
        joint_apply="\n".join(apply_lines),
        drag=drag,
    )


def render_pybullet(
    m: Measurement,
    mass_kg: float | None = None,
    inertia_kg_m2: float | None = None,
    *,
    replay: bool = False,
) -> str:
    """The PyBullet script for the winner. `replay=True` allows a reference mass for a pendulum,
    for the replay check, which asks no one for it."""
    mass, mass_note = _resolve_mass(m, mass_kg, inertia_kg_m2, replay)
    inertia = moment_of_inertia(m, mass, inertia_kg_m2)
    scenario = REPLAY_SCENARIOS.get(m.model_id)
    needs = {
        "ramp": ["math"],
        "bounce": ["math"],
        "free_fall": ["math"],
        "pendulum": ["math", "os", "tempfile"],
    }.get(scenario or "", ["math"])
    if m.model_id == "R3" or m.system == "spin_down":
        needs = ["math"]
    imports = "\n".join(f"import {name}" for name in sorted(needs)) + "\n\nimport pybullet as p"
    helpers = {
        "ramp": (),
        "bounce": (replay_math.rebound_heights, replay_math.geometric_ratio),
        "free_fall": (),
        "pendulum": (replay_math.half_life, replay_math.mean_period),
    }.get(scenario or "", ())

    notes = "\n".join(
        textwrap.fill(n, width=94, initial_indent="- ", subsequent_indent="  ")
        for n in applicability_notes(m)
    )
    parts = [
        _HEADER.substitute(
            object=m.object_name,
            surface=m.surface_name or "(not identified)",
            url=m.experiment_url(),
            confidence=m.experiment.conclusion.confidence if m.experiment.conclusion else "low",
            notes=notes,
            imports=imports,
            hz=TIMESTEP_HZ,
        ),
        "# --- What Ghostline measured " + "-" * 49,
        _constants(m),
    ]
    scenario_text = _scenario(m, mass, mass_note, inertia)
    parts.append(scenario_text)
    if helpers:
        parts.append("\n# --- Measurements on the simulated motion " + "-" * 36)
        parts.append(_helpers(*helpers))
    parts.append(
        _TAIL.substitute(
            randomization=_randomization(m),
            main=_MAIN_REPLAY if scenario else _MAIN_APPLY,
        )
    )
    text = "\n".join(parts).rstrip() + "\n"
    text = re.sub(r"(?m)^[ \t]+$", "", text)
    return re.sub(r"\n{4,}", "\n\n\n", text)
