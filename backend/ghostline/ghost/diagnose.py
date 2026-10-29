"""Wonder's diagnosis of a drifting ghost (build plan Part 17).

Wonder sees the error-versus-time plot, four frames around the onset with the ghost and the real
object both drawn, the model with its parameters and trial 1's summary. It answers with a
diagnosis, a revised model (a library model or custom code) and the signature it predicts.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel

import ghostline
from ghostline.llm import MetaClient, input_image, input_text, user_message
from ghostline.physics.harness.custom import CUSTOM_MODEL_CONTRACT
from ghostline.physics.library import catalog

from .model import Divergence, Forecast, FrozenFit
from .render import RenderPlan, render_stills
from .series import TrialSeries

PROMPT_PATH = Path(ghostline.__file__).parent / "prompts" / "diagnosis.md"
FRAME_OFFSETS_S = (-0.5, 0.0, 0.5, 1.0)  # around the onset
MAX_FRAME_WIDTH = 640


class ParamDef(BaseModel):
    name: str
    init: float
    lower: float
    upper: float
    unit: str
    kind: Literal["physical", "initial_condition", "nuisance"]


class RevisedModel(BaseModel):
    source: Literal["library", "custom"]
    library_ref: str  # "" for custom code
    id: str  # for custom code: a short name of letters and digits
    name: str
    mechanism: str
    equations_latex: str
    code: str  # "" for a library model
    state: list[str]  # custom code only
    params: list[ParamDef]  # custom code only


class DiagnosisReply(BaseModel):
    diagnosis: str  # one or two sentences naming the missing physics
    missing_physics: str  # a short label, such as "viscous damping"
    revised_model: RevisedModel
    predicted_signature: str  # what the prediction should look like if the diagnosis is right


def load_prompt() -> tuple[str, str]:
    """(prompt text, version): the version is a short hash of the file."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode()).hexdigest()[:12]


@dataclass
class DiagnosisInputs:
    """What Wonder is shown: a text brief and PNG/JPEG images, in the order the brief names them."""

    text: str
    images: list[bytes] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


Ask = Callable[[DiagnosisInputs], DiagnosisReply]


def error_plot_png(div: Divergence, t0: float, launch_end_rel: float, ghost_label: str) -> bytes:
    """Error against time, with the threshold bands, the launch window and the onset marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    t = div.t - t0
    ax.axvspan(0, launch_end_rel, color="#a78bfa", alpha=0.18, label="launch window (fitted)")
    ax.axhline(div.tau_mm, color="#22a55b", lw=1.2, ls="--", label=f"tau = {div.tau_mm:.1f} mm")
    ax.axhline(2 * div.tau_mm, color="#e0a100", lw=1.0, ls=":", label="2 tau")
    ax.plot(t, div.error_mm, color="#0a7ea4", lw=1.6, label=ghost_label)
    if div.onset_t is not None:
        ax.axvline(div.onset_t - t0, color="#d64545", lw=1.2, label="onset of drift")
    ax.set_xlabel("time since the start of the motion (s)")
    ax.set_ylabel("distance from the real object (mm)")
    ax.set_title("How far the ghost is from the real object")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", metadata={"Software": None})
    plt.close(fig)
    return buffer.getvalue()


def frame_indices(plan: RenderPlan, onset_rel: float | None, error: np.ndarray) -> list[int]:
    """Four plan indices around the onset (or the largest error, if the ghost never drifted)."""
    t = plan.t_rel
    centre = onset_rel if onset_rel is not None else float(t[int(np.argmax(error))])
    low = float(min(plan.launch_end_rel, t[-1]))
    picked: list[int] = []
    for offset in FRAME_OFFSETS_S:
        i = int(np.argmin(np.abs(t - float(np.clip(centre + offset, low, t[-1])))))
        if i not in picked:
            picked.append(i)
    for i in np.linspace(int(np.searchsorted(t, low)), len(t) - 1, 6).astype(int):
        if len(picked) >= 4:
            break
        if int(i) not in picked:
            picked.append(int(i))
    return sorted(picked)


def frame_jpegs(master: Path, plan: RenderPlan, indices: list[int]) -> dict[int, bytes]:
    out = {}
    for i, image in render_stills(master, plan, indices).items():
        scale = MAX_FRAME_WIDTH / image.shape[1]
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        out[i] = buffer.tobytes()
    return out


def _fmt(values: dict[str, float], sigma: dict[str, float] | None = None) -> str:
    parts = []
    for k, v in values.items():
        s = (sigma or {}).get(k)
        parts.append(f"{k} = {v:.5g}" + (f" +/- {s:.2g}" if s else ""))
    return "; ".join(parts) or "none"


def build_brief(
    *,
    system: str,
    model: dict,
    frozen: FrozenFit,
    forecast: Forecast,
    div: Divergence,
    trial1: TrialSeries,
    trial2: TrialSeries,
    t0: float,
    frame_times: list[float],
) -> str:
    """The text of the brief: the model and its numbers, trial 1's summary, the drift."""
    s1 = trial1.summary
    window = s1.get("motion_window", {})
    noise = s1.get("noise", {})
    onset = None if div.onset_t is None else div.onset_t - t0
    launch = forecast.launch_end_t - t0
    drift = (
        f"The ghost first stayed outside the threshold for five frames in a row at {onset:.2f} s, "
        f"{onset - launch:.2f} s after the launch window ended."
        if onset is not None
        else "The ghost never stayed outside the threshold for five frames in a row."
    )
    lines = [
        f"System: {system}.",
        f"The model you are diagnosing is {model['id']} ({model['name']}): {model['mechanism']}",
        f"Equations: {model['equations_latex']}",
        f"Its signature: {model['signature']}",
        "",
        "Frozen from trial 1 (statistical standard errors): "
        + _fmt(frozen.values, frozen.sigma)
        + f". Constants: {_fmt(frozen.constants)}. Trial 1 rms residual: "
        + (f"{frozen.train_rmse:.3g}." if frozen.train_rmse is not None else "unknown."),
        f"Trial 1: {window.get('t_end', 0) - window.get('t_start', 0):.2f} s of motion, "
        f"{window.get('n_points', '?')} points, position noise "
        f"{1000 * noise.get('sigma_pos_m', 0):.2f} mm, ranges {json.dumps(s1.get('ranges', {}))}.",
        f"Trial 2: {trial2.t_end - trial2.t_start:.2f} s of motion. Only the starting state "
        f"({_fmt(forecast.ic.values)}) was fitted, to the first {forecast.launch_window_s:.2f} s.",
        "",
        f"Threshold tau = {div.tau_mm:.2f} mm (the larger of 3 sigma_noise = "
        f"{3 * div.sigma_noise_mm:.2f} mm and half the object's radius = "
        f"{0.5 * div.radius_mm:.2f} mm). Largest error {div.max_error_mm:.1f} mm. {drift}",
        "",
        "Image 1 is the error plot. Images 2 to 5 are frames at "
        + ", ".join(f"{x:.2f} s" for x in frame_times)
        + " after the start of the motion: the cyan translucent object is the ghost, the white "
        "line the real object's trail, the dotted cyan line the predicted path, and the ghost's "
        "outline is green below tau, amber up to 2 tau, red beyond.",
        "",
        "Library models you may choose from (id, name, mechanism, what it predicts):",
    ]
    for entry in catalog(system):
        lines.append(f"- {entry['id']}: {entry['name']}. {entry['mechanism']} {entry['signature']}")
    lines += ["", "Custom model contract, if you must write code:", CUSTOM_MODEL_CONTRACT]
    return "\n".join(lines)


def ask_wonder(
    meta: MetaClient,
    inputs: DiagnosisInputs,
    *,
    experiment_id: str | None = None,
    repair: str | None = None,
) -> DiagnosisReply:
    """One structured Wonder call. `repair` carries validation errors from a first attempt."""
    text = inputs.text if repair is None else inputs.text + f"\n\nYour previous answer: {repair}"
    parts = []
    for image, label in zip(inputs.images, inputs.labels, strict=True):
        parts += [input_text(label), input_image(image, _mime(image))]
    response = meta.respond(
        model=meta.settings.physicist_model,
        instructions=load_prompt()[0],
        input=[user_message(text, *parts)],
        text_format=DiagnosisReply,
        reasoning_effort="medium",
        max_output_tokens=12000,
        experiment_id=experiment_id,
        phase="diagnosing",
    )
    return response.parsed


def _mime(image: bytes) -> str:
    return "image/png" if image[:4] == b"\x89PNG" else "image/jpeg"
