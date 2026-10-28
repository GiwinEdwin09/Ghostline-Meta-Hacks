"""Scene analysis with Muse Wonder (build plan Part 6).

Wonder watches the clip and decides what to measure: the moving objects (with SAM prompts), the
calibration sheet, the kind of system, quality problems, and its initial hypotheses, stated
before any data exists.

    result = analyze_scene(meta, SceneInput(duration_s=8.0, capture="60 fps normal video",
                                            hint="a bouncing ball", video_file_id=file.id,
                                            stills=[start_png, middle_png, end_png]))
    if not result.analysis.proceed:
        raise SceneNotMeasurable(result.analysis)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import ghostline
from ghostline.llm import MetaClient, ModelResponse, input_image, input_text, input_video

from .schema import SceneAnalysis

PROMPT_PATH = Path(ghostline.__file__).parent / "prompts" / "scene.md"
STILL_LABELS = ["the start of the clip", "the middle of the clip", "the end of the clip"]
MAX_IMAGES = 50  # a request can carry up to 50 images


@dataclass
class SceneInput:
    """What Wonder sees.

    `video_file_id` is the 10 fps, 480p `wonder_480.mp4` uploaded to the Files API. The Day 0
    smoke test measured about 1,550 input tokens for 10 s of it ($0.006), so the whole video is
    sent. But the service samples video at roughly one frame per second, so fast motion (bounces,
    oscillations) falls between frames: the stills are how Wonder sees detail. With no video,
    pass 8 to 12 sampled frames as `stills` instead (a still costs about 500 tokens at 480p).
    `stills` are encoded images (JPEG or PNG bytes, or paths): full-resolution frames from the
    start, middle and end of the trim when a video is given, and `still_times_s` their times.
    """

    duration_s: float
    capture: str  # for example "60 fps normal video" or "240 fps slow-mo"
    hint: str = ""
    description: str = ""
    video_file_id: str | None = None
    stills: list[bytes | Path] = field(default_factory=list)
    still_times_s: list[float] = field(default_factory=list)


@dataclass
class SceneResult:
    analysis: SceneAnalysis
    response: ModelResponse
    prompt_version: str  # a hash of the prompt file, so results can be tied to a prompt version

    @property
    def cost_usd(self) -> float:
        return self.response.cost_usd


class SceneNotMeasurable(Exception):
    """Wonder set `proceed` to false. The message is Wonder's reason, safe to show the user."""

    def __init__(self, analysis: SceneAnalysis):
        reason = analysis.reason_if_not or "the clip can't be measured"
        tips = setup_tips(analysis)
        super().__init__(reason + ("\n\nTips:\n- " + "\n- ".join(tips) if tips else ""))
        self.analysis = analysis
        self.tips = tips


def load_prompt() -> tuple[str, str]:
    """(prompt text, version). The version is a short hash of the file."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode()).hexdigest()[:12]


def render_prompt(scene: SceneInput) -> str:
    """Fill the `{duration}`, `{capture}` and `{hint}` placeholders of the scene prompt."""
    text, _ = load_prompt()
    return (
        text.replace("{duration}", f"{scene.duration_s:g}")
        .replace("{capture}", scene.capture)
        .replace("{hint}", scene.hint.strip() or "nothing")
    )


def build_user_content(scene: SceneInput) -> list[dict]:
    """The user message parts: the user's description, the video, and the labelled stills."""
    parts = [
        input_text(
            f"The user describes the experiment as: {scene.description.strip() or 'no description'}"
        )
    ]
    if scene.video_file_id:
        parts.append(
            input_text(
                "Here is the whole clip. The service shows you only about one frame per second of "
                "it, so anything faster than that (bounces, oscillations, fast spins) can fall "
                "between the frames you see: don't conclude it is absent from the video alone."
            )
        )
        parts.append(input_video(scene.video_file_id))
    stills = scene.stills[:MAX_IMAGES]
    for i, still in enumerate(stills):
        if i < len(scene.still_times_s):
            label = f"An exact frame from the clip, at t = {scene.still_times_s[i]:.2f} s:"
        elif scene.video_file_id and i < len(STILL_LABELS):
            label = f"A full-resolution still from {STILL_LABELS[i]}:"
        else:
            label = f"Frame {i + 1} of {len(stills)} sampled across the clip:"
        parts.append(input_text(label))
        parts.append(input_image(still, "image/jpeg" if isinstance(still, bytes) else None))
    return parts


def analyze_scene(
    meta: MetaClient, scene: SceneInput, *, experiment_id: str | None = None
) -> SceneResult:
    """Ask Wonder to analyze the scene.

    Streaming, structured output with the `SceneAnalysis` schema, medium reasoning effort, and
    one repair call if the reply fails validation (which includes the SAM prompt rules). Uses
    `SCENE_MODEL`. Raises `ModelNotFoundError` if the model isn't available to this key. A reply
    with `proceed=false` is returned, not raised; call `require_measurable`.
    """
    if not scene.video_file_id and not scene.stills:
        raise ValueError("give Wonder a video file or at least one still")
    _, version = load_prompt()
    response = meta.respond(
        model=meta.settings.scene_model,
        instructions=render_prompt(scene),
        input=[{"type": "message", "role": "user", "content": build_user_content(scene)}],
        text_format=SceneAnalysis,
        reasoning_effort="medium",
        max_output_tokens=12000,
        experiment_id=experiment_id,
        phase="scene",
    )
    return SceneResult(response.parsed, response, version)


def require_measurable(result: SceneResult) -> SceneAnalysis:
    """The analysis, or `SceneNotMeasurable` when Wonder says the clip can't be measured."""
    if not result.analysis.proceed:
        raise SceneNotMeasurable(result.analysis)
    return result.analysis


# Part 3's capture rules, matched to the kinds of problems Wonder reports.
_TIPS: list[tuple[tuple[str, ...], str]] = [
    (("camera", "shake", "handheld", "moving camera", "tripod"),
     "Use a tripod and never film handheld."),
    (("sheet", "paper", "a4", "calibration"),
     "Tape a flat A4 sheet in the frame, fully visible the whole time, against a contrasting "
     "background."),
    (("blur", "fast", "frame rate", "fps"),
     "Use 60 fps, or 240 fps slow-mo for fast motion: free fall, bounces, fast spinners."),
    (("dark", "light", "flicker", "exposure"),
     "Film in daylight or bright, steady light; LED light causes flicker in slow-mo."),
    (("toward", "away", "depth", "plane", "perspective"),
     "Keep the motion in one plane parallel to the sheet, and stand at least 2 m back."),
    (("out of frame", "leaves", "cut off", "exits", "border"),
     "Keep the whole motion in frame; it should fill at least half the frame width."),
    (("occlu", "hidden", "behind"),
     "Make sure nothing blocks the object while it moves."),
    (("background", "contrast", "clutter"),
     "Put a dark board or cloth behind the sheet and the object, for contrast."),
]  # fmt: skip


def setup_tips(analysis: SceneAnalysis) -> list[str]:
    """Part 3 filming tips that match the quality issues (and the reason) Wonder reported."""
    text = " ".join(
        [analysis.reason_if_not or ""] + [f"{q.issue} {q.note}" for q in analysis.quality_issues]
    ).lower()
    tips = [tip for words, tip in _TIPS if any(w in text for w in words)]
    return tips or ["Follow the setup checklist: tripod, A4 sheet in frame, steady light, 60 fps."]
