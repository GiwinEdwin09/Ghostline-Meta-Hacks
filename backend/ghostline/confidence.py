"""Confidence and "can't tell" (build plan Part 15).

Deterministic rules over the held-out selection, the residual check, the data-quality flags and
the reviewer's status. Low beats medium beats high: one low reason is enough for low.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .physics.select import Selection

Level = Literal["high", "medium", "low"]
ReviewStatus = Literal["accepted", "accepted_with_caveats", "contested", "not_reviewed"]

MAJOR_FLAGS = {
    "out_of_plane": "The object moved toward or away from the camera.",
    "camera_motion": "The camera moved during the clip.",
    "time_base_uncertain": "The video's timing is uncertain (re-timed or variable frame rate).",
    "few_periods": "The clip holds fewer than two periods of motion.",
    "short_duration": "The clip holds less than 0.5 s of motion.",
}
MIN_PERIODS = 2.0
MIN_DURATION_S = 0.5

FIXES = {
    "out_of_plane": "Film square-on, keep the object in the plane of the sheet, and stand further "
    "back with the 2x lens.",
    "camera_motion": "Put the phone on a tripod or rest it on something rigid and don't touch it "
    "while recording.",
    "time_base_uncertain": "Film at normal speed (60 fps) or upload the original file instead of "
    "an edited or re-timed export.",
    "few_periods": "Record at least two full periods: start recording earlier and stop later, or "
    "use a longer string.",
    "short_duration": "Start recording before the release and stop after the motion ends, so the "
    "clip holds at least 0.5 s of motion.",
    "contested": "Repeat the experiment with a cleaner setup, then run it again.",
    "systematic_residual": "Look at the residual plot: the fit leaves a pattern, so some physics "
    "is missing. Try a longer clip or a different starting condition.",
}
TIE_FIXES = {
    "pendulum": "Increase the swing amplitude (about 40 degrees) so the models separate.",
    "free_fall": "Film in 240 fps slow-mo and drop from higher, so drag has time to show.",
    "projectile": "Film in 240 fps slow-mo and throw faster, so drag has time to show.",
    "ramp": "Use a longer ramp, or film the same object at a second angle.",
    "bounce": "Film at 240 fps and record more bounces.",
    "spin_down": "Film the whole spin-down, from the flick until the spinner stops.",
    "oscillator": "Record more cycles so the decay is visible.",
}


class QualityFlag(BaseModel):
    code: str
    severity: Literal["minor", "major"] = "minor"
    note: str = ""

    @classmethod
    def major(cls, code: str, note: str = "") -> QualityFlag:
        return cls(code=code, severity="major", note=note or MAJOR_FLAGS.get(code, ""))


class Reason(BaseModel):
    code: str
    level: Level
    text: str


class ConfidenceAssessment(BaseModel):
    level: Level
    cant_tell: bool
    reasons: list[Reason] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)

    @property
    def codes(self) -> set[str]:
        return {r.code for r in self.reasons}


def motion_extent_flags(duration_s: float, period_s: float | None = None) -> list[QualityFlag]:
    """Major flags for a clip with too little motion (under 0.5 s, or under two periods)."""
    flags = []
    if duration_s < MIN_DURATION_S:
        flags.append(
            QualityFlag.major("short_duration", f"Only {duration_s:.2f} s of motion was measured.")
        )
    if period_s and duration_s / period_s < MIN_PERIODS:
        flags.append(
            QualityFlag.major(
                "few_periods", f"The clip holds {duration_s / period_s:.1f} periods of motion."
            )
        )
    return flags


def assess_confidence(
    selection: Selection,
    *,
    system: str = "",
    flags: list[QualityFlag] | None = None,
    review: ReviewStatus | None = None,
    systematic_residual: bool | None = None,
) -> ConfidenceAssessment:
    """Turn the evidence into high, medium or low, with reasons and what to change.

    `review=None` means no reviewer ran, which caps confidence at medium. `systematic_residual`
    defaults to the winner's full-data residual check from the selection.
    """
    flags = flags or []
    reasons: list[Reason] = []

    def add(code: str, level: Level, text: str) -> None:
        reasons.append(Reason(code=code, level=level, text=text))

    if selection.status == "inconclusive" or selection.selected_id is None:
        add("inconclusive", "low", selection.message or "The held-out splits disagree.")
    elif selection.status == "tied":
        if selection.tied_equal_or_simpler or not selection.tied_with:
            add("tied", "low", selection.message)
        else:
            add("parsimony", "medium", selection.message)
    elif selection.splits_won < selection.n_splits or selection.n_splits < 3:
        add(
            "partial_win",
            "medium",
            f"The winner beat its runner-up in {selection.splits_won} of "
            f"{selection.n_splits} splits.",
        )

    if systematic_residual is None and selection.selected_id:
        diag = selection.score(selection.selected_id).full_diagnostics
        systematic_residual = bool(diag and diag.systematic)
    if systematic_residual:
        add("systematic_residual", "medium", "The winner's residuals still show structure.")

    for flag in flags:
        if flag.severity == "major":
            add(flag.code, "low", flag.note or MAJOR_FLAGS.get(flag.code, flag.code))
        else:
            add(flag.code, "medium", flag.note or flag.code)

    if review == "contested":
        add("contested", "low", "The reviewer still challenges the claim after two rounds.")
    elif review == "accepted_with_caveats":
        add("review_caveats", "medium", "The reviewer accepted with caveats.")
    elif review in (None, "not_reviewed"):
        add("not_reviewed", "medium", "No independent review was available.")

    levels = {r.level for r in reasons}
    level: Level = "low" if "low" in levels else "medium" if "medium" in levels else "high"
    if level == "high":
        add("all_clear", "high", "The winner beat its runner-up in every split and passed review.")
    return ConfidenceAssessment(
        level=level,
        cant_tell=level == "low",
        reasons=reasons,
        fixes=_fixes(system, reasons) if level != "high" else [],
    )


def _fixes(system: str, reasons: list[Reason]) -> list[str]:
    out: list[str] = []
    for reason in reasons:
        if reason.code in ("tied", "inconclusive", "parsimony", "partial_win"):
            out.append(TIE_FIXES.get(system, "Film a longer clip so the models separate."))
            if reason.code in ("tied", "inconclusive"):
                out.append("Film a longer clip so the data can tell the models apart.")
        elif reason.code in FIXES:
            out.append(FIXES[reason.code])
    return list(dict.fromkeys(out))
