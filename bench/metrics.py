"""Scoring a benchmark clip against its ground truth (build plan Part 25).

The inputs are what the API stores: the experiment's state (GET /experiments/{id}) and its
notebook events. Nothing here knows how the pipeline got there.

    error, z               z = (measured - truth) / sqrt(sigma_measured^2 + sigma_truth^2)
    coverage               the share of scored clips whose 95% interval contains the truth
    model selection        the winner is one of the clip's expected library models
    failure cases          low confidence, and the reason found is the one the clip was made for
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from bench.truth import (
    ManifestRow,
    Truth,
    interval_contains,
    sigma_from_interval,
    wilson_interval,
    z_score,
)

# Statuses of a clip that was never run, which no metric counts for or against.
NOT_RUN = {"pending", "missing_file"}

# Phrases (and the machine codes themselves) that say a clip was marked down for a given reason.
FAILURE_PATTERNS = {
    "out_of_plane": r"out[\s_-]*of[\s_-]*plane|(toward|towards) or away from the camera|"
    r"motion in depth|moved in depth",
    "camera_motion": r"camera[\s_-]*(motion|movement|moved|moving|shak\w*|shook)|hand[\s-]?held|"
    r"shaky",
    "time_base_uncertain": r"re-?timed|time[\s_-]*base|variable[\s_-]*frame[\s_-]*rate|\bvfr\b|"
    r"timing (is )?uncertain|speed[\s_-]*ramp|frame timing",
    "few_periods": r"(fewer|less) than two periods|periods of motion|few[\s_-]*periods",
    "short_duration": r"less than 0\.5 ?s|short[\s_-]*duration|only [\d.]+ ?s of motion",
}
_NEGATION = re.compile(r"\b(no|not|without|never|nor)\b[^.;\n]{0,20}$|n't\b[^.;\n]{0,20}$", re.I)

# Library names the conclusion may use for the quantity a clip is scored on.
_ALIASES = {"e0": "e", "mu": "muk", "muk": "muk", "beta": "beta"}


def normalize_name(name: str) -> str:
    """'mu_k', 'μ_k', 'mu k' -> 'muk'; 'g (video L)' -> 'g'; 'g_tape' -> 'gtape'."""
    text = name.strip().lower().replace("μ", "mu").replace("β", "beta")
    token = re.split(r"[\s(\[]", text, maxsplit=1)[0]
    token = re.sub(r"[^a-z0-9]", "", token)
    return _ALIASES.get(token, token)


def find_quantity(conclusion: Mapping[str, Any] | None, quantity: str | None) -> dict | None:
    if not conclusion or not quantity:
        return None
    wanted = normalize_name(quantity)
    for item in conclusion.get("quantities") or []:
        if normalize_name(str(item.get("name", ""))) == wanted:
            return dict(item)
    return None


# --- why a clip was marked down ----------------------------------------------------------------
def _texts(value: Any) -> Iterable[str]:
    """Every human-readable string in a JSON value. A dict key counts only when its value is
    True, so {"out_of_plane": false} doesn't read as a finding."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if item is True:
                yield str(key)
            elif not isinstance(item, bool | int | float):
                yield from _texts(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _texts(item)


def reason_corpus(experiment: Mapping[str, Any] | None, events: Iterable[Mapping[str, Any]]):
    """The places Ghostline says why it lowered its confidence or refused: the conclusion's
    caveats, warnings in the notebook, the experiment's error, and a reviewer's major issues. An
    observation like 'the camera is steady' isn't one of them."""
    experiment = experiment or {}
    conclusion = experiment.get("conclusion") or {}
    texts: list[str] = [*conclusion.get("caveats", [])]
    if experiment.get("error"):
        texts.append(str(experiment["error"]))
    scene = experiment.get("scene") or {}
    texts += [str(scene["reason_if_not"])] if scene.get("reason_if_not") else []
    for review in experiment.get("reviews") or []:
        for issue in review.get("issues") or []:
            if issue.get("severity") == "major":
                texts += [str(issue.get(k, "")) for k in ("claim", "evidence", "detail")]
    for event in events:
        if event.get("kind") == "warning":
            texts += [str(event.get("title", "")), str(event.get("body_md", ""))]
            texts += list(_texts(event.get("data") or {}))
    return [t for t in texts if t]


def mentions(texts: Iterable[str], pattern: str) -> bool:
    """True if any text matches, unless the match is negated ('no camera motion detected')."""
    regex = re.compile(pattern, re.I)
    for text in texts:
        for match in regex.finditer(text):
            if not _NEGATION.search(text[: match.start()][-40:]):
                return True
    return False


def reasons_found(texts: Iterable[str]) -> list[str]:
    texts = list(texts)
    return [code for code, pattern in FAILURE_PATTERNS.items() if mentions(texts, pattern)]


# --- one clip ----------------------------------------------------------------------------------
class ClipResult(BaseModel):
    clip_id: str
    kind: str = "scored"
    system: str
    object: str = ""
    experiment_id: str | None = None
    status: str = "pending"  # complete | failed | timeout | pending | missing_file
    error: str | None = None
    quantity: str | None = None
    unit: str = ""
    truth: float | None = None
    truth_sigma: float | None = None
    truth_method: str = ""
    measured: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    sigma_measured: float | None = None
    error_abs: float | None = None
    error_rel: float | None = None
    z: float | None = None
    covered: bool | None = None
    consistent: bool | None = None  # |z| <= 1.96: the two values agree within their uncertainties
    model_selected: str | None = None
    model_expected: str = ""
    model_correct: bool | None = None
    confidence: str | None = None
    review_status: str | None = None
    caveats: list[str] = Field(default_factory=list)
    failure_expected: str = ""
    failure_low_confidence: bool | None = None
    failure_reasons: list[str] = Field(default_factory=list)
    failure_flagged: bool | None = None
    cost_usd: float | None = None
    duration_s: float | None = None
    note: str = ""

    @property
    def ran(self) -> bool:
        return self.status not in NOT_RUN


def blank_result(row: ManifestRow, truth: Truth | None, status: str = "pending", **extra):
    return ClipResult(
        clip_id=row.clip_id,
        kind=row.kind,
        system=row.system,
        object=row.object,
        status=status,
        quantity=row.quantity,
        unit=row.unit,
        truth=None if truth is None else truth.value,
        truth_sigma=None if truth is None else truth.sigma,
        truth_method="" if truth is None else truth.method,
        model_expected=row.expected_model,
        failure_expected=row.failure_expected,
        **extra,
    )


def score_clip(
    row: ManifestRow,
    truth: Truth | None,
    experiment: Mapping[str, Any] | None,
    events: Iterable[Mapping[str, Any]] = (),
    *,
    status: str | None = None,
    error: str | None = None,
    duration_s: float | None = None,
) -> ClipResult:
    """Everything the benchmark says about one clip, from the stored experiment."""
    experiment = experiment or {}
    events = list(events)
    exp_status = experiment.get("status")
    status = status or ("complete" if exp_status == "complete" else "failed")
    result = blank_result(
        row,
        truth,
        status,
        experiment_id=str(experiment["id"]) if experiment.get("id") else None,
        error=error or experiment.get("error"),
        duration_s=duration_s,
        cost_usd=experiment.get("cost_usd"),
    )
    conclusion = experiment.get("conclusion") if status == "complete" else None
    if conclusion:
        result.model_selected = conclusion.get("winner_id") or experiment.get("winner_model_id")
        result.confidence = conclusion.get("confidence") or experiment.get("confidence")
        result.caveats = [str(c) for c in conclusion.get("caveats", [])]
    result.review_status = experiment.get("review_status")

    found = find_quantity(conclusion, row.quantity)
    if found is not None and None not in (found.get("value"), found.get("ci_low")):
        result.measured = float(found["value"])
        result.ci_low, result.ci_high = float(found["ci_low"]), float(found["ci_high"])
        result.sigma_measured = sigma_from_interval(result.ci_low, result.ci_high)
        if not result.unit:
            result.unit = str(found.get("unit", ""))

    if result.measured is not None and truth is not None:
        result.error_abs = result.measured - truth.value
        result.error_rel = result.error_abs / abs(truth.value) if truth.value else None
        result.z = z_score(result.measured, truth.value, result.sigma_measured, truth.sigma)
        result.covered = interval_contains(result.ci_low, result.ci_high, truth.value)
        result.consistent = None if result.z is None else abs(result.z) <= 1.96

    if row.expected_models and status in ("complete", "failed", "timeout"):
        result.model_correct = result.model_selected in row.expected_models

    if row.kind == "failure" and status != "pending":
        texts = reason_corpus(experiment, events)
        result.failure_reasons = reasons_found(texts)
        refused = status == "failed"
        result.failure_low_confidence = True if refused else result.confidence == "low"
        right_reason = row.failure_expected in result.failure_reasons
        result.failure_flagged = bool(result.failure_low_confidence and right_reason)
        if refused and right_reason:
            result.note = "Refused to analyze, for the expected reason."
        elif result.failure_low_confidence and not right_reason:
            result.note = "Low confidence, but not for the expected reason."
        elif not result.failure_low_confidence:
            result.note = f"Confidence was {result.confidence}, not low."
    return result


# --- the whole benchmark -----------------------------------------------------------------------
class SystemSummary(BaseModel):
    n: int = 0
    covered: int = 0
    median_abs_rel_error: float | None = None


class Summary(BaseModel):
    n_clips: int
    n_run: int
    n_complete: int
    n_failed: int
    n_scored: int  # scored clips with a ground truth, run or not
    n_covered: int
    coverage: float | None
    coverage_low: float | None = None
    coverage_high: float | None = None
    n_measured: int = 0
    coverage_of_measured: float | None = None
    n_consistent: int = 0
    median_abs_rel_error: float | None = None
    median_abs_z: float | None = None
    rms_z: float | None = None
    n_model_scored: int = 0
    n_model_correct: int = 0
    model_selection_accuracy: float | None = None
    n_failure: int = 0
    n_failure_flagged: int = 0
    total_cost_usd: float | None = None
    by_system: dict[str, SystemSummary] = Field(default_factory=dict)

    @property
    def failure_cases_flagged(self) -> str:
        return f"{self.n_failure_flagged}/{self.n_failure}"


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize(results: Iterable[ClipResult]) -> Summary:
    results = list(results)
    ran = [r for r in results if r.ran]
    scored = [r for r in ran if r.kind == "scored" and r.truth is not None]
    measured = [r for r in scored if r.measured is not None]
    covered = [r for r in scored if r.covered]
    low, high = wilson_interval(len(covered), len(scored)) if scored else (None, None)
    rel = [abs(r.error_rel) for r in measured if r.error_rel is not None]
    zs = [r.z for r in measured if r.z is not None]
    model_scored = [r for r in ran if r.model_expected and r.kind == "scored"]
    model_correct = [r for r in model_scored if r.model_correct]
    failures = [r for r in ran if r.kind == "failure"]
    by_system: dict[str, list[ClipResult]] = defaultdict(list)
    for r in scored:
        by_system[r.system].append(r)
    costs = [r.cost_usd for r in ran if r.cost_usd is not None]
    return Summary(
        n_clips=len(results),
        n_run=len(ran),
        n_complete=sum(r.status == "complete" for r in ran),
        n_failed=sum(r.status != "complete" for r in ran),
        n_scored=len(scored),
        n_covered=len(covered),
        coverage=_ratio(len(covered), len(scored)),
        coverage_low=low,
        coverage_high=high,
        n_measured=len(measured),
        coverage_of_measured=_ratio(sum(bool(r.covered) for r in measured), len(measured)),
        n_consistent=sum(bool(r.consistent) for r in measured),
        median_abs_rel_error=statistics.median(rel) if rel else None,
        median_abs_z=statistics.median(abs(z) for z in zs) if zs else None,
        rms_z=math.sqrt(sum(z * z for z in zs) / len(zs)) if zs else None,
        n_model_scored=len(model_scored),
        n_model_correct=len(model_correct),
        model_selection_accuracy=_ratio(len(model_correct), len(model_scored)),
        n_failure=len(failures),
        n_failure_flagged=sum(bool(r.failure_flagged) for r in failures),
        total_cost_usd=sum(costs) if costs else None,
        by_system={
            system: SystemSummary(
                n=len(group),
                covered=sum(bool(r.covered) for r in group),
                median_abs_rel_error=statistics.median(
                    abs(r.error_rel) for r in group if r.error_rel is not None
                )
                if any(r.error_rel is not None for r in group)
                else None,
            )
            for system, group in sorted(by_system.items())
        },
    )


def to_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=False, default=str)
