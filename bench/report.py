"""The benchmark's outputs (build plan Part 25): results.csv, results.md and results.json.

results.md is the table that goes into the README and the pitch. results.json has the shape the
web app's benchmark page reads (`BenchmarkTable` in web/src/lib/types.ts), so the page can be fed
from it until GET /benchmark serves the same fields.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from bench.metrics import ClipResult, Summary

CSV_COLUMNS = [
    "clip_id",
    "kind",
    "system",
    "object",
    "experiment_id",
    "status",
    "quantity",
    "unit",
    "truth",
    "truth_sigma",
    "truth_method",
    "measured",
    "ci_low",
    "ci_high",
    "error_abs",
    "error_rel",
    "z",
    "covered",
    "model_selected",
    "model_expected",
    "model_correct",
    "confidence",
    "review_status",
    "failure_expected",
    "failure_reasons",
    "failure_flagged",
    "cost_usd",
    "duration_s",
    "error",
    "note",
]
SYSTEM_LABELS = {
    "free_fall": "Free fall",
    "projectile": "Projectile",
    "pendulum": "Pendulum",
    "ramp": "Ramp",
    "bounce": "Bounce",
    "spin_down": "Spin-down",
}


# --- number formatting -------------------------------------------------------------------------
def sig(value: float | None, digits: int = 4) -> str:
    """Round to significant figures: 9.8017 -> '9.802', 0.30688 -> '0.3069', 0.0021 -> '0.0021'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if value == 0:
        return "0"
    places = max(0, digits - 1 - math.floor(math.log10(abs(value))))
    return f"{value:.{places}f}"


def pct(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else f"{value * 100:+.{digits}f}%"


def mark(flag: bool | None) -> str:
    return "–" if flag is None else ("yes" if flag else "no")


# --- results.csv -------------------------------------------------------------------------------
def write_csv(results: Iterable[ClipResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for r in results:
            row = r.model_dump()
            row["failure_reasons"] = ";".join(r.failure_reasons)
            writer.writerow({c: "" if row[c] is None else row[c] for c in CSV_COLUMNS})


# --- results.json (the web app's BenchmarkTable) -------------------------------------------------
def web_rows(results: Iterable[ClipResult]) -> list[dict[str, Any]]:
    """One row per clip that has a truth to show, or is a failure case."""
    rows = []
    for r in results:
        failure = r.kind == "failure"
        if not r.ran or (r.truth is None and not failure) or r.kind == "story":
            continue
        rows.append(
            {
                "clip_id": r.clip_id,
                "experiment_id": r.experiment_id,
                "system": r.system,
                "object": r.object,
                "quantity": r.quantity or "",
                "unit": r.unit,
                "truth": r.truth if r.truth is not None else 0.0,
                "truth_sigma": r.truth_sigma if r.truth_sigma is not None else 0.0,
                "truth_method": r.truth_method,
                "measured": r.measured,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "z": r.z,
                "covered": r.covered,
                "model_selected": r.model_selected,
                "model_expected": r.model_expected.replace("|", " or ") or None,
                "model_correct": r.model_correct,
                "confidence": r.confidence,
                "is_failure_case": failure,
                "failure_expected": r.failure_expected.replace("_", " ") or None,
                "failure_flagged": r.failure_flagged,
                "note": r.note or None,
            }
        )
    return rows


def web_table(results: list[ClipResult], summary: Summary, generated_at: str) -> dict[str, Any]:
    return {
        "rows": web_rows(results),
        "summary": {
            "n_clips": summary.n_scored,
            "coverage": summary.coverage or 0.0,
            "model_selection_accuracy": summary.model_selection_accuracy or 0.0,
            "failure_cases_flagged": summary.failure_cases_flagged,
            "generated_at": generated_at,
        },
    }


# --- results.md --------------------------------------------------------------------------------
def _truth_cell(r: ClipResult) -> str:
    if r.truth is None:
        return "not yet measured"
    sigma = f" ± {sig(r.truth_sigma, 2)}" if r.truth_sigma else " (theory)"
    return f"{sig(r.truth)}{sigma}"


def _measured_cell(r: ClipResult) -> str:
    if r.measured is None:
        return "no result" if r.status != "complete" else "not reported"
    return f"{sig(r.measured)} [{sig(r.ci_low)}, {sig(r.ci_high)}]"


def _quantity_cell(r: ClipResult) -> str:
    return f"{r.quantity} ({r.unit})" if r.unit else (r.quantity or "–")


def _headline(summary: Summary) -> list[str]:
    lines = []
    if summary.n_scored:
        low, high = summary.coverage_low or 0.0, summary.coverage_high or 0.0
        lines.append(
            f"**Our 95% intervals contained the true value in {summary.n_covered} of "
            f"{summary.n_scored} clips ({summary.coverage:.0%}; 95% Wilson interval for that "
            f"share {low:.0%} to {high:.0%}).** Target: 90 to 98%."
        )
        if summary.median_abs_rel_error is not None:
            lines.append(
                f"The median absolute error was {summary.median_abs_rel_error:.1%}, and "
                f"{summary.n_consistent} of {summary.n_measured} measurements agreed with the "
                f"truth within their combined uncertainty (|z| ≤ 1.96)."
            )
    else:
        lines.append("**No scored clip has a ground truth yet**, so there is no coverage number.")
    if summary.n_model_scored:
        lines.append(
            f"The right physics was chosen in {summary.n_model_correct} of "
            f"{summary.n_model_scored} clips where it is defined "
            f"({summary.model_selection_accuracy:.0%})."
        )
    if summary.n_failure:
        lines.append(
            f"The deliberately bad clips: {summary.n_failure_flagged} of {summary.n_failure} came "
            "out low confidence for the right reason."
        )
    return lines


def render_markdown(
    results: list[ClipResult], summary: Summary, meta: Mapping[str, Any] | None = None
) -> str:
    meta = meta or {}
    out = ["# Ghostline benchmark results", ""]
    stamp = [f"Generated {meta['generated_at']}"] if meta.get("generated_at") else []
    for key, label in (
        ("mode", "back end in {} mode"),
        ("prompt_version", "prompts {}"),
        ("api_version", "API {}"),
    ):
        if meta.get(key):
            stamp.append(label.format(meta[key]))
    if stamp:
        out += [", ".join(stamp) + ".", ""]
    out += [*_headline(summary), ""]

    scored = [r for r in results if r.kind == "scored" and r.ran]
    out += [
        "| Clip | System | Quantity | Truth | Measured [95% interval] | Error | z | "
        "In interval | Model | Confidence |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in scored:
        model = r.model_selected or "–"
        if r.model_correct is not None:
            model += " ✓" if r.model_correct else f" ✗ (expected {r.model_expected})"
        out.append(
            f"| {r.clip_id} | {SYSTEM_LABELS.get(r.system, r.system)} | {_quantity_cell(r)} | "
            f"{_truth_cell(r)} | {_measured_cell(r)} | {pct(r.error_rel)} | {sig(r.z, 2)} | "
            f"{mark(r.covered)} | {model} | {r.confidence or '–'} |"
        )
    out.append("")

    failures = [r for r in results if r.kind == "failure" and r.ran]
    if failures:
        out += [
            "## Deliberately bad clips",
            "",
            "Expected: low confidence, for the reason the clip was made to show.",
            "",
            "| Clip | What is wrong | Confidence | Reasons Ghostline gave | Flagged correctly |",
            "|---|---|---|---|---|",
        ]
        for r in failures:
            reasons = ", ".join(r.failure_reasons) or ("refused" if r.status != "complete" else "–")
            out.append(
                f"| {r.clip_id} | {r.failure_expected.replace('_', ' ')} | "
                f"{r.confidence or ('no result' if r.status != 'complete' else '–')} | "
                f"{reasons} | {mark(r.failure_flagged)} |"
            )
        out.append("")

    if summary.by_system:
        out += [
            "## By system",
            "",
            "| System | Clips | Contain truth | Median abs. error |",
            "|---|---|---|---|",
        ]
        for system, s in summary.by_system.items():
            median = "–" if s.median_abs_rel_error is None else f"{s.median_abs_rel_error:.1%}"
            out.append(
                f"| {SYSTEM_LABELS.get(system, system)} | {s.n} | {s.covered} of {s.n} | {median} |"
            )
        out.append("")

    open_items = [r for r in results if r.kind == "scored" and (not r.ran or r.truth is None)]
    failed = [r for r in results if r.ran and r.status != "complete" and r.kind != "failure"]
    if open_items or failed:
        out += ["## Not scored", ""]
        for r in open_items:
            why = "not run yet" if not r.ran else "no ground truth yet"
            out.append(f"- {r.clip_id}: {why}")
        for r in failed:
            out.append(f"- {r.clip_id}: {r.status}. {r.error or ''}".rstrip())
        out.append("")

    out += [
        "## How the truth was obtained",
        "",
        "- **g** (free fall, pendulum): local gravity from latitude and altitude, "
        "9.780327 (1 + 0.0053024 sin²φ − 0.0000058 sin²2φ) − 3.086×10⁻⁶ h.",
        "- **μ_k** (ramp, block): tan θ of the tilt at which the block slides at constant "
        "speed; five repeats, mean ± SD.",
        "- **β** (ramp, ball): theory for a solid sphere, β = 2/5, so a = (5/7) g sin α.",
        "- **e** (bounce): the click timing in a normal-speed recording, e = Δt(n+1) / Δt(n), "
        "averaged over the drops.",
        "- z = (measured − truth) / √(σ_measured² + σ_truth²), with σ_measured taken from the "
        "95% interval.",
        "",
    ]
    if summary.total_cost_usd is not None:
        out += [f"Model spend for this pass: ${summary.total_cost_usd:.2f}.", ""]
    return "\n".join(out)


def write_all(
    results: list[ClipResult], summary: Summary, out_dir: Path, meta: Mapping[str, Any]
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": out_dir / "results.csv",
        "md": out_dir / "results.md",
        "json": out_dir / "results.json",
    }
    write_csv(results, paths["csv"])
    paths["md"].write_text(render_markdown(results, summary, meta), encoding="utf-8")
    payload = web_table(results, summary, str(meta.get("generated_at", "")))
    payload["summary_full"] = summary.model_dump()
    payload["meta"] = dict(meta)
    paths["json"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return paths
