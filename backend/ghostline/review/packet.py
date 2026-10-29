"""The reviewer's packet (build plan Part 14): the artifacts a referee would get, and nothing else.

    the conclusion JSON, the full comparison table (held-out RMSE with intervals on every split,
    parameters with intervals, AIC/BIC, diagnostics), the data-quality flags, the calibration and
    timing uncertainty budget, and two images: the winner's fit plot and residual plot.

It never includes Wonder's conversation or private reasoning. It is built from the session's
harness results only (`agent.session.Session`), so it needs no model call to assemble.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ghostline.agent.numbers import round_tree
from ghostline.agent.session import Session
from ghostline.llm import data_url
from ghostline.physics.harness.schema import Diagnostics
from ghostline.physics.select import ModelScore, Selection
from ghostline.store.core_schemas import Conclusion


@dataclass
class Packet:
    text: str
    images: list[tuple[str, bytes]] = field(default_factory=list)  # (caption, PNG bytes)

    def content(self) -> list[dict[str, Any]]:
        """Chat Completions content parts: the text, then each image as an `image_url` data URL."""
        parts: list[dict[str, Any]] = [{"type": "text", "text": self.text}]
        for caption, png in self.images:
            parts.append({"type": "text", "text": caption})
            parts.append({"type": "image_url", "image_url": {"url": data_url(png, "image/png")}})
        return parts


def _diag(d: Diagnostics | None) -> dict[str, Any] | None:
    if d is None:
        return None
    return {
        "reduced_chi2": d.chi2_red,
        "lag1_autocorrelation": d.lag1,
        "durbin_watson": d.durbin_watson,
        "runs_test_p": d.runs_p,
        "systematic_residual_flag": d.systematic,
        "systematic_r2": d.systematic_r2,
        "rmse": d.rmse,
        "n_points": d.n,
        "n_parameters": d.n_params,
        "condition_number": d.condition,
    }


def _row(score: ModelScore, name: str) -> dict[str, Any]:
    return {
        "id": score.id,
        "name": name,
        "status": score.status,
        "error": score.error,
        "free_physical_parameters": score.n_physical,
        "parameters": [
            {
                "name": p.name,
                "value": p.value,
                "ci95": [p.ci_low, p.ci_high],
                "unit": p.unit,
                "kind": p.kind,
                "at_bound": p.at_bound,
            }
            for p in score.params
            if not p.derived
        ],
        "training_rmse": score.train_rmse,
        "heldout_rmse_by_training_fraction": score.heldout_rmse,
        "heldout_rmse_ci95_by_training_fraction": {k: list(v) for k, v in score.heldout_ci.items()},
        "heldout_nrmse": score.nrmse,
        "time_to_divergence_s": score.divergence_time,
        "aic_full_data": score.aic,
        "bic_full_data": score.bic,
        "training_residual_diagnostics": _diag(score.train_diagnostics),
        "full_data_residual_diagnostics": _diag(score.full_diagnostics),
    }


def comparison(sel: Selection, names: dict[str, str]) -> dict[str, Any]:
    return {
        "held_out_selection": {
            "status": sel.status,
            "selected": sel.selected_id,
            "splits_won": sel.splits_won,
            "splits": sel.n_splits,
            "tied_with": sel.tied_with,
            "message": sel.message,
        },
        "models": [_row(s, names.get(s.id, s.id)) for s in sel.models],
    }


def build_packet(
    session: Session,
    conclusion: Conclusion,
    plots: dict[str, bytes],
    *,
    heading: str = "Review this claim.",
) -> Packet:
    """The packet for a conclusion. `plots` maps `fit` and `resid` to the winner's PNGs."""
    sel = session.selection
    assert sel is not None and sel.selected_id is not None
    names = {m.id: m.name for m in session.models.values()}
    quantities = session.quantities.get(sel.selected_id, [])
    body = {
        "claim": conclusion.model_dump(),
        "comparison_table": comparison(sel, names),
        "winner_quantities_with_uncertainty_budget": [
            {
                "name": q.name,
                "value": q.value,
                "unit": q.unit,
                "ci95": [q.ci_low, q.ci_high],
                "sigma_total": q.sigma_total,
                "budget_one_sigma": q.budget,
                "at_bound": q.at_bound,
            }
            for q in quantities
        ],
        "data_quality_flags": [f.model_dump() for f in session.flags],
        "calibration_and_timing_uncertainty_one_sigma": {
            "scale_relative": session.budget.sigma_scale,
            "plane_offset_relative": session.budget.sigma_offset,
            "ramp_angle_rad": session.budget.sigma_angle,
            "timing_relative": session.budget.sigma_timing,
        },
        "measurement": {
            "system": session.system,
            "duration_s": session.duration_s,
            "training_fraction": session.primary_split,
            "noise_sigma_per_point": session.sigma,
            "range_of_motion": session.motion_range,
        },
    }
    text = (
        f"{heading}\n\n```json\n"
        + json.dumps(round_tree(body), indent=1, ensure_ascii=False)
        + "\n```\n\nThe two images are the winner's fit (training points blue, held-out points "
        "red, the model's curve orange) and its residuals."
    )
    images = []
    if "fit" in plots:
        images.append((f"Fit plot of {sel.selected_id}:", plots["fit"]))
    if "resid" in plots:
        images.append((f"Residual plot of {sel.selected_id}:", plots["resid"]))
    return Packet(text, images)
