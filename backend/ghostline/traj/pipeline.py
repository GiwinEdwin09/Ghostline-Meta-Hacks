"""build_trajectory: tracks + calibration -> clean physical time series (Part 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ghostline.calib.gravity import Gravity, from_free_fall, plumb_from_oscillation
from ghostline.calib.model import Calibration
from ghostline.calib.offset import depth_of_sheet_points
from ghostline.calib.ramp import cross_check
from ghostline.ingest.models import Notice

from .angles import (
    MIN_ARC_DEG,
    effective_length,
    fit_circle,
    pendulum_theta,
    spinner_phi,
)
from .clean import clean_track
from .events import (
    detect_impacts,
    detect_release_stop,
    flight_segments,
    impact_drop_mask,
    out_of_plane,
    retime_check,
)
from .model import (
    FLAG_ALIAS_RISK,
    FLAG_AT_REST,
    FLAG_NEAR_IMPACT,
    TrackSeries,
    TrajectoryResult,
)
from .noise import WINDOW, local_fit, noise_sigma, robust_std

FLIGHT_SYSTEMS = ("free_fall", "projectile", "bounce")
MIN_SAMPLES = 10
MIN_SIGMA_PX = 0.02  # floor for the noise level, so a noiseless track can't give a zero threshold
N_SAMPLE_POINTS = 60
GRAVITY_DATA_MAX_SIGMA_DEG = 3.0
GRAVITY_MAX_SHIFT_DEG = 20.0


class TrajectoryError(RuntimeError):
    """Too little usable tracking data to build a trajectory."""


def _static_point(track: TrackSeries) -> np.ndarray:
    """Median pixel position of a static tracked object (pivot or hub)."""
    ok = (track.conf >= 0.5) & ~track.border
    if ok.sum() == 0:
        ok = np.ones(len(track), bool)
    return np.array([np.median(track.cx[ok]), np.median(track.cy[ok])])


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))))


def _refine_gravity(
    system, t, plane_mm, ev, segment, pivot_mm, calib: Calibration, notices: list[Notice]
) -> Gravity | None:
    """Better 'down' than the camera-level default, from the data or a plumb reference."""
    current = calib.gravity_down
    found: Gravity | None = None
    start = ev.release_t if ev.release_t is not None else -np.inf
    end = ev.window_end_t if ev.window_end_t is not None else np.inf
    window = (t >= start) & (t <= end)
    if system in ("free_fall", "projectile"):
        found = from_free_fall(t[window], plane_mm[window])
    elif system == "bounce":
        parts = []
        for seg in np.unique(segment):
            m = (segment == seg) & window
            g = from_free_fall(t[m], plane_mm[m])
            if g is not None and g.sigma_deg < GRAVITY_DATA_MAX_SIGMA_DEG:
                parts.append(g)
        if parts:
            w = np.array([1 / max(p.sigma_deg, 0.1) ** 2 for p in parts])
            d = sum(wi * p.down for wi, p in zip(w, parts, strict=True))
            found = Gravity(d / np.linalg.norm(d), "data", float(1 / np.sqrt(w.sum())))
    elif system == "pendulum" and pivot_mm is not None:
        found = plumb_from_oscillation(t[window], plane_mm[window], pivot_mm)
    if found is None:
        return None
    if found.sigma_deg > GRAVITY_DATA_MAX_SIGMA_DEG:
        return None
    shift = _angle_between(found.down, current)
    if shift > GRAVITY_MAX_SHIFT_DEG:
        notices.append(
            Notice(
                code="gravity_disagreement",
                message=f"The {found.source} gravity direction differs from the camera's "
                f"vertical by {shift:.0f} degrees; keeping the camera's vertical.",
            )
        )
        return None
    return found


def build_trajectory(
    tracks: TrackSeries,
    calib: Calibration,
    system: str,
    *,
    pivot: TrackSeries | None = None,
    hub: TrackSeries | None = None,
    timing: dict | None = None,
) -> TrajectoryResult:
    """Turn one object's per-frame track into metres, radians and real seconds.

    `system` is the SceneAnalysis system type. For a pendulum, `pivot` is the tracked pivot
    (otherwise it is found by a circle fit to the bob's path); for a spinner, `tracks` is the
    marker and `hub` the hub (otherwise the circle centre of the marker's path). `timing` is the
    dumped Part 5 TimingModel, used only for the uncertainty summary.
    """
    notices: list[Notice] = []
    frames = clean_track(tracks)
    kept = np.nonzero(frames["drop"].to_numpy() == "")[0]
    if len(kept) < MIN_SAMPLES:
        raise TrajectoryError(
            f"only {len(kept)} usable frames out of {len(tracks)}; need at least {MIN_SAMPLES}"
        )
    t = tracks.t[kept]
    fr = tracks.frame[kept]
    plane_mm = calib.px_to_plane_mm(tracks.px[kept])
    xy = calib.plane_mm_to_world(plane_mm, calib.gravity_down)

    # A fast spinner turns too sharply for a 15-frame cubic, so its position noise comes from how
    # far the marker strays from a circle around the hub instead.
    hub_mm = None
    spin_sigma = None
    if system == "spin_down":
        if hub is not None:
            hub_mm = calib.px_to_plane_mm(_static_point(hub)[None])[0]
        else:
            circle = fit_circle(plane_mm)
            hub_mm = circle.centre
        radial = np.hypot(*(plane_mm - hub_mm).T) / 1000
        spin_sigma = robust_std(radial - np.median(radial))

    fit0 = local_fit(t, xy)
    sigma_pos = (
        spin_sigma if spin_sigma is not None else float(np.sqrt(np.mean(noise_sigma(fit0) ** 2)))
    )
    sigma_pos = max(sigma_pos, MIN_SIGMA_PX * calib.m_per_px)
    ev = detect_release_stop(t, xy, fit0, sigma_pos)
    if ev.release_t is None:
        notices.append(
            Notice(code="no_motion", message="No motion was detected in the tracked object.")
        )

    # bounces: split into flights, dropping two frames each side of every impact
    segment = np.zeros(len(t), int)
    impacts = []
    if system == "bounce":
        impacts = detect_impacts(t, xy[:, 1], fr, sigma_pos)
        gone = impact_drop_mask(fr, impacts)
        frames.loc[kept[gone], "drop"] = "impact"
        kept = kept[~gone]
        t, fr, plane_mm = t[~gone], fr[~gone], plane_mm[~gone]
        segment = flight_segments(t, impacts)
        if not impacts:
            notices.append(Notice(code="no_impacts", message="No bounces were found."))

    # pivot / hub in the sheet-plane millimetre frame
    pivot_mm = None
    circle = None
    if system == "pendulum":
        if pivot is not None:
            pivot_mm = calib.px_to_plane_mm(_static_point(pivot)[None])[0]
        elif len(t) >= 10:
            circle = fit_circle(plane_mm)
            pivot_mm = circle.centre
            if circle.arc_deg < MIN_ARC_DEG:
                notices.append(
                    Notice(
                        code="pivot_unconstrained",
                        message=f"The bob swings only {circle.arc_deg:.0f} degrees, too little to "
                        "locate the pivot from its path. Track the pivot or enter the length.",
                    )
                )

    # which way is down
    down = calib.gravity_down
    gravity_info = {
        "source": calib.gravity.source,
        "angle_deg": calib.gravity.angle_deg,
        "sigma_deg": calib.gravity.sigma_deg,
    }
    better = _refine_gravity(system, t, plane_mm, ev, segment, pivot_mm, calib, notices)
    if better is not None:
        down = better.down
        gravity_info = {
            "source": better.source,
            "angle_deg": better.angle_deg,
            "sigma_deg": better.sigma_deg,
            "camera_level_angle_deg": calib.gravity.angle_deg,
        }
    xy = calib.plane_mm_to_world(plane_mm, down)

    # angles
    angle = None
    angle_name = None
    extras: dict = {}
    alias = np.zeros(len(t), bool)
    if system == "pendulum" and pivot_mm is not None:
        pivot_w = calib.plane_mm_to_world(pivot_mm[None], down)[0]
        angle, angle_name = pendulum_theta(xy, pivot_w), "theta"
        dist = np.hypot(*(xy - pivot_w).T)
        length = float(np.median(dist))
        radius = float(np.sqrt(np.nanmedian(tracks.area[kept]) / np.pi) * calib.m_per_px)
        extras["pendulum"] = {
            "pivot_source": "tracked" if pivot is not None else "circle_fit",
            "pivot_xy_m": pivot_w.tolist(),
            "length_m": length,
            "length_rms_m": float(np.std(dist)),
            "bob_radius_m": radius,
            "length_eff_m": effective_length(length, radius),
            "sigma_length_m": (circle.sigma_radius / 1000) if circle else None,
            "arc_deg": float(np.degrees(np.ptp(np.unwrap(angle)))),
        }
    elif system == "spin_down" and hub_mm is not None:
        hub_w = calib.plane_mm_to_world(hub_mm[None], down)[0]
        rho = float(np.median(np.hypot(*(xy - hub_w).T)))
        spin = spinner_phi(xy, hub_w, fr, sigma_pos / max(rho, 1e-6))
        angle, angle_name, alias = spin.phi, "phi", spin.alias_risk
        extras["spinner"] = {
            "hub_source": "tracked" if hub is not None else "circle_fit",
            "hub_xy_m": hub_w.tolist(),
            "marker_radius_m": rho,
            "max_step_deg_per_frame": spin.max_step_deg,
            "direction": "ccw" if spin.direction > 0 else "cw" if spin.direction < 0 else "none",
            "direction_consistent": spin.direction_consistent,
            "total_rotation_rev": float(np.ptp(spin.phi) / (2 * np.pi)),
        }
        if alias.any():
            notices.append(
                Notice(
                    code="aliasing_risk",
                    message=f"The spinner turns up to {spin.max_step_deg:.0f} degrees per frame, "
                    "so its angle may alias. Film at a higher frame rate or flick it more gently.",
                )
            )
        if not spin.direction_consistent:
            notices.append(
                Notice(
                    code="spin_direction_inconsistent",
                    message="The spin direction flips during the clip; the angle may be "
                    "unreliable.",
                )
            )

    # final fits: noise, velocities
    fit = local_fit(t, xy, segment)
    # leave out the neighbourhood of the release and the stop: the motion has a kink there
    margin = (WINDOW // 2) * float(np.median(np.diff(t)))
    away = np.ones(len(t), bool)
    for event_t in (ev.release_t, ev.stop_t):
        if event_t is not None:
            away &= np.abs(t - event_t) > margin
    away = away if away.sum() >= MIN_SAMPLES else None
    sig_xy = noise_sigma(fit, away) if spin_sigma is None else np.array([spin_sigma, spin_sigma])
    sigma_pos = float(np.sqrt(np.mean(sig_xy**2)))
    sigma_angle = omega = None
    if angle is not None:
        afit = local_fit(t, angle, segment)
        sigma_angle = float(noise_sigma(afit, away)[0])
        omega = afit.slope
    s_along = None
    if system == "ramp":
        window = np.ones(len(t), bool)
        if ev.release_t is not None:
            window = (t >= ev.release_t) & (t <= (ev.window_end_t or np.inf))
        pts = xy[window] if window.sum() >= 5 else xy
        axis = np.linalg.svd(pts - pts.mean(axis=0))[2][0]
        if (pts[-1] - pts[0]) @ axis < 0:
            axis = -axis
        s_along = (xy - (xy[window][0] if window.any() else xy[0])) @ axis
        path_angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
        extras["ramp"] = {"path_angle_deg": path_angle, "board_angle_deg": None}
        if calib.ramp is not None:
            diff, note = cross_check(calib.ramp.angle_deg, pts)
            extras["ramp"].update(board_angle_deg=calib.ramp.angle_deg, angle_diff_deg=diff)
            if note is not None:
                notices.append(note)

    # flags, weights
    flags = np.zeros(len(t), int)
    if ev.release_t is not None:
        flags[t < ev.release_t] |= FLAG_AT_REST
    if ev.window_end_t is not None:
        flags[t > ev.window_end_t] |= FLAG_AT_REST
    for imp in impacts:
        before = np.nonzero(t < imp.t)[0]
        after = np.nonzero(t >= imp.t)[0]
        for idx in [*before[-1:], *after[:1]]:
            flags[idx] |= FLAG_NEAR_IMPACT
    flags[alias] |= FLAG_ALIAS_RISK
    weight = np.clip(tracks.conf[kept], 0.0, 1.0)

    # checks
    in_window = (flags & FLAG_AT_REST) == 0
    speed = np.hypot(fit.slope[:, 0], fit.slope[:, 1])
    columns = {"x": xy[:, 0], "y": xy[:, 1], "t": t}
    if angle is not None:
        columns[angle_name] = angle
    # area in (depth / f)^2 units: removes the perspective change of a camera that is not square
    # to the sheet
    area_kept = tracks.area[kept]
    depth = depth_of_sheet_points(
        calib.H_np,
        calib.offset.f_px,
        calib.image_size,
        calib.px_to_plane_mm(tracks.px[kept], False),
    )
    metric_area = area_kept if depth is None else area_kept * (depth / calib.offset.f_px) ** 2
    oop = out_of_plane(
        metric_area[in_window], {k: v[in_window] for k, v in columns.items()}, speed[in_window]
    )
    if oop["flag"]:
        notices.append(
            Notice(
                code="out_of_plane",
                message="The object's apparent size changes with its position, so it may be moving "
                f"toward or away from the camera (about {oop['dz_over_z']:.1%} of the distance).",
            )
        )
    retime = {"checked": False}
    if system in FLIGHT_SYSTEMS:
        retime = retime_check(t, xy[:, 1], segment, in_window)
        if retime["suspected_factor"]:
            k = retime["suspected_factor"]
            notices.append(
                Notice(
                    code="looks_retimed",
                    message=f"Gravity fitted from this flight is about 9.81/{k**2}, as if the clip "
                    f"were slowed {k}x. Check the capture mode and frame rate.",
                )
            )

    df = pd.DataFrame({"t": t, "x": xy[:, 0], "y": xy[:, 1]})
    if angle is not None:
        df[angle_name] = angle
    if s_along is not None:
        df["s"] = s_along
    df["weight"] = weight
    df["flags"] = flags
    df["segment_id"] = segment
    df["frame"] = fr
    df["vx"], df["vy"] = fit.slope[:, 0], fit.slope[:, 1]
    if omega is not None:
        df["omega"] = omega

    summary = _summary(
        system, tracks, frames, df, angle_name, ev, impacts, in_window, sigma_pos, sig_xy,
        sigma_angle, gravity_info, oop, retime, extras, calib, timing, notices,
    )  # fmt: skip
    return TrajectoryResult(df, frames, summary, notices)


def _summary(
    system, tracks, frames, df, angle_name, ev, impacts, in_window, sigma_pos, sig_xy,
    sigma_angle, gravity_info, oop, retime, extras, calib, timing, notices,
) -> dict:  # fmt: skip
    path = tracks.px[np.isin(tracks.frame, df["frame"].to_numpy())]
    probe_px = path[np.linspace(0, len(path) - 1, min(12, len(path))).round().astype(int)]
    scale_at_object = calib.scale_sigma_at(probe_px)
    drops = frames["drop"].value_counts().to_dict()
    drops.pop("", None)
    win = df[in_window] if in_window.any() else df
    cols = ["x", "y"] + ([angle_name] if angle_name else [])
    idx = np.unique(
        np.linspace(0, len(win) - 1, min(N_SAMPLE_POINTS, len(win))).round().astype(int)
    )
    samples = win.iloc[idx][["t", *cols]].round(6).to_dict(orient="list")
    t0 = float(win["t"].iloc[0])
    t1 = float(win["t"].iloc[-1])
    dt = np.diff(df["t"].to_numpy())
    px_per_m = 1.0 / calib.m_per_px
    return {
        "system": system,
        "angle_column": angle_name,
        "n_points": int(len(df)),
        "n_frames_in": int(len(tracks)),
        "n_dropped": {k: int(v) for k, v in drops.items()},
        "duration_s": t1 - t0,
        "t_range": [float(df["t"].iloc[0]), float(df["t"].iloc[-1])],
        "dt_median_s": float(np.median(dt)) if len(dt) else None,
        "noise": {
            "sigma_x_m": float(sig_xy[0]),
            "sigma_y_m": float(sig_xy[1]),
            "sigma_pos_m": sigma_pos,
            "sigma_pos_px": sigma_pos * px_per_m,
            "sigma_angle_rad": sigma_angle,
            "method": "robust std of leverage-corrected residuals of a local cubic fit, 15 frames",
        },
        "events": {
            "release_t": ev.release_t,
            "stop_t": ev.stop_t,
            "window_end_t": ev.window_end_t,
            "impacts_t": [i.t for i in impacts],
            "n_flights": int(df["segment_id"].nunique()),
        },
        "motion_window": {"t_start": t0, "t_end": t1, "n_points": int(len(win))},
        "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in cols}
        | {"speed_max_m_s": float(np.hypot(df["vx"], df["vy"]).max())},
        "gravity": gravity_info,
        "out_of_plane": oop,
        "retime_check": retime,
        **extras,
        "uncertainty": {
            "scale_rel": calib.uncertainty.sigma_scale_rel,
            "scale_rel_at_object": scale_at_object,
            "out_of_plane_rel": oop.get("dz_over_z", 0.0) if oop.get("flag") else 0.0,
            "gravity_deg": gravity_info["sigma_deg"],
            "timing_rel": (timing or {}).get("sigma_time_rel"),
        },
        "units": {"t": "s", "x": "m", "y": "m", "theta": "rad", "phi": "rad", "s": "m"},
        "samples": samples,
        "notices": [n.model_dump() for n in notices],
    }


__all__ = ["TrajectoryError", "build_trajectory"]
