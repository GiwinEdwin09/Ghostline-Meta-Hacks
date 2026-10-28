"""The timing model: from frame timestamps and what we know about the capture to real seconds.

A wrong time base gives a wrong g, so every decision here is recorded with its source and
anything doubtful is turned into a warning the user must confirm.
"""

from __future__ import annotations

import numpy as np

from .models import Cluster, DeclaredCapture, Notice, ProbeInfo, TimingModel

CFR_REL_STD = 0.02  # std of the intervals under 2% of their mean
CLUSTER_GAP = 1.25  # sorted intervals further apart than this ratio start a new cluster
RATE_MATCH = 0.03  # file rate within 3% of a nominal rate counts as equal (59.94 vs 60)
GAP_RATIO = 1.5  # an interval this many times the median is a gap
MIN_SECOND_CLUSTER = 0.05  # a second cluster needs at least 5% of the intervals


def _group_intervals(dt: np.ndarray) -> list[np.ndarray]:
    """Indices of intervals grouped by size: sorted values split where the ratio jumps."""
    order = np.argsort(dt)
    sorted_dt = dt[order]
    cuts = np.nonzero(sorted_dt[1:] / np.maximum(sorted_dt[:-1], 1e-12) > CLUSTER_GAP)[0] + 1
    return [order[a:b] for a, b in zip([0, *cuts], [*cuts, len(dt)], strict=True)]


def _longest_run(labels: np.ndarray, wanted: int) -> tuple[int, int]:
    """(start, stop) of the longest run of `wanted` in labels."""
    best = (0, 0)
    start = None
    for i, label in enumerate([*labels, -2]):
        if label == wanted and start is None:
            start = i
        elif label != wanted and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def _decide_factor(
    file_fps: float, declared: DeclaredCapture, tag_fps: float | None, warnings: list[Notice]
):
    """Return (time_factor, source, retimed, needs_confirmation)."""
    confirm = False
    capture, source = None, "assumed_realtime"
    if declared.mode == "slow_mo":
        if declared.fps:
            capture, source = declared.fps, "declared"
            if tag_fps and abs(tag_fps - declared.fps) / declared.fps > RATE_MATCH:
                warnings.append(
                    Notice(
                        code="capture_rate_conflict",
                        message=f"You said {declared.fps:g} fps but the file's metadata says "
                        f"{tag_fps:g} fps. Using what you said.",
                    )
                )
                confirm = True
        elif tag_fps:
            capture, source = tag_fps, "metadata_tag"
        elif file_fps < 100:
            warnings.append(
                Notice(
                    code="slow_mo_fps_unknown",
                    message="This looks like slow-mo but the capture rate is unknown. Tell us "
                    "the fps (for example 240) so the time base is right.",
                )
            )
            confirm = True
        else:
            source = "declared"
    elif declared.mode == "normal":
        source = "declared"
        if tag_fps and tag_fps > file_fps * (1 + RATE_MATCH):
            capture, source = tag_fps, "metadata_tag"
            warnings.append(
                Notice(
                    code="capture_rate_conflict",
                    message=f"You said normal speed but the file's metadata says it was captured "
                    f"at {tag_fps:g} fps and stored at {file_fps:g} fps. Trusting the metadata.",
                )
            )
            confirm = True
        elif declared.fps and abs(file_fps - declared.fps) / declared.fps > RATE_MATCH:
            warnings.append(
                Notice(
                    code="declared_fps_mismatch",
                    message=f"You said {declared.fps:g} fps but the file runs at {file_fps:g} "
                    "fps. Using the file's timestamps as real time.",
                )
            )
    elif tag_fps:
        capture, source = tag_fps, "metadata_tag"

    if capture is None:
        return 1.0, source, False, confirm
    ratio = file_fps / capture
    if abs(ratio - 1) <= RATE_MATCH:
        return 1.0, source, False, confirm
    if ratio > 1:
        warnings.append(
            Notice(
                code="declared_fps_mismatch",
                message=f"The file runs at {file_fps:g} fps, faster than the {capture:g} fps "
                "capture rate it was declared with. Using the file's timestamps as real time.",
            )
        )
        return 1.0, source, False, confirm
    warnings.append(
        Notice(
            code="retimed",
            message=f"Captured at {capture:g} fps but stored at {file_fps:g} fps: the clip was "
            f"re-timed, so one file second is {ratio:.4g} real seconds. Please confirm.",
        )
    )
    return ratio, source, True, True


def build_timing(
    pts: np.ndarray, probe: ProbeInfo | None = None, declared: DeclaredCapture | None = None
) -> TimingModel:
    """Classify the frame intervals and combine them with what the user declared.

    `pts` are file timestamps in seconds. The result's `time_factor` converts file seconds to
    real seconds; `usable_start:usable_stop` is the part with uniform timing.
    """
    declared = declared or DeclaredCapture()
    pts = np.asarray(pts, float)
    n = len(pts)
    warnings: list[Notice] = []
    if n < 3:
        raise ValueError("need at least three frames to build a timing model")
    dt = np.diff(pts)
    if np.any(dt <= 0):
        warnings.append(
            Notice(
                code="non_monotonic_timestamps",
                message="Some frames share or reverse timestamps; they were ordered by time.",
            )
        )
        dt = np.maximum(dt, 1e-9)
    median = float(np.median(dt))

    groups = _group_intervals(dt)
    big = [g for g in groups if len(g) >= max(3, MIN_SECOND_CLUSTER * len(dt))]
    labels = np.full(len(dt), -1)
    for k, g in enumerate(big):
        labels[g] = k
    spans = [_longest_run(labels, k) for k in range(len(big))]
    structural = [k for k in range(len(big)) if spans[k][1] - spans[k][0] >= max(5, 0.03 * len(dt))]
    clusters: list[Cluster] = []
    start, stop = 0, n
    used = dt
    if len(structural) >= 2:
        classification = "mixed"
        best = max(structural, key=lambda k: spans[k][1] - spans[k][0])
        start, stop = spans[best][0], spans[best][1] + 1
        used = dt[start : stop - 1]
        clusters = [
            Cluster(
                dt_s=float(np.median(dt[big[k]])),
                count=len(big[k]),
                start_frame=spans[k][0],
                stop_frame=spans[k][1] + 1,
            )
            for k in structural
        ]
    median = float(np.median(used))
    in_main = np.abs(used / median - 1) <= CLUSTER_GAP - 1
    core = used[in_main] if in_main.any() else used
    jitter = float(core.std() / core.mean())
    if len(structural) < 2:
        classification = "cfr" if jitter < CFR_REL_STD else "vfr_jitter"
    n_gaps = int(np.sum(used > GAP_RATIO * median))
    if n_gaps:
        warnings.append(
            Notice(
                code="frame_gaps",
                message=f"{n_gaps} frame interval(s) are much longer than usual; frames may have "
                "been dropped. Fits use the real timestamps.",
            )
        )
    if classification == "mixed":
        warnings.append(
            Notice(
                code="mixed_frame_rate",
                message="The frame rate changes inside this clip (for example slow-mo with "
                f"normal-speed ends). Only frames {start} to {stop - 1} are used. Export the clip "
                "with the whole thing slowed, or upload the original from a computer.",
            )
        )

    file_fps = 1.0 / median
    tag_fps = probe.capture_fps_tag if probe else None
    factor, source, retimed, confirm = _decide_factor(file_fps, declared, tag_fps, warnings)
    n_used = max(stop - start - 1, 1)
    sigma = 2e-4 if factor == 1.0 else 1e-3
    sigma = float(np.hypot(sigma, jitter / np.sqrt(n_used)))
    return TimingModel(
        classification=classification,
        n_frames=n,
        file_fps=file_fps,
        mean_dt_s=median,
        jitter_rel=jitter,
        time_factor=factor,
        time_factor_source=source,
        real_fps=file_fps / factor,
        usable_start=start,
        usable_stop=stop,
        sigma_time_rel=sigma,
        retimed=retimed,
        needs_confirmation=confirm or classification == "mixed",
        clusters=clusters,
        n_gaps=n_gaps,
        warnings=warnings,
    )


def real_timestamps(pts: np.ndarray, timing: TimingModel) -> np.ndarray:
    """Real seconds since the first frame, after the time factor."""
    pts = np.asarray(pts, float)
    return (pts - pts[0]) * timing.time_factor
