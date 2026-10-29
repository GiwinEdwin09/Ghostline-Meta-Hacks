"""The five measurement stages of the pipeline (build plan Parts 5 to 9): ingest, scene analysis,
tracking, calibration and trajectory extraction. `jobs/stages.py` exposes them under the names
the handlers call.

Each stage reads the files the stage before it left on the Volume and writes its own, reports
progress through the `JobContext`, commits the Volume when it is done, and describes what it saw
in the lab notebook. Frame numbers in everything written to disk are master frame numbers.
"""

import logging
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ghostline.calib import calibrate as run_calibration
from ghostline.calib import detect_sheet_corners, write_calibration
from ghostline.calib.model import Calibration
from ghostline.ingest import DeclaredCapture as IngestDeclared
from ghostline.ingest import IngestError, ingest_trial
from ghostline.jobs import media
from ghostline.jobs.clients import llm_errors
from ghostline.jobs.context import JobContext
from ghostline.jobs.errors import StageError
from ghostline.jobs.stage_types import (
    CalibrationResult,
    IngestResult,
    TrackingResult,
    TrajectoryResult,
)
from ghostline.jobs.validate import validate_upload
from ghostline.llm import LLMError
from ghostline.scene import SceneInput, setup_tips
from ghostline.scene import analyze_scene as run_scene_analysis
from ghostline.scene.rescue import make_rescue_hooks
from ghostline.store.core_schemas import SceneAnalysis, TrackedObject
from ghostline.store.records import Attachment, Experiment, Ref, TrackRecord, Trial, Trim
from ghostline.traj import TrackSeries, TrajectoryError, build_trajectory
from ghostline.vision import (
    Track,
    TrackingFailed,
    read_events_jsonl,
    segment_sheet,
    track_from_events,
    track_object,
    write_tracks_parquet,
)
from ghostline.vision.frames import jpeg_stills, video_size

log = logging.getLogger("ghostline.jobs")

SHEET_KEY = "sheet"  # the TrackRecord for the calibration sheet
DEFAULT_SHEET_PROMPT = "white sheet of paper"
STILL_MAX_SIDE = 1280  # the three stills Wonder gets are capped at this long side
HINTS = {
    "free_fall": "something falling",
    "pendulum": "a pendulum",
    "ramp": "an object on a ramp",
    "bounce": "a bouncing ball",
    "spin_down": "a spinner slowing down",
}
PAPER_NAMES = {"a4": "A4", "letter": "Letter"}


# --- small helpers ---------------------------------------------------------------------------
def file_key(key: str) -> str:
    """A scene's object key as a safe file name (keys come from a model)."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("._-")[:64]
    return cleaned or "object"


def _load_timestamps(ctx: JobContext, trial: Trial) -> np.ndarray:
    path = ctx.paths.trial_file(trial.experiment_id, trial.id, "timestamps.npy")
    if not path.is_file():
        raise StageError(
            "The prepared clip is missing. Upload it again.", code="no_master", retryable=False
        )
    return np.load(path)


def _rel(ctx: JobContext, trial: Trial, path: Path) -> str:
    return ctx.paths.rel(trial.experiment_id, path)


def _capture_label(experiment: Experiment, trial: Trial) -> str:
    timing = trial.timing or {}
    fps = timing.get("real_fps") or experiment.declared_capture.fps
    slow = experiment.declared_capture.mode == "slow_mo"
    rate = f"{fps:.0f} fps " if fps else ""
    return rate + ("slow-mo" if slow else "normal video")


def _f35_from_tags(tags: dict[str, str]) -> float | None:
    """The 35 mm-equivalent focal length, when the phone wrote one into the file's tags."""
    for key, value in tags.items():
        if "35mm" in key:
            try:
                parsed = float(str(value).split()[0])
            except (ValueError, IndexError):
                continue
            if 10 <= parsed <= 400:
                return parsed
    return None


# --- ingest ----------------------------------------------------------------------------------
def ingest(ctx: JobContext, trial: Trial) -> IngestResult:
    """Part 5: validate, probe, build the timing model, write master.mp4 and the derived files,
    suggest a trim, and put the clip on the Meta Files API."""
    experiment = ctx.repo.get_experiment(trial.experiment_id)
    declared = experiment.declared_capture if experiment else None
    original = ctx.paths.experiment_dir(trial.experiment_id) / trial.paths["original"]
    trial_dir = ctx.paths.trial_dir(trial.experiment_id, trial.id)
    if not original.is_file():
        raise StageError(
            "The uploaded file is missing. Upload it again.", code="no_original", retryable=False
        )
    try:
        validate_upload(original, ctx.settings)
    except StageError:
        original.unlink(missing_ok=True)  # nothing here is worth keeping
        raise
    ctx.progress(0.05)

    try:
        done = ingest_trial(
            original,
            trial_dir,
            IngestDeclared(mode=declared.mode, fps=declared.fps) if declared else None,
        )
    except IngestError as exc:
        raise StageError(str(exc), code="bad_clip", retryable=False) from exc
    ctx.progress(0.7)

    t = done.timestamps
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    duration = float(t[-1] - t[0] + dt)
    suggestion = done.trim
    end_s = float(t[suggestion.stop_frame]) if suggestion.stop_frame < len(t) else float(t[-1] + dt)
    trim = Trim(
        start_s=suggestion.start_t,
        end_s=end_s,
        suggested_start_s=suggestion.start_t,
        suggested_end_s=end_s,
        source="auto",
    )
    warnings = [notice.message for notice in done.warnings]

    meta_file_ids: dict[str, str] = {}
    try:
        meta_file_ids = _upload_for_analysis(ctx, trial, trial_dir, len(t), suggestion)
    except (StageError, LLMError) as exc:
        # The clip is usable without the upload: analysis uploads again if it has to.
        log.warning("Meta upload at ingest failed", exc_info=True)
        warnings.append(
            f"Couldn't send the clip to Wonder yet ({exc}). It will be sent on Analyze."
        )
    ctx.progress(1.0)

    paths = {
        "master": "master.mp4",
        "playback_720": "playback_720.mp4",
        "wonder_480": "wonder_480.mp4",
        "poster": "poster.jpg",
        "timestamps": "timestamps.npy",
        "probe": "probe.json",
        "timing": "timing.json",
    }
    if (trial_dir / "audio.wav").is_file():
        paths["audio"] = "audio.wav"
    ctx.notebook.append(
        actor="system",
        kind="observation",
        title="Prepared the clip",
        body_md=(
            f"{done.probe.display_width}x{done.probe.display_height}, {len(t)} frames over "
            f"{duration:.2f} s ({done.timing.real_fps:.0f} fps, {done.timing.classification}). "
            f"Suggested analysis window: {trim.start_s:.2f} to {trim.end_s:.2f} s."
        ),
        data={
            "sha256": done.sha256,
            "timing": done.timing.model_dump(mode="json"),
            "normalize": done.normalize.model_dump(mode="json"),
        },
    )
    ctx.commit()
    return IngestResult(
        probe=done.probe.model_dump(mode="json"),
        timing=done.timing.model_dump(mode="json"),
        n_frames=len(t),
        duration_s=duration,
        paths={name: _rel(ctx, trial, trial_dir / file) for name, file in paths.items()},
        meta_file_ids=meta_file_ids,
        trim=trim,
        warnings=warnings,
    )


def _upload_for_analysis(ctx, trial, trial_dir, n_frames, suggestion) -> dict[str, str]:
    """Part 5, step 8: Wonder's 480p clip and the master trimmed to the suggested window."""
    window = media.Window(suggestion.start_frame, suggestion.stop_frame)
    stored = ctx.repo.get_trial(trial.id) or trial
    ids = {}
    ids[media.WONDER_KEY], _ = media.ensure_uploaded(
        ctx, stored, media.WONDER_KEY, trial_dir / "wonder_480.mp4"
    )
    clip = media.ensure_sam_clip(ctx, stored, window, n_frames)
    ids[window.key], _ = media.ensure_uploaded(ctx, stored, window.key, clip)
    return ids


# --- analyze_scene ---------------------------------------------------------------------------
def analyze_scene(ctx: JobContext, experiment: Experiment, trial: Trial) -> SceneAnalysis:
    """Part 6: Wonder watches the clip and decides what to measure."""
    trial_dir = ctx.paths.trial_dir(trial.experiment_id, trial.id)
    t_real = _load_timestamps(ctx, trial)
    window = media.analysis_window(trial, t_real, ctx.settings)
    with llm_errors():
        wonder_id, _ = media.ensure_uploaded(
            ctx, trial, media.WONDER_KEY, trial_dir / "wonder_480.mp4"
        )
        ctx.progress(0.2)
        indices = [window.start, (window.start + window.stop - 1) // 2, window.stop - 1]
        stills = jpeg_stills(trial_dir / "master.mp4", indices, max_side=STILL_MAX_SIDE)
        scene_input = SceneInput(
            duration_s=round(trial.duration_s or float(t_real[-1]), 2),
            capture=_capture_label(experiment, trial),
            hint=HINTS.get(experiment.system_type, ""),
            description=experiment.description,
            video_file_id=wonder_id,
            stills=stills,
            still_times_s=[round(float(t_real[i]), 3) for i in indices],
        )
        result = run_scene_analysis(ctx.meta, scene_input, experiment_id=str(experiment.id))
    ctx.progress(0.9)

    analysis = result.analysis
    scene = SceneAnalysis.model_validate(analysis.model_dump(mode="json"))
    if not analysis.proceed:
        tips = setup_tips(analysis)
        scene.reason_if_not = (analysis.reason_if_not or "") + "\n\nTips:\n- " + "\n- ".join(tips)
    _write_scene_notes(ctx, scene, result.prompt_version, result.cost_usd)
    ctx.commit()
    return scene


def _write_scene_notes(ctx: JobContext, scene: SceneAnalysis, prompt_version: str, cost: float):
    props = scene.object_properties
    detail = ", ".join(
        f"{k.replace('_', ' ')}: {v}" for k, v in props.items() if v not in (None, "")
    )
    ctx.notebook.append(
        actor="wonder",
        kind="observation",
        title="What Wonder saw",
        body_md=scene.description + (f"\n\n{detail}." if detail else ""),
        data={
            "system_type": scene.system_type,
            "tracked_objects": [o.model_dump() for o in scene.tracked_objects],
            "proceed": scene.proceed,
            "prompt_version": prompt_version,
            "cost_usd": cost,
        },
    )
    if scene.initial_hypotheses:
        lines = [
            f"{i}. **{h.get('name', '')}**: {h.get('mechanism', '')} "
            f"It would show as: {h.get('observable_signature', '')}"
            for i, h in enumerate(scene.initial_hypotheses, 1)
        ]
        ctx.notebook.append(
            actor="wonder",
            kind="hypothesis",
            title="First guesses, before any measurement",
            body_md="\n".join(lines),
            data={"hypotheses": scene.initial_hypotheses},
        )
    for issue in scene.quality_issues:
        if str(issue.get("severity")) == "major":
            ctx.notebook.warning(
                str(issue.get("issue", "Quality issue")), str(issue.get("note", ""))
            )


# --- track_objects ---------------------------------------------------------------------------
def _sam_cache_key(trial: Trial, window: media.Window, prompt: str) -> str:
    return media.cache_identity(trial, f"{window.key}:{prompt}")


def _track_one(ctx, trial, obj, *, window, file_id, identity, frame_size, hooks, sam_dir):
    """The Track for one object: read back from a saved SAM stream when the same request was
    already made for this clip (a retry), else asked of SAM."""
    cache_key = _sam_cache_key(trial, window, obj.sam_prompt)
    events = ctx.paths.sam_events(trial.experiment_id, trial.id, file_key(obj.key))
    known = {t.object_key: t for t in ctx.repo.list_tracks([trial.id])}.get(obj.key)
    if known and known.cache_key == cache_key and events.is_file():
        saved = track_from_events(read_events_jsonl(events), key=obj.key, frame_size=frame_size)
        if saved is not None:
            stats = {**known.stats, "from_cache": True}
            return saved, obj.sam_prompt, stats
    found = track_object(
        ctx.meta,
        file_id=file_id,
        key=file_key(obj.key),
        prompt=obj.sam_prompt,
        frame_size=frame_size,
        n_frames=window.n,
        media_sha256=identity,
        rescue=hooks,
        events_dir=sam_dir,
        experiment_id=str(trial.experiment_id),
    )
    stats = {
        "attempts": found.attempts,
        "cost_usd": found.cost_usd,
        "n_candidates": len(found.candidates),
        "from_cache": found.sam.cache_hit,
    }
    return found.track.renamed(obj.key), found.prompt, stats


def _track_stats(ctx, trial, obj: TrackedObject, track: Track, prompt: str, extra: dict) -> dict:
    return {
        "role": obj.role,
        "frames": len(track),
        "first_frame": int(track.frame[0]),
        "last_frame": int(track.frame[-1]),
        "extent_px": track.extent,
        "radius_px": track.radius,
        "confidence": track.mean_confidence if len(track) else None,
        "n_border": int(track.touches_border.sum()),
        "n_stitched": int(track.stitched.sum()),
        "prompt_used": prompt,
        "events_path": _rel(
            ctx, trial, ctx.paths.sam_events(trial.experiment_id, trial.id, file_key(obj.key))
        ),
        **extra,
    }


def _is_sheet(obj: TrackedObject, sheet_prompt: str) -> bool:
    """Wonder sometimes lists the sheet of paper among the objects to track (it belongs in
    `calibration`). It doesn't move, so tracking it as a body would waste a SAM request."""
    words = set(re.findall(r"[a-z]+", f"{obj.key} {obj.sam_prompt}".lower()))
    same_prompt = obj.sam_prompt.strip().lower() == sheet_prompt.strip().lower()
    return same_prompt or {"sheet", "paper"} <= words


def _sheet_prompt(scene: SceneAnalysis) -> str:
    return str(scene.calibration.get("sam_prompt") or DEFAULT_SHEET_PROMPT)


def _segment_sheet_ladder(ctx, trial, clip, window, prompts: list[str]):
    """Segment the sheet with Wonder's prompt, then with the plain one that Day 0 verified."""
    tried = []
    for prompt in dict.fromkeys(prompts):
        try:
            sheet = segment_sheet(
                ctx.meta,
                clip,
                prompt=prompt,
                n_frames=window.n,
                experiment_id=str(trial.experiment_id),
            )
        except TrackingFailed:
            tried.append(prompt)
        else:
            return sheet, prompt
    raise StageError(
        "Ghostline couldn't find the sheet of paper. Make sure the whole sheet is in view against "
        f"a contrasting background (tried: {', '.join(tried)}).",
        code="sheet_not_found",
        retryable=False,
    )


def _probe_corners(sheet) -> tuple[list, list[int]]:
    """Sheet corners found in the first, middle and last frames (Part 8's camera-motion check)."""
    corners, frames = [], []
    for probe in sheet.probes:
        try:
            found = detect_sheet_corners(probe.mask)
        except Exception:  # a probe frame the object covers badly just isn't compared
            log.warning("sheet corners failed on a probe frame", exc_info=True)
            continue
        corners.append(np.asarray(found.corners_px).round(3).tolist())
        frames.append(probe.frame or 0)
    return corners, frames


def track_objects(ctx: JobContext, trial: Trial, scene: SceneAnalysis) -> TrackingResult:
    """Part 7: SAM 3.1 tracks each object and segments the calibration sheet."""
    trial_dir = ctx.paths.trial_dir(trial.experiment_id, trial.id)
    t_real = _load_timestamps(ctx, trial)
    n_frames = len(t_real)
    window = media.analysis_window(trial, t_real, ctx.settings)
    sam_dir = trial_dir / "sam"
    calib_dir = trial_dir / "calib"
    records: list[TrackRecord] = []
    tracks: list[Track] = []

    with llm_errors():
        clip = media.ensure_sam_clip(ctx, trial, window, n_frames)
        file_id, identity = media.ensure_uploaded(ctx, trial, window.key, clip)
        frame_size = video_size(clip)
        middle = window.n // 2
        hook_frames = {middle: jpeg_stills(clip, [middle], max_side=STILL_MAX_SIDE)[0]}
        ctx.progress(0.05)

        wonder_sheet_prompt = _sheet_prompt(scene)
        movers = [
            o
            for o in scene.tracked_objects
            if o.role != "surface" and not _is_sheet(o, wonder_sheet_prompt)
        ]
        total = len(movers) + 2
        for done, obj in enumerate(movers):
            hooks = make_rescue_hooks(
                ctx.meta,
                description=f"{obj.sam_prompt} ({obj.why})",
                frames=hook_frames,
                frame_size=frame_size,
                experiment_id=str(trial.experiment_id),
            )
            try:
                track, used, extra = _track_one(
                    ctx, trial, obj, window=window, file_id=file_id,
                    identity=identity, frame_size=frame_size, hooks=hooks, sam_dir=sam_dir,
                )  # fmt: skip
            except TrackingFailed as exc:
                if obj is movers[0] or (obj.role == "moving_body" and not tracks):
                    raise StageError(str(exc), code="object_not_found", retryable=False) from exc
                ctx.notebook.warning(
                    f"Couldn't track the {obj.key}",
                    f"SAM didn't find '{obj.sam_prompt}'. The analysis goes on without it.",
                )
                continue
            track = track.shifted(window.start)
            tracks.append(track)
            records.append(
                TrackRecord(
                    trial_id=trial.id,
                    object_key=obj.key,
                    sam_prompt=used,
                    cache_key=_sam_cache_key(trial, window, obj.sam_prompt),
                    stats=_track_stats(ctx, trial, obj, track, used, extra),
                )
            )
            ctx.progress((done + 1) / total)

        parquet = write_tracks_parquet(trial_dir / "tracks.parquet", tracks, t_real)
        rel_parquet = _rel(ctx, trial, parquet)
        for record in records:
            record.parquet_path = rel_parquet

        sheet, sheet_prompt = _segment_sheet_ladder(
            ctx, trial, clip, window, [wonder_sheet_prompt, DEFAULT_SHEET_PROMPT]
        )
        sheet.write(calib_dir)
        corners, probe_frames = _probe_corners(sheet)
        records.append(
            TrackRecord(
                trial_id=trial.id,
                object_key=SHEET_KEY,
                sam_prompt=sheet_prompt,
                stats={
                    "role": "sheet",
                    "mask_path": _rel(ctx, trial, calib_dir / "sheet_mask.png"),
                    "median_path": _rel(ctx, trial, calib_dir / "median.png"),
                    "box": list(sheet.median.box),
                    "confidence": sheet.median.confidence,
                    "area_px": sheet.median.area,
                    "frame_size": list(sheet.frame_size),
                    "probe_corners_px": corners,
                    "probe_frames": [window.start + f for f in probe_frames],
                    "cost_usd": sheet.cost_usd,
                },
            )
        )
        ctx.progress((len(movers) + 1) / total)

        for obj in (o for o in scene.tracked_objects if o.role == "surface"):
            records.append(_segment_surface(ctx, trial, obj, clip, window, calib_dir))

    ctx.progress(1.0)
    ctx.commit()
    return TrackingResult(tracks=records)


def _segment_surface(ctx, trial, obj, clip, window, calib_dir) -> TrackRecord:
    """A ramp board: segmented once, like the sheet. The calibration reads its top edge."""
    folder = calib_dir / f"surface_{file_key(obj.key)}"
    stats: dict = {"role": "surface"}
    try:
        found = segment_sheet(
            ctx.meta, clip, prompt=obj.sam_prompt, n_frames=window.n,
            experiment_id=str(trial.experiment_id),
        )  # fmt: skip
    except TrackingFailed:
        ctx.notebook.warning(
            f"Couldn't find the {obj.key}", "The ramp angle will come from the object's path."
        )
    else:
        found.write(folder)
        stats["mask_path"] = _rel(ctx, trial, folder / "sheet_mask.png")
        stats["cost_usd"] = found.cost_usd
    return TrackRecord(
        trial_id=trial.id, object_key=obj.key, sam_prompt=obj.sam_prompt, stats=stats
    )


# --- calibrate -------------------------------------------------------------------------------
def _read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise StageError("A saved mask is missing. Press Retry.", code="no_mask")
    return (image > 127).astype(np.uint8)


def calibrate(
    ctx: JobContext,
    experiment: Experiment,
    trial: Trial,
    scene: SceneAnalysis,
    tracking: TrackingResult,
) -> CalibrationResult:
    """Part 8: the sheet's corners, the homography, the plane-offset correction, which way is
    down, and the uncertainty budget."""
    exp_dir = ctx.paths.experiment_dir(trial.experiment_id)
    trial_dir = ctx.paths.trial_dir(trial.experiment_id, trial.id)
    by_key = {r.object_key: r for r in tracking.tracks}
    sheet = by_key.get(SHEET_KEY)
    if sheet is None or "mask_path" not in sheet.stats:
        raise StageError("The sheet was not segmented.", code="sheet_not_found", retryable=False)
    mask = _read_mask(exp_dir / sheet.stats["mask_path"])
    median = cv2.imread(str(exp_dir / sheet.stats["median_path"]))
    if median is None:
        raise StageError("The median image is missing. Press Retry.", code="no_mask")
    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    size = (int(median.shape[1]), int(median.shape[0]))

    ramp_mask = None
    if scene.system_type == "ramp":
        surface = next((r for r in tracking.tracks if r.stats.get("role") == "surface"), None)
        if surface is not None and "mask_path" in surface.stats:
            ramp_mask = _read_mask(exp_dir / surface.stats["mask_path"])
    corner_sets = None
    probes = sheet.stats.get("probe_corners_px") or []
    if len(probes) >= 2:
        corner_sets = np.asarray(probes, float)

    options = {
        "sheet_mask": mask,
        "gray": gray,
        "plane_offset_m": experiment.setup.plane_offset_cm / 100.0,
        "f35_mm": _f35_from_tags((trial.probe or {}).get("tags", {})),
        "ramp_mask": ramp_mask,
        "corner_sets": corner_sets,
        "timing": trial.timing,
    }
    ctx.progress(0.2)
    calib = run_calibration(size, paper="auto", **options)
    if calib.paper_auto_confident is False:
        # The sheet's shape can't tell A4 from Letter at this angle: use what the user said.
        calib = run_calibration(size, paper=PAPER_NAMES[experiment.setup.paper], **options)
    if calib.camera_motion is not None and calib.camera_motion.status == "stop":
        raise StageError(
            f"The camera moved {calib.camera_motion.max_shift_px:.1f} px during the clip, which "
            "is too much to measure. Film again with the phone on a tripod.",
            code="camera_moved",
            retryable=False,
        )
    ctx.progress(0.7)

    calib_dir = trial_dir / "calib"
    write_calibration(calib, calib_dir, median)
    paths = {
        "calibration": _rel(ctx, trial, calib_dir / "calibration.json"),
        "calib_overlay": _rel(ctx, trial, calib_dir / "calib_overlay.png"),
        "rectified": _rel(ctx, trial, calib_dir / "rectified.png"),
    }
    sigma = calib.uncertainty.sigma_scale_rel
    ctx.notebook.append(
        actor="harness",
        kind="observation",
        title="Calibrated the scale from the sheet",
        body_md=(
            f"{calib.paper} sheet found ({calib.corner_method}). One pixel at the centre is "
            f"{calib.mm_per_px_center * calib.offset.scale:.3f} mm in the plane of motion, "
            f"uncertain by {sigma:.2%}. Down is taken from the {calib.gravity.source}."
        ),
        data={
            "mm_per_px": calib.mm_per_px_center * calib.offset.scale,
            "sigma_scale_rel": sigma,
            "gravity": calib.gravity.model_dump(),
            "offset": calib.offset.model_dump(),
        },
        attachments=[
            Attachment(path=paths["calib_overlay"], kind="image", caption="Sheet and 5 cm grid")
        ],
        refs=[Ref(kind="trial", id=str(trial.id))],
    )
    for notice in calib.notices:
        if notice.severity != "info":
            ctx.notebook.warning(notice.code.replace("_", " ").capitalize(), notice.message)
    ctx.commit()
    return CalibrationResult(calibration=calib.model_dump(mode="json"), paths=paths)


# --- extract_trajectory ----------------------------------------------------------------------
def _series(table: pd.DataFrame, key: str | None) -> TrackSeries | None:
    if key is None or key not in set(table["object_key"]):
        return None
    return TrackSeries.from_dataframe(table, key)


def _choose(scene: SceneAnalysis, tracked: set[str]) -> tuple[str, str | None]:
    """(key of the object to follow, key of the pivot or hub): the first moving body, except a
    spinner, whose angle comes from the marker on it."""
    by_role = {role: [o.key for o in scene.tracked_objects if o.role == role] for role in
               ("moving_body", "marker", "pivot")}  # fmt: skip
    present = {role: [k for k in keys if k in tracked] for role, keys in by_role.items()}
    pivot = present["pivot"][0] if present["pivot"] else None
    if scene.system_type == "spin_down" and present["marker"]:
        return present["marker"][0], pivot
    if present["moving_body"]:
        return present["moving_body"][0], pivot
    raise StageError("No moving object was tracked.", code="object_not_found", retryable=False)


def extract_trajectory(
    ctx: JobContext,
    experiment: Experiment,
    trial: Trial,
    scene: SceneAnalysis,
    tracking: TrackingResult,
    calibration: CalibrationResult,
) -> TrajectoryResult:
    """Part 9: tracks and calibration to clean metres, radians and real seconds."""
    trial_dir = ctx.paths.trial_dir(trial.experiment_id, trial.id)
    table = pd.read_parquet(trial_dir / "tracks.parquet")
    key, pivot_key = _choose(scene, set(table["object_key"]))
    system = scene.system_type
    if system == "other" and experiment.system_type not in ("auto", "other"):
        system = experiment.system_type
    try:
        result = build_trajectory(
            TrackSeries.from_dataframe(table, key),
            Calibration.model_validate(calibration.calibration),
            system,
            pivot=_series(table, pivot_key) if system == "pendulum" else None,
            hub=_series(table, pivot_key) if system == "spin_down" else None,
            timing=trial.timing,
        )
    except TrajectoryError as exc:
        raise StageError(
            f"{exc}. Check the mask overlay, and film again with the object fully in view.",
            code="too_little_data",
            retryable=False,
        ) from exc
    ctx.progress(0.8)

    summary = result.summary
    summary["object_key"] = key  # which track the trail on the video follows
    result.write(trial_dir)
    noise = summary["noise"]
    ctx.notebook.append(
        actor="harness",
        kind="observation",
        title="Extracted the trajectory",
        body_md=(
            f"{summary['n_points']} usable points over {summary['duration_s']:.2f} s "
            f"(of {summary['n_frames_in']} frames). Position noise is "
            f"{noise['sigma_pos_m'] * 1000:.2f} mm."
        ),
        data={
            "n_points": summary["n_points"],
            "noise": noise,
            "events": summary["events"],
            "n_dropped": summary["n_dropped"],
        },
        refs=[Ref(kind="trial", id=str(trial.id))],
    )
    for notice in result.notices:
        ctx.notebook.warning(notice.code.replace("_", " ").capitalize(), notice.message)
    ctx.commit()
    return TrajectoryResult(
        parquet_path=_rel(ctx, trial, trial_dir / "trajectory.parquet"), summary=summary
    )
