"""Synthetic phone-video generator (build plan Part 26).

Renders a textured background, an A4 sheet seen through a perspective camera and one of the five
systems with known parameters, with motion blur (averaged sub-frames) and sensor noise. Writes an
MP4 plus ground-truth JSON, per-frame object masks and a clean plate next to it.

    uv run python -m synth pendulum --out /tmp/synth --duration 3

Pixel coordinates follow the OpenCV convention (pixel centres at integer coordinates), and all
ground truth refers to the upright (displayed) image, whatever rotation tag the file carries.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from .camera import A4_M, Camera, sheet_corners_world
from .systems import Prim, make_system

SS = 4  # supersampling factor for dynamic shapes
TIME_BASE = Fraction(1, 90000)
PAPER_BGR = (238, 240, 240)


@dataclass
class ClipSpec:
    system: str = "free_fall"
    params: dict = field(default_factory=dict)  # overrides for the system's constructor
    duration_s: float = 3.0
    size: tuple[int, int] = (854, 480)  # upright (width, height)
    capture_fps: float = 60.0
    export_fps: float | None = None  # None: same as capture. 30 with 240 captured = slow-mo
    pts_jitter_frac: float = 0.0  # VFR jitter as a fraction of the frame period
    mixed_ramps: tuple[int, int, float] | None = None  # (head frames, tail frames, ramp fps)
    plane_offset_m: float = 0.02
    distance_m: float | None = None  # camera-to-sheet distance; None frames the scene
    azimuth_deg: float = 10.0
    elevation_deg: float = 4.0
    roll_deg: float = 0.0
    f35: float = 26.0
    sheet_center: tuple[float, float] = (0.0, 0.0)
    sheet_rot_deg: float = 0.0
    sheet_landscape: bool = False
    paper_m: tuple[float, float] = A4_M
    exposure_frac: float = 0.5
    subframes: int = 5
    sensor_sigma: float = 2.0  # grey levels
    shake_sigma_px: float = 0.0
    drift_px: tuple[float, float] = (0.0, 0.0)  # total camera drift over the clip
    rotation_tag: int = 0  # counter-clockwise degrees stored as a display matrix
    hdr: bool = False  # HEVC 10-bit tagged HLG BT.2020 (tags only; pixels stay SDR)
    crf: int = 18
    seed: int = 0


@dataclass
class ClipResult:
    video: Path
    truth: Path
    masks: Path
    plate: Path
    sheet_mask: Path
    gt: dict


def _project_prim(cam: Camera, p: Prim) -> dict:
    if p.kind == "disc":
        c = cam.project(p.center[None])[0]
        r = cam.f_px * p.radius / cam.depth(p.center[None])[0]
        return {"p": p, "c": c, "r": r}
    return {"p": p, "pts": cam.project(p.pts)}


def _bbox(q: dict) -> tuple[float, float, float, float]:
    if "c" in q:
        return q["c"][0] - q["r"], q["c"][1] - q["r"], q["c"][0] + q["r"], q["c"][1] + q["r"]
    pts = q["pts"]
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def _grid(roi, ss, box):
    """Sample-centre grid (cv coordinates) for the part of the ROI that a bbox covers."""
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    i0 = max(int(np.floor((box[0] - (x0 - 0.5)) * ss)), 0)
    i1 = min(int(np.ceil((box[2] - (x0 - 0.5)) * ss)) + 1, w * ss)
    j0 = max(int(np.floor((box[1] - (y0 - 0.5)) * ss)), 0)
    j1 = min(int(np.ceil((box[3] - (y0 - 0.5)) * ss)) + 1, h * ss)
    if i1 <= i0 or j1 <= j0:
        return None
    xs = (x0 - 0.5 + (np.arange(i0, i1) + 0.5) / ss).astype(np.float32)
    ys = (y0 - 0.5 + (np.arange(j0, j1) + 0.5) / ss).astype(np.float32)
    return (j0, j1, i0, i1), np.meshgrid(xs, ys)


def _disc_mask(cx, cy, r, roi, ss) -> np.ndarray:
    x0, y0, x1, y1 = roi
    out = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), bool)
    g = _grid(roi, ss, (cx - r, cy - r, cx + r, cy + r))
    if g is not None:
        (j0, j1, i0, i1), (xx, yy) = g
        out[j0:j1, i0:i1] = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return out


def _poly_mask(pts, roi, ss) -> np.ndarray:
    """Exact sub-sample test for a convex polygon (no boundary dilation, unlike cv2.fillPoly)."""
    x0, y0, x1, y1 = roi
    out = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), bool)
    g = _grid(roi, ss, (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    if g is None:
        return out
    (j0, j1, i0, i1), (xx, yy) = g
    area = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    sign = 1.0 if area >= 0 else -1.0
    inside = np.ones(xx.shape, bool)
    for k in range(len(pts)):
        a, b = pts[k], pts[(k + 1) % len(pts)]
        inside &= sign * ((b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0])) >= 0
    out[j0:j1, i0:i1] = inside
    return out


def _shape_mask(q: dict, roi, ss) -> np.ndarray:
    if q.get("c") is not None:
        return _disc_mask(q["c"][0], q["c"][1], q["r"], roi, ss)
    return _poly_mask(q["pts"], roi, ss)


def _paint(big: np.ndarray, q: dict, roi, ss) -> None:
    p: Prim = q["p"]
    base = np.array(p.color, np.float32)
    if q.get("c") is not None and p.shaded:
        cx, cy = q["c"]
        r = q["r"]
        rings = [
            (1.0, 0.0, 0.72 * base),
            (0.85, -0.12, base),
            (0.45, -0.28, np.minimum(255, 1.25 * base + 55)),
        ]
        for scale, off, col in rings:
            big[_disc_mask(cx + off * r, cy + off * r, scale * r, roi, ss)] = col
    else:
        big[_shape_mask(q, roi, ss)] = base


def _block_mean(a: np.ndarray, ss: int) -> np.ndarray:
    h, w = a.shape[0] // ss, a.shape[1] // ss
    return a.reshape(h, ss, w, ss, *a.shape[2:]).mean(axis=(1, 3))


def _render_roi(plate_roi: np.ndarray, shapes: list[dict], roi, tags: list[str]):
    """Composite shapes over a plate patch; returns (patch, {tag: coverage})."""
    big = np.repeat(np.repeat(plate_roi, SS, axis=0), SS, axis=1).astype(np.float32)
    masks = {t: np.zeros(big.shape[:2], bool) for t in tags}
    for q in shapes:
        _paint(big, q, roi, SS)
        if q["p"].tag in masks:
            masks[q["p"].tag] |= _shape_mask(q, roi, SS)
    return _block_mean(big, SS), {t: _block_mean(m.astype(np.float32), SS) for t, m in masks.items()}


def _texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    base = np.zeros((h, w), np.float32)
    for sigma, amp in ((40, 14.0), (12, 9.0), (3, 6.0)):
        n = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), sigma)
        base += amp * n / max(float(n.std()), 1e-6)
    gray = 62.0 + base
    return np.stack([gray * 0.92, gray, gray * 1.08], axis=2).clip(0, 255).astype(np.float32)


def _coverage(shapes: list[dict], shape: tuple[int, int], ss: int = 8) -> np.ndarray:
    h, w = shape
    union = np.zeros((h * ss, w * ss), bool)
    for q in shapes:
        union |= _shape_mask(q, (0, 0, w, h), ss)
    return _block_mean(union.astype(np.float32), ss)


def _composite(plate: np.ndarray, shapes: list[dict], color) -> np.ndarray:
    a = _coverage(shapes, plate.shape[:2])[..., None]
    return plate * (1 - a) + np.array(color, np.float32) * a


def _auto_camera(spec: ClipSpec, system, corners: np.ndarray) -> Camera:
    w, h = spec.size
    t = np.arange(0, spec.duration_s, 1 / 30)
    pts = [corners[:, :2]]
    for arr in system.truth(t)["tracks"].values():
        pts.append(arr[:, :2])
    for p in system.static():
        if p.kind == "poly" and p.tag != "floor":
            pts.append(p.pts[:, :2])
    allp = np.vstack(pts)
    lo, hi = allp.min(0), allp.max(0)
    centre = (lo + hi) / 2
    span = np.maximum(hi - lo, 1e-3) + 0.10
    f_px = w * spec.f35 / 36.0
    dist = spec.distance_m or max(span[0] * f_px / (0.88 * w), span[1] * f_px / (0.88 * h)) * 1.08
    for _ in range(8):
        cam = Camera.look_at(
            w, h, tuple(centre), dist, spec.f35, spec.azimuth_deg, spec.elevation_deg, spec.roll_deg
        )
        px = cam.project(np.column_stack([allp, np.full(len(allp), system.d)]))
        if spec.distance_m or (
            px.min() > 6 and px[:, 0].max() < w - 7 and px[:, 1].max() < h - 7
        ):  # fmt: skip
            return cam
        dist *= 1.08
    return cam


def _file_pts(spec: ClipSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    """Presentation time of every frame in file seconds."""
    dt = 1.0 / (spec.export_fps or spec.capture_fps)
    if spec.mixed_ramps:
        head, tail, ramp_fps = spec.mixed_ramps
        steps = np.full(n, dt)
        steps[:head] = 1.0 / ramp_fps
        if tail:
            steps[n - tail :] = 1.0 / ramp_fps
        pts = np.concatenate([[0.0], np.cumsum(steps[:-1])])
    else:
        pts = np.arange(n) * dt
    if spec.pts_jitter_frac:
        jitter = rng.uniform(-1, 1, n) * spec.pts_jitter_frac * dt
        pts = np.maximum.accumulate(pts + jitter - jitter[0])
        pts += np.arange(n) * 1e-9
    return pts


def _write_video(path: Path, frames, pts: np.ndarray, size: tuple[int, int], fps: float, crf: int):
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=Fraction(round(fps * 1000), 1000))
    stream.width, stream.height = size
    stream.pix_fmt = "yuv420p"
    stream.codec_context.time_base = TIME_BASE
    stream.codec_context.options = {"crf": str(crf), "preset": "veryfast", "bf": "0"}
    for i, bgr in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = int(round(pts[i] / float(TIME_BASE)))
        frame.time_base = TIME_BASE
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def _postprocess(path: Path, spec: ClipSpec) -> None:
    """HDR/HEVC tagging and rotation metadata, both via ffmpeg."""
    if not (spec.hdr or spec.rotation_tag):
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for the hdr and rotation_tag options")
    tmp = path.with_name(path.stem + "_pp.mp4")
    if spec.hdr:
        _ffmpeg(
            "-i", str(path), "-c:v", "libx265", "-preset", "ultrafast", "-crf", str(spec.crf),
            "-pix_fmt", "yuv420p10le", "-color_primaries", "bt2020", "-color_trc", "arib-std-b67",
            "-colorspace", "bt2020nc", "-tag:v", "hvc1", "-fps_mode", "passthrough",
            "-x265-params", "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:log-level=error",
            str(tmp),
        )  # fmt: skip
        tmp.replace(path)
    if spec.rotation_tag:
        _ffmpeg("-display_rotation", str(spec.rotation_tag), "-i", str(path), "-c", "copy", str(tmp))
        tmp.replace(path)


def _stats(mask: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[float, np.ndarray | None]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, None
    return float(len(xs)), np.array([xs.mean() + roi[0], ys.mean() + roi[1]])


def render_clip(spec: ClipSpec, out_dir: str | Path, name: str | None = None) -> ClipResult:
    """Render one clip and its ground truth into `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = name or spec.system
    rng = np.random.default_rng(spec.seed)
    system = make_system(spec.system, spec.plane_offset_m, **spec.params)
    w, h = spec.size
    n = int(round(spec.duration_s * spec.capture_fps))
    t_cap = np.arange(n) / spec.capture_fps
    corners_w = sheet_corners_world(
        spec.sheet_center, spec.sheet_rot_deg, spec.sheet_landscape, spec.paper_m
    )
    cam0 = _auto_camera(spec, system, corners_w)

    shift = np.zeros((n, 2))
    if spec.shake_sigma_px:
        shift += rng.normal(0, spec.shake_sigma_px, (n, 2))
    shift += np.linspace(0, 1, n)[:, None] * np.asarray(spec.drift_px)[None]
    moving_cam = bool(np.any(shift))

    # Static plate: texture, lit sheet, static props.
    plate = _texture(h, w, rng)
    sheet_px = cam0.project(corners_w)
    sheet_shape = [{"p": Prim("poly", "sheet", PAPER_BGR, pts=corners_w), "pts": sheet_px}]
    cov = _coverage(sheet_shape, (h, w))[..., None]
    light = (0.93 + 0.07 * np.linspace(0, 1, w, dtype=np.float32))[None, :, None]
    plate = plate * (1 - cov) + np.array(PAPER_BGR, np.float32) * light * cov
    for prim in system.static():
        plate = _composite(plate, [_project_prim(cam0, prim)], prim.color)
    plate_u8 = plate.clip(0, 255).astype(np.uint8)
    sheet_mask = ((cov[..., 0] > 0.5) * 255).astype(np.uint8)

    # Ground truth that does not need rendering.
    truth = system.truth(t_cap)
    keys = list(truth["tracks"])
    tags = {"object": "object", "marker": "marker", "hub": "hub"}
    expo = spec.exposure_frac / spec.capture_fps
    sub_dt = expo * ((np.arange(spec.subframes) + 0.5) / spec.subframes - 0.5)
    pts_file = _file_pts(spec, n, rng)

    centroid = {k: np.zeros((n, 2)) for k in keys}
    mask_centroid = {k: np.full((n, 2), np.nan) for k in keys}
    area = {k: np.zeros(n) for k in keys}
    packed = {k: [] for k in keys if k in tags}
    corners_px = np.zeros((n, 4, 2))
    rot_k = -(spec.rotation_tag // 90)

    def frames():
        for i in range(n):
            cam = cam0.with_shift(*shift[i]) if moving_cam else cam0
            base = plate
            if moving_cam:
                m = np.float32([[1, 0, shift[i][0]], [0, 1, shift[i][1]]])
                base = cv2.warpAffine(plate, m, (w, h), borderMode=cv2.BORDER_REFLECT)
            corners_px[i] = cam.project(corners_w)
            for k in keys:
                centroid[k][i] = cam.project(truth["tracks"][k][i : i + 1])[0]
            subs = [[_project_prim(cam, p) for p in system.dynamic(t_cap[i] + dt)] for dt in sub_dt]
            boxes = np.array([_bbox(q) for s in subs for q in s])
            frame = base.copy()
            roi_masks = {}
            if len(boxes):
                x0 = int(max(np.floor(boxes[:, 0].min()) - 3, 0))
                y0 = int(max(np.floor(boxes[:, 1].min()) - 3, 0))
                x1 = int(min(np.ceil(boxes[:, 2].max()) + 4, w))
                y1 = int(min(np.ceil(boxes[:, 3].max()) + 4, h))
                if x1 > x0 and y1 > y0:
                    roi = (x0, y0, x1, y1)
                    present = sorted({q["p"].tag for s in subs for q in s} & set(tags.values()))
                    acc = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
                    cov_acc = {t: 0.0 for t in present}
                    for shapes in subs:
                        patch, cov = _render_roi(base[y0:y1, x0:x1], shapes, roi, present)
                        acc += patch
                        for t in present:
                            cov_acc[t] = cov_acc[t] + cov[t]
                    frame[y0:y1, x0:x1] = acc / len(subs)
                    for k in keys:
                        if k in tags and tags[k] in cov_acc:
                            roi_masks[k] = (cov_acc[tags[k]] / len(subs) > 0.5, roi)
            for k in keys:
                if k in tags:
                    full = np.zeros((h, w), bool)
                    if k in roi_masks:
                        m, (rx0, ry0, rx1, ry1) = roi_masks[k]
                        full[ry0:ry1, rx0:rx1] = m
                        area[k][i], c = _stats(m, (rx0, ry0, rx1, ry1))
                        if c is not None:
                            mask_centroid[k][i] = c
                    packed[k].append(np.packbits(full.ravel()))
                else:  # static items such as a pivot: analytic
                    mask_centroid[k][i] = centroid[k][i]
            if spec.sensor_sigma:
                frame = frame + rng.normal(0, spec.sensor_sigma, frame.shape)
            bgr = frame.clip(0, 255).round().astype(np.uint8)
            yield np.ascontiguousarray(np.rot90(bgr, rot_k)) if rot_k else bgr

    video = out / f"{name}.mp4"
    fps_file = spec.export_fps or spec.capture_fps
    factor = fps_file / spec.capture_fps  # real seconds per file second
    stored = (h, w) if spec.rotation_tag in (90, 270) else (w, h)
    _write_video(video, frames(), pts_file, stored, fps_file, spec.crf)
    _postprocess(video, spec)

    masks_path = out / f"{name}.masks.npz"
    np.savez_compressed(
        masks_path, shape=np.array([h, w]), **{k: np.stack(v) for k, v in packed.items()}
    )
    plate_path = out / f"{name}.plate.png"
    cv2.imwrite(str(plate_path), plate_u8)
    sheet_path = out / f"{name}.sheet_mask.png"
    cv2.imwrite(str(sheet_path), sheet_mask)

    tracks = {}
    for k in keys:
        visible = np.zeros(n, bool)
        if k in tags:
            visible = area[k] > 0
        else:
            visible[:] = True
        tracks[k] = {
            "centroid_px": centroid[k].round(4).tolist(),
            "mask_centroid_px": np.nan_to_num(mask_centroid[k], nan=-1).round(4).tolist(),
            "area_px": area[k].tolist(),
            "visible": visible.tolist(),
            "xyz_m": truth["tracks"][k].round(6).tolist(),
        }
    head, tail, _ = spec.mixed_ramps or (0, 0, 0)
    gt = {
        "version": 1,
        "system": spec.system,
        "sam_prompt": system.sam_prompt,
        "primary": system.primary,
        "spec": {k: v for k, v in asdict(spec).items() if k != "params"},
        "params": system.params(),
        "events": system.events(),
        "plane_offset_m": system.d,
        "video": {
            "file": video.name, "width": w, "height": h, "n_frames": n,
            "stored_size": list(stored), "rotation_tag": spec.rotation_tag, "hdr": spec.hdr,
        },  # fmt: skip
        "timing": {
            "capture_fps": spec.capture_fps,
            "export_fps": fps_file,
            "time_factor": factor,
            "pts_mode": "mixed" if spec.mixed_ramps else "vfr" if spec.pts_jitter_frac else "cfr",
            "pts_s": pts_file.round(7).tolist(),
            "t_real_s": t_cap.tolist(),
            "usable_range": [head, n - tail],
        },
        "camera": {
            "f_px": cam0.f_px, "K": cam0.K.tolist(), "R": cam0.R.tolist(),
            "center_m": cam0.center.tolist(), "distance_m": float(np.linalg.norm(cam0.center)),
        },  # fmt: skip
        "sheet": {
            "size_m": list(spec.paper_m), "landscape": spec.sheet_landscape,
            "corners_world_m": corners_w[:, :2].tolist(),
            "corners_px": corners_px.round(4).tolist(),
            "mask_file": sheet_path.name, "plate_file": plate_path.name,
        },
        "tracks": tracks,
        "series": {k: np.asarray(v).round(8).tolist() for k, v in truth["series"].items()},
        "camera_shift_px": shift.round(4).tolist(),
        "masks_file": masks_path.name,
    }
    truth_path = out / f"{name}.json"
    truth_path.write_text(json.dumps(gt))
    return ClipResult(video, truth_path, masks_path, plate_path, sheet_path, gt)
