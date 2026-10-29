"""Numbers behind the "Done when" lines of build plan Parts 5, 8 and 9, on synthetic data.

    uv run python -m synth.validate [--cameras 24] [--seeds 10]

Takes about a minute. Not part of the test suite; the tests check the same things on fewer cases.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ghostline.calib import calibrate
from ghostline.calib.homography import local_scale, local_scale_many, order_corners
from ghostline.ingest import DeclaredCapture, ingest_trial
from ghostline.traj import TrackSeries, build_trajectory

from .render import ClipSpec, build_scene, render_clip
from .tracks import analytic_tracks
from .truth import GroundTruth

SIZE = (854, 480)


def part8(n_cameras: int) -> None:
    print(f"\nPart 8: scale error on {n_cameras} random perspective renders (854x480)")
    rng = np.random.default_rng(1)
    err = {"mask only": [], "mask + grey edge": []}
    dist_err, covered = [], []
    for _ in range(n_cameras):
        spec = ClipSpec(
            system="free_fall", seed=int(rng.integers(1000)),
            azimuth_deg=float(rng.uniform(-30, 30)), elevation_deg=float(rng.uniform(-10, 15)),
            roll_deg=float(rng.uniform(-3, 3)), sheet_rot_deg=float(rng.uniform(-20, 20)),
            sheet_landscape=bool(rng.random() < 0.3),
            sheet_center=(float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.05, 0.05))),
        )  # fmt: skip
        sc = build_scene(spec)
        gray = cv2.cvtColor(sc.plate_u8, cv2.COLOR_BGR2GRAY)
        inv = np.linalg.inv(sc.cam.plane_homography(0.02))
        centre = np.array([(SIZE[0] - 1) / 2, (SIZE[1] - 1) / 2])
        truth = 1000 * local_scale(inv, centre)
        for name, g in (("mask only", None), ("mask + grey edge", gray)):
            cal = calibrate(SIZE, sheet_mask=sc.sheet_mask > 0, gray=g, paper="A4")
            err[name].append(abs(cal.m_per_px * 1000 / truth - 1) * 100)
            if g is not None:
                xyz = np.column_stack([rng.uniform(-0.5, 0.5, 30), rng.uniform(-0.5, 0.5, 30)])
                xyz = np.column_stack([xyz, np.full(30, 0.02)])
                px = sc.cam.project(xyz)
                world = cal.px_to_world(px)
                iu = np.triu_indices(30, 1)
                d_true = np.linalg.norm(xyz[:, None, :2] - xyz[None, :, :2], axis=2)[iu]
                d_est = np.linalg.norm(world[:, None] - world[None], axis=2)[iu]
                dist_err.append(np.abs(d_est / d_true - 1).max() * 100)
                s_est = local_scale_many(cal.H_np, px) * cal.offset.scale
                s_true = 1000 * np.array([local_scale(inv, p) for p in px])
                actual = np.abs(np.mean(s_est / s_true) - 1)
                covered.append(actual <= 1.96 * cal.scale_sigma_at(px))
    for name, v in err.items():
        v = np.array(v)
        stats = f"median {np.median(v):.3f}%  p95 {np.percentile(v, 95):.3f}%  max {v.max():.3f}%"
        print(f"  {name:18s} scale error: {stats}")
    print(f"  distances over +-0.5 m (grey edge): worst pair {np.max(dist_err):.3f}%")
    print(
        f"  95% scale interval at the object contains the error in {np.mean(covered):.0%} of cases"
    )


def part9(n_seeds: int) -> None:
    print(f"\nPart 9: noise estimate and round trip, {n_seeds} seeds per cell")
    systems = {
        "free_fall": (2.0, 120.0, {}),
        "pendulum": (4.0, 60.0, {"params": {"gamma": 0.08, "theta0": 0.6}}),
        "ramp": (2.0, 60.0, {}),
        "bounce": (3.0, 120.0, {}),
        "spin_down": (4.0, 60.0, {"params": {"omega0": 30.0, "coulomb": 3.0, "viscous": 0.4}}),
    }
    print(
        f"  {'system':10s} {'sigma px':>8s} {'est/true':>9s} {'worst':>7s} {'pos rms/sigma':>14s}"
    )
    worst_all = 0.0
    for name, (dur, fps, extra) in systems.items():
        sc = build_scene(ClipSpec(system=name, duration_s=dur, capture_fps=fps, **extra))
        cal = calibrate(SIZE, corners_px=order_corners(sc.corners_px), paper="A4")
        for sigma_px in (0.15, 0.3, 0.6):
            ratios, rms = [], []
            for seed in range(n_seeds):
                a = analytic_tracks(sc, dur, fps, None, sigma_px, seed)
                track = TrackSeries.from_arrays(
                    a["frame"], a["cx"], a["cy"], a["area"], a["conf"], None, t=a["t"]
                )
                res = build_trajectory(track, cal, name)
                true_m = sigma_px * float(
                    np.mean(local_scale_many(cal.H_np, track.px)) * cal.offset.scale / 1000
                )
                ratios.append(res.summary["noise"]["sigma_pos_m"] / true_m)
                fr = res.trajectory["frame"].to_numpy()
                e = res.trajectory[["x", "y"]].to_numpy() - a["xyz_m"][fr, :2]
                rms.append(np.sqrt((e**2).sum(axis=1).mean()) / (true_m * np.sqrt(2)))
            dev = np.abs(np.array(ratios) - 1).max()
            worst_all = max(worst_all, dev)
            row = f"{np.mean(ratios):9.3f} {dev:7.1%} {np.mean(rms):14.2f}"
            print(f"  {name:10s} {sigma_px:8.2f} {row}")
    print(f"  worst single-run deviation of the noise estimate: {worst_all:.1%} (target 20%)")


def part5() -> None:
    print("\nPart 5: ingest of the three clip kinds in the plan's Done-when, plus VFR and mixed")
    if shutil.which("ffmpeg") is None:
        print("  ffmpeg not found, skipped")
        return
    tmp = Path(tempfile.mkdtemp(prefix="ghostline-validate-"))
    base = {"size": (320, 180), "params": {"radius": 0.035}}
    cases = {
        "iPhone HEVC HDR portrait": (
            dict(duration_s=1.5, **(base | {"size": (180, 320)}), hdr=True, rotation_tag=90),
            DeclaredCapture(mode="normal", fps=60),
        ),
        "Android (VFR, 60 fps)": (
            dict(duration_s=1.5, pts_jitter_frac=0.04, **base),
            DeclaredCapture(mode="normal", fps=60),
        ),
        "re-timed iPhone slow-mo": (
            dict(duration_s=1.2, capture_fps=240, export_fps=30, **base),
            DeclaredCapture(mode="slow_mo", fps=240),
        ),
        "mixed-rate slow-mo export": (
            dict(
                duration_s=1.0, capture_fps=240, export_fps=30, mixed_ramps=(40, 40, 240.0), **base
            ),
            DeclaredCapture(mode="slow_mo", fps=240),
        ),
    }
    try:
        for name, (kw, declared) in cases.items():
            gt = GroundTruth.load(
                render_clip(ClipSpec(system="free_fall", **kw), tmp, name[:6]).truth
            )
            res = ingest_trial(gt.video_path, tmp / f"{name[:6]}_trial", declared)
            # real time of each frame, from the file's own timestamps and the true time factor,
            # compared over the usable section (the whole clip unless the rate is mixed)
            lo, hi = res.timing.usable_start, res.timing.usable_stop
            expected = (gt.pts - gt.pts[0]) * gt.time_factor
            t_err = np.abs(
                (res.timestamps[lo:hi] - res.timestamps[lo]) - (expected[lo:hi] - expected[lo])
            ).max()
            print(
                f"  {name:26s} frames {res.normalize.n_frames}/{gt.n_frames}  "
                f"class {res.timing.classification:10s} factor {res.timing.time_factor:.4f} "
                f"(true {gt.time_factor:.4f})  time err {t_err * 1000:.3f} ms  "
                f"tonemap {res.normalize.tonemap}"
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cameras", type=int, default=24)
    p.add_argument("--seeds", type=int, default=10)
    a = p.parse_args()
    part8(a.cameras)
    part9(a.seeds)
    part5()


if __name__ == "__main__":
    main()
