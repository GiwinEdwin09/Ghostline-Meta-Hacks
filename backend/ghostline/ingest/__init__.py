"""Probing, normalization, the timing model and auto-trim (build plan Part 5).

Main entry point: `ingest_trial(src, trial_dir, declared)`.
"""

from .frames import iter_frames, median_image, save_stills
from .models import DeclaredCapture, Notice, ProbeInfo, TimingModel, TrimSuggestion
from .normalize import trim_master
from .pipeline import IngestResult, assemble_chunks, ingest_trial
from .probe import IngestError, probe_file
from .timing import build_timing, real_timestamps

__all__ = [
    "DeclaredCapture",
    "IngestError",
    "IngestResult",
    "Notice",
    "ProbeInfo",
    "TimingModel",
    "TrimSuggestion",
    "assemble_chunks",
    "build_timing",
    "ingest_trial",
    "iter_frames",
    "median_image",
    "probe_file",
    "real_timestamps",
    "save_stills",
    "trim_master",
]
