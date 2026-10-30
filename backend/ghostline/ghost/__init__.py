"""Prove it: the ghost prediction, its overlay and the divergence revision (build plan Parts 16
and 17).

    from ghostline.ghost.stage import predict_ghost, diagnose_revision

`forecast.py` freezes trial 1's law and predicts trial 2, `divergence.py` measures the drift,
`sprite.py` and `render.py` draw the ghost video, and `diagnose.py` and `revise.py` are the
Part 17 loop. `stage.py` holds the two entry points the jobs call.
"""

from .model import Divergence, Forecast, FrozenFit, GhostError, GhostTrack, ICFit

__all__ = ["Divergence", "Forecast", "FrozenFit", "GhostError", "GhostTrack", "ICFit"]
