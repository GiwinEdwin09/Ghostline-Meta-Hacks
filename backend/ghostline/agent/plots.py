"""The plot Wonder gets with its first prompt: the training window of the motion, nothing else."""

from __future__ import annotations

import io

import numpy as np

from .inputs import PhysicsInputs

_LABELS = {
    "x": "x (m)",
    "y": "y (m)",
    "s": "s along the ramp (m)",
    "theta": "theta (rad)",
    "phi": "phi (rad)",
}


def training_plot(inputs: PhysicsInputs) -> bytes:
    """PNG of every observable over the motion window, with the points after the end of the
    training window left out and that stretch shaded as hidden. Wonder sees this, not the data
    behind the shading."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, obs = inputs.arrays()
    train = inputs.train_mask
    names = list(obs)
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 3.2 * len(names)), dpi=100, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names, strict=True):
        ax.plot(t[train], obs[name][train], ".", ms=4, c="#1f4e79", label="measured (training)")
        ax.axvspan(inputs.t_split, t[-1], color="#bbbbbb", alpha=0.35, label="held out (hidden)")
        ax.axvline(inputs.t_split, c="#7f8c8d", ls="--", lw=1)
        ax.set_ylabel(_LABELS.get(name, name))
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_title("Measured motion: training window")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
