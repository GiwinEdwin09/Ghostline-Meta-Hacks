"""Fit and residual plots, 1000x600 PNG (build plan Part 11).

Each model gets up to four: `fit` and `resid` show everything (for the reviewer and the user),
`fit_train` and `resid_train` hide the held-out points (for Wonder, who never sees them).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..library.base import ModelDef
from .fitting import Dataset
from .schema import PlotRef

_COLORS = {"train": "#1f4e79", "test": "#c0392b", "fit": "#e67e22", "split": "#7f8c8d"}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _label(model: ModelDef, name: str) -> str:
    unit = model.observable_units.get(name, "")
    return f"{name} ({unit})" if unit else name


def _data_omega(t: np.ndarray, phi: np.ndarray, k: int = 3) -> np.ndarray:
    """Central-difference angular velocity over +/- k samples. Display only."""
    out = np.full(t.size, np.nan)
    if t.size > 2 * k:
        out[k:-k] = (phi[2 * k :] - phi[: -2 * k]) / (t[2 * k :] - t[: -2 * k])
    return out


def render_plots(
    model: ModelDef,
    ds: Dataset,
    p: dict[str, float],
    t_split: float,
    train_mask: np.ndarray,
    out_dir: Path,
    *,
    include_heldout: bool = True,
) -> list[PlotRef]:
    """Write the plots for one fit into `out_dir`. Returns their references (file names only)."""
    from .fitting import predict_all

    plt = _pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = float(ds.t[0])
    grid = np.unique(np.concatenate([ds.t, np.linspace(ds.t[0], ds.t[-1], 400)]))
    curve = predict_all(model, grid - t0, p)
    at_points = predict_all(model, ds.t - t0, p)
    omega_curve = getattr(model, "omega_curve", None)
    test_mask = ~train_mask
    refs: list[PlotRef] = []
    variants = [("", "reviewer", True)]
    if include_heldout:
        variants.append(("_train", "agent", False))
    for suffix, audience, show_test in variants:
        for kind in ("fit", "resid"):
            panels = [(name, None) for name in model.observables]
            if kind == "fit" and omega_curve is not None:
                panels.append(("omega", omega_curve))
            fig, axes = plt.subplots(
                len(panels), 1, figsize=(10, 6), dpi=100, sharex=True, squeeze=False
            )
            for row, (name, extra) in enumerate(panels):
                ax = axes[row, 0]
                if extra is not None:
                    y = _data_omega(ds.t, ds.obs[model.observables[0]])
                    ax.plot(grid, extra(grid - t0, p), "-", lw=1.6, c=_COLORS["fit"], label="model")
                    label = "omega (rad/s)"
                elif kind == "fit":
                    y = ds.obs[name]
                    ax.plot(
                        grid, curve[row], "-", lw=1.6, c=_COLORS["fit"], label=f"{model.id} fit"
                    )
                    label = _label(model, name)
                else:
                    y = ds.obs[name] - at_points[row]
                    ax.axhline(0, c="k", lw=0.8)
                    label = f"residual {_label(model, name)}"
                ax.plot(
                    ds.t[train_mask], y[train_mask], ".", ms=4, c=_COLORS["train"], label="training"
                )
                if show_test:
                    ax.plot(
                        ds.t[test_mask],
                        y[test_mask],
                        ".",
                        ms=4,
                        c=_COLORS["test"],
                        label="held-out",
                    )
                    ax.axvspan(t_split, ds.t[-1], color=_COLORS["test"], alpha=0.06)
                else:
                    ax.axvspan(t_split, ds.t[-1], color="#bbbbbb", alpha=0.25)
                ax.axvline(t_split, c=_COLORS["split"], ls="--", lw=1)
                ax.set_ylabel(label)
                ax.grid(alpha=0.25)
            axes[-1, 0].set_xlabel("time (s)")
            axes[0, 0].legend(loc="best", fontsize=8)
            what = "Data and fit" if kind == "fit" else "Residuals"
            window = "" if show_test else " (training window)"
            axes[0, 0].set_title(f"{model.id} {model.name}: {what}{window}")
            fig.tight_layout()
            plot_id = f"{model.id}.{kind}{suffix}"
            path = out_dir / f"{plot_id}.png"
            fig.savefig(path)
            plt.close(fig)
            refs.append(
                PlotRef(id=plot_id, path=path.name, kind=f"{kind}{suffix}", audience=audience)  # type: ignore[arg-type]
            )
    return refs
