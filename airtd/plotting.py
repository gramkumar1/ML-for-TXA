"""Final OD map figure."""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import Config


def plot_od_maps(energy_ev: np.ndarray, delays_ps: np.ndarray,
                 maps: dict, cfg: Config, save_to: Optional[Path] = None,
                 show: bool = True):
    """Side-by-side OD maps over the zoom window; `maps` is {title: (D, P) array}."""
    zoom = (energy_ev >= cfg.ev_lo) & (energy_ev <= cfg.ev_hi)
    last = list(maps.values())[-1]
    vmax = float(np.percentile(np.abs(last[:, zoom]), 99.5)) * 1e3

    fig, axes = plt.subplots(1, len(maps), figsize=(6 * len(maps), 4.5), dpi=120,
                             squeeze=False)
    for ax, (title, data) in zip(axes[0], maps.items()):
        im = ax.pcolormesh(energy_ev[zoom], delays_ps, data[:, zoom] * 1e3,
                           shading="auto", cmap="seismic")
        im.set_clim(-vmax, vmax)
        ax.set_ylim(*cfg.delay_lim)
        ax.set_title(title)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Delay (ps)")
        plt.colorbar(im, ax=ax, label="mOD")
    fig.suptitle(f"{cfg.ev_lo:.1f}-{cfg.ev_hi:.1f} eV", y=1.02)
    fig.tight_layout()

    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig
