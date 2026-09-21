"""Hyperparameters for the airTD pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "transient_data"

# Google Drive ids, used only when a file is missing from the data directory.
GDRIVE_IDS = {
    "spectra_on_spectra_off.h5": "1auU7Y2Sa-jgE2JVsXCdTYXrqwp2WpN8t",
    "log_transients.h5": "1adFbDRslyuah50XbZLJpNBy6dRJNr8Ud",
    "calibration.h5": "1jhwgBcXt3QaE8p23fgB5QzDiuzSAFNQL",
}


@dataclass
class Config:
    # Calibration basis
    k_latent: int = 35          # principal components (paper's q)
    k_fa: int = 100             # factor-analysis components for the reliability prior
    val_fraction: float = 0.2
    seed: int = 0

    # Dictionary
    n_atoms: int = 140
    lam_init: float = 0.05

    # Training
    epochs: int = 150
    lr: float = 1e-3
    batch_size: int = 128
    p_clear: float = 0.25       # probability a training mask leaves every pixel observed
    mask_bands: tuple = (1, 3)
    mask_width: tuple = (20, 260)
    mask_soft_sigma: float = 4.0

    # Discriminator / inference
    airpcr_c: float = 0.2
    k_sweep: Sequence[int] = field(default_factory=lambda: [
        5, 10, 15, 20, 25, 30, 35, 38, 40, 42, 44, 46, 48, 50, 55, 60, 70, 81])
    # High k can drive the weighted ridge ill-conditioned and inflate NPR, so the
    # known-good values are pinned by default; set to None to select by argmax.
    k_opt_airpcr: Optional[int] = 46
    k_opt_airtd: Optional[int] = 44

    # Metrics
    test_delay_idx: int = 0     # reserved reference delay, excluded from NPR_avg
    edge_ev_lo: float = 55.0    # fixed edge-pixel referencing region (paper Sec. 4.1)
    edge_ev_hi: float = 65.4

    # Plotting
    ev_lo: float = 54.0
    ev_hi: float = 58.5
    delay_lim: tuple = (-200.0, 200.0)
