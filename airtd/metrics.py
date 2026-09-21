"""Noise power reduction (NPR).

The paper reports two structurally different NPR numbers that must not be
conflated:

  (A) Eq. 6 / Fig. 4(e), the discriminator's own convergence diagnostic --
      weighted by (1 - w_k), airPCR's continuous iterative weight.
      Paper value at k=46: NPR_avg = 15.9 dB.
  (B) Sec. 5 / Fig. 6(i), the headline result -- a plain unweighted mean using
      the *fixed* edge-pixel referencing region (E < 55 eV or E > 65.4 eV).
      Paper values: 16.2 dB over all pixels, 16.5 dB outside the region.

Both average in linear units and take 10*log10 exactly once; averaging
already-logged per-pixel values is biased low by Jensen's inequality.
"""
from typing import Callable, Tuple

import numpy as np

from .config import Config
from .data import Dataset
from .dictionary import ridge_denoise


def _per_pixel_npr(ds: Dataset, fit_fn: Callable[[np.ndarray], np.ndarray],
                   test_delay_idx: int) -> np.ndarray:
    """Per-pixel NPR (linear), from shot-to-shot variance at each delay, averaged
    over delays excluding the reserved test delay."""
    d, r, c, p = ds.shape
    npr = np.empty((d, p), dtype=np.float64)
    for t in range(d):
        shots = ds.measure[t].reshape(r * c, p)
        referenced = shots - fit_fn(shots)
        npr[t] = shots.var(axis=0) / (referenced.var(axis=0) + 1e-30)
    keep = np.ones(d, dtype=bool)
    keep[test_delay_idx] = False
    return npr[keep].mean(axis=0)


def _weighted_db(npr_px: np.ndarray, w_k: np.ndarray) -> float:
    outside = 1.0 - w_k
    return float(10 * np.log10(np.sum(outside * npr_px) / (outside.sum() + 1e-12)))


def npr_airpcr(ds: Dataset, pc: np.ndarray, w_k: np.ndarray,
               cfg: Config) -> Tuple[float, np.ndarray]:
    """Metric (A) for airPCR. w_k enters the regression itself, not just the
    final pixel average."""
    wd = w_k.astype(np.float64)
    a_inv = np.linalg.inv((pc * wd[:, None]).T @ pc)

    def fit(shots):
        z = a_inv @ ((pc * wd[:, None]).T @ shots.T.astype(np.float64))
        return (pc @ z).T

    npr_px = _per_pixel_npr(ds, fit, cfg.test_delay_idx)
    return _weighted_db(npr_px, w_k), npr_px


def npr_airtd(ds: Dataset, atoms: np.ndarray, lam: np.ndarray, inv_psi: np.ndarray,
              w_k: np.ndarray, cfg: Config) -> Tuple[float, np.ndarray]:
    """Metric (A) for airTD, weighted by both w_k and the reliability prior."""
    w = w_k * inv_psi
    npr_px = _per_pixel_npr(ds, lambda shots: ridge_denoise(shots, atoms, w, lam),
                            cfg.test_delay_idx)
    return _weighted_db(npr_px, w_k), npr_px


def plain_db(npr_px: np.ndarray, mask: np.ndarray = None) -> float:
    """Metric (B): plain unweighted mean of linear per-pixel NPR, in dB."""
    vals = npr_px if mask is None else npr_px[mask]
    return float(10 * np.log10(np.mean(vals)))


def outside_edge_mask(energy_ev: np.ndarray, cfg: Config) -> np.ndarray:
    """Complement of the fixed edge-pixel referencing region."""
    return (energy_ev >= cfg.edge_ev_lo) & (energy_ev <= cfg.edge_ev_hi)
