"""Calibration noise pool, PCA basis and per-pixel reliability prior.

No whitening is applied anywhere: the reference implementation fits PCA on raw,
mean-centred OD-unit data, and whitening changes which pixels the discriminator
downweights first rather than merely rescaling.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import train_test_split

from .config import Config
from .data import Dataset


@dataclass
class Calibration:
    noise_pool: np.ndarray     # (n_samples, P) mean-centred calibration noise
    train_pool: np.ndarray
    val_pool: np.ndarray
    pca: PCA
    pc: np.ndarray             # (P, k_latent) principal components
    psi: np.ndarray            # (P,) factor-analysis noise variance
    inv_psi: np.ndarray        # (P,) per-pixel reliability weight, capped at 1.0


def build_noise_pool(ds: Dataset) -> np.ndarray:
    """Calibration noise from *disjoint* pairs of consecutive UV-off spectra.

    Overlapping consecutive differences would share a shot between neighbouring
    rows, correlating the samples and inflating the apparent pool size.
    """
    log_off = np.log10(np.maximum(ds.spectra_off, 1e-10))
    _, r, _, p = ds.shape
    n_pairs = r // 2
    a = log_off[:, 0::2][:, :n_pairs]
    b = log_off[:, 1::2][:, :n_pairs]
    pool = (-(b - a)).reshape(-1, p).astype(np.float32)
    return pool - pool.mean(axis=0)


def reliability_weight(psi: np.ndarray, n_pixels: int) -> np.ndarray:
    """1/Psi, normalised to unit mean and capped at 1.0.

    The cap is one-sided on purpose: it suppresses idiosyncratic / flux-minimum
    pixels without ever amplifying trust above baseline in well-correlated
    real-signal regions, where amplification attenuates genuine signal.
    """
    inv = (1.0 / psi).astype(np.float32)
    inv *= n_pixels / inv.sum()
    return np.minimum(inv, 1.0).astype(np.float32)


def fit_calibration(ds: Dataset, cfg: Config) -> Calibration:
    pool = build_noise_pool(ds)
    p = pool.shape[1]
    train_pool, val_pool = train_test_split(
        pool, test_size=cfg.val_fraction, random_state=cfg.seed)

    pca = PCA(n_components=cfg.k_latent).fit(train_pool)
    fa = FactorAnalysis(n_components=cfg.k_fa, random_state=cfg.seed).fit(train_pool)
    psi = np.clip(fa.noise_variance_, 1e-4, None).astype(np.float32)

    return Calibration(
        noise_pool=pool,
        train_pool=train_pool,
        val_pool=val_pool,
        pca=pca,
        pc=pca.components_.T.astype(np.float32),
        psi=psi,
        inv_psi=reliability_weight(psi, p),
    )
