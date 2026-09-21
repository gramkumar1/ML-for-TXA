"""Deployment: the airPCR iterative discriminator, driving either basis.

Both methods run on the *full per-shot dataset* (every D*R*C row), not the
delay-averaged map: the discriminator converges against a different, much
smoother quantity if fed the mean, and lands on a different weight map.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .dictionary import ridge_denoise


@dataclass
class Iterate:
    """State saved at one discriminator iteration."""
    residual: Optional[np.ndarray]   # (n_rows, P), None when save_full=False
    fit: Optional[np.ndarray]
    w_k: np.ndarray                  # the weight that produced this residual


def _pcr_fit(x: np.ndarray, pc: np.ndarray, w: np.ndarray) -> np.ndarray:
    a = (pc * w[:, None]).T @ pc
    b = (w[None, :] * x) @ pc
    return (np.linalg.solve(a, b.T).T @ pc.T)


def discriminator_sweep(x: np.ndarray, fit_fn: Callable[[np.ndarray], np.ndarray],
                        k_max: int, k_opts: Sequence[int], c: float = 0.2,
                        save_full: bool = True) -> Dict[int, Iterate]:
    """Run w_{k+1} = exp(-alpha_k d_k / ||d_k||), alpha_k = (1+c)^k.

    `fit_fn(w)` returns the fit of every row of `x` under pixel weight `w`.
    The saved `w_k` is the one *entering* the iteration, i.e. the weight that
    produced that iteration's residual; saving the updated weight instead pairs
    each map with the next iteration's weight.
    """
    k_set = set(k_opts)
    w_k = np.ones(x.shape[1], dtype=np.float32)
    saved: Dict[int, Iterate] = {}
    for k in range(k_max):
        fit = fit_fn(w_k)
        residual = x - fit
        if k in k_set:
            saved[k] = Iterate(
                residual=residual.astype(np.float32) if save_full else None,
                fit=fit.astype(np.float32) if save_full else None,
                w_k=w_k.copy())
        d_k = np.abs(residual.mean(axis=0))
        w_k = np.exp(-((1.0 + c) ** k) * d_k / (np.linalg.norm(d_k) + 1e-12)).astype(np.float32)
    return saved


def airpcr_sweep(shots: np.ndarray, pc: np.ndarray, k_max: int,
                 k_opts: Sequence[int], c: float = 0.2,
                 save_full: bool = True) -> Dict[int, Iterate]:
    return discriminator_sweep(shots, lambda w: _pcr_fit(shots, pc, w.astype(np.float64)),
                               k_max, k_opts, c, save_full)


def airtd_sweep(shots: np.ndarray, atoms: np.ndarray, lam: np.ndarray,
                inv_psi: np.ndarray, k_max: int, k_opts: Sequence[int],
                c: float = 0.2, save_full: bool = True) -> Dict[int, Iterate]:
    return discriminator_sweep(shots, lambda w: ridge_denoise(shots, atoms, w * inv_psi, lam),
                               k_max, k_opts, c, save_full)


def shots_to_delay_map(iterate: Iterate, shape: Tuple[int, int, int]) -> np.ndarray:
    """Average a per-shot residual over shots, per delay -> (D, P) OD map."""
    return iterate.residual.reshape(shape).mean(axis=1).astype(np.float32)
