"""The task-trained dictionary and the differentiable weighted ridge solve."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from .config import Config


class TaskDict(nn.Module):
    """Unit-norm dictionary D (n_atoms x P) plus a learned per-atom Tikhonov
    shrinkage lambda = softplus(rho).

    Atoms are scaled structurally by sqrt(1/Psi) before normalisation; applying
    the reliability prior through the loss alone is not sufficient.
    """

    def __init__(self, n_pixels: int, n_atoms: int, sqrt_rel: torch.Tensor,
                 lam_init: float = 0.05):
        super().__init__()
        self.D_raw = nn.Parameter(torch.randn(n_atoms, n_pixels) * 0.02)
        self.rho = nn.Parameter(torch.full((n_atoms,), float(np.log(np.expm1(lam_init)))))
        self.register_buffer("sqrt_rel", sqrt_rel)

    def atoms(self) -> torch.Tensor:
        scaled = self.D_raw * self.sqrt_rel.unsqueeze(0)
        return scaled / (scaled.norm(dim=1, keepdim=True) + 1e-8)

    def lam(self) -> torch.Tensor:
        return F.softplus(self.rho)

    def numpy_params(self):
        with torch.no_grad():
            return (self.atoms().cpu().numpy().astype(np.float32),
                    self.lam().cpu().numpy().astype(np.float32))


def batched_ridge(model: TaskDict, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Solve z = argmin_z sum_p w_p (x_p - (D^T z)_p)^2 + sum_n lam_n z_n^2 and
    return D^T z. Differentiable, and the same operator used at deployment."""
    d_n = model.atoms()
    lam = model.lam()
    d_w = d_n.unsqueeze(0) * w.unsqueeze(1)
    a = d_w @ d_n.T
    tr = a.diagonal(dim1=1, dim2=2).sum(-1) / d_n.shape[0]
    a = a + torch.diag_embed(lam.unsqueeze(0) * tr.view(-1, 1))
    b = torch.einsum("np,bp->bn", d_n, w * x)
    z = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
    return z @ d_n


def ridge_denoise(x: np.ndarray, atoms: np.ndarray, w: np.ndarray,
                  lam: np.ndarray) -> np.ndarray:
    """NumPy deployment path, identical math to :func:`batched_ridge`."""
    d_w = atoms * w[None, :]
    a = d_w @ atoms.T
    tr = np.trace(a) / len(a)
    a[np.diag_indices_from(a)] += lam * tr
    z = np.linalg.solve(a, d_w @ x.T)
    return (atoms.T @ z).T.astype(np.float32)


def sample_masks(n: int, n_pixels: int, rng: np.random.Generator,
                 cfg: Config) -> np.ndarray:
    """Soft-edged observation masks: bands of untrusted pixels, occasionally none."""
    obs = np.ones((n, n_pixels), dtype=np.float32)
    clear = rng.random(n) < cfg.p_clear
    for i in range(n):
        if clear[i]:
            continue
        for _ in range(int(rng.integers(cfg.mask_bands[0], cfg.mask_bands[1] + 1))):
            width = int(rng.integers(cfg.mask_width[0], cfg.mask_width[1]))
            centre = int(rng.integers(0, n_pixels))
            obs[i, max(0, centre - width // 2):min(n_pixels, centre + width // 2)] = 0.0
    if cfg.mask_soft_sigma > 0:
        obs = gaussian_filter1d(obs, cfg.mask_soft_sigma, axis=1)
    return obs.astype(np.float32)
