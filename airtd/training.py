"""Training the dictionary by backpropagating through the deployment operator."""
from dataclasses import replace
from typing import Tuple

import numpy as np
import torch

from .calibration import Calibration
from .config import Config, DEVICE
from .dictionary import TaskDict, batched_ridge, sample_masks


def train_dictionary(calib: Calibration, cfg: Config,
                     verbose: bool = True) -> Tuple[TaskDict, dict]:
    """Train a :class:`TaskDict` on the calibration noise pool.

    Each step draws a fresh random mask and backprops through the same weighted
    ridge solve used at deployment. The loss is a plain MSE over all pixels:
    upweighting the masked region was tested and found mildly harmful, since
    mask diversity plus limited atom capacity already force the atoms to capture
    the correlation structure needed to predict any subset of pixels.
    """
    n_pixels = calib.train_pool.shape[1]
    inv_psi_t = torch.tensor(calib.inv_psi, device=DEVICE)
    sqrt_rel_t = torch.tensor(np.sqrt(calib.inv_psi), device=DEVICE)

    model = TaskDict(n_pixels, cfg.n_atoms, sqrt_rel_t, cfg.lam_init).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-5)

    train_t = torch.tensor(calib.train_pool, device=DEVICE)
    val_t = torch.tensor(calib.val_pool, device=DEVICE)
    rng = np.random.default_rng(cfg.seed)

    # A single fixed validation mask keeps model selection stable across epochs.
    val_cfg = replace(cfg, p_clear=0.0)
    val_obs = sample_masks(len(calib.val_pool), n_pixels,
                           np.random.default_rng(cfg.seed + 1), val_cfg)
    val_w = torch.tensor(val_obs, device=DEVICE) * inv_psi_t.unsqueeze(0)
    val_masked = val_obs < 0.5

    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_t))
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(train_t), cfg.batch_size):
            x = train_t[perm[i:i + cfg.batch_size]]
            obs = torch.tensor(sample_masks(len(x), n_pixels, rng, cfg), device=DEVICE)
            loss = ((batched_ridge(model, x, obs * inv_psi_t.unsqueeze(0)) - x) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred = batched_ridge(model, val_t, val_w).cpu().numpy()
        val_mse = float(np.mean((pred[val_masked] - calib.val_pool[val_masked]) ** 2))
        history["train"].append(epoch_loss / n_batches)
        history["val"].append(val_mse)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and epoch % 25 == 0:
            print(f"epoch {epoch:3d}  train={history['train'][-1]:.6f}  "
                  f"val masked-MSE={val_mse:.6f}  best={best_val:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f"restored best checkpoint (val masked-MSE={best_val:.6f})")
    return model, history
