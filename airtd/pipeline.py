"""End-to-end airTD run: calibration -> training -> deployment -> NPR."""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .calibration import Calibration, fit_calibration
from .config import Config
from .data import Dataset
from .dictionary import TaskDict
from .inference import airpcr_sweep, airtd_sweep, shots_to_delay_map
from .metrics import npr_airpcr, npr_airtd, outside_edge_mask, plain_db
from .training import train_dictionary


@dataclass
class Result:
    calib: Calibration
    model: TaskDict
    history: dict
    k_opt: Dict[str, int]
    od_maps: Dict[str, np.ndarray]     # method -> (D, P) referenced OD map
    npr_px: Dict[str, np.ndarray]      # method -> (P,) linear per-pixel NPR
    npr_db: Dict[str, Dict[str, float]]

    def report(self) -> str:
        lines = []
        for method, scores in self.npr_db.items():
            lines.append(
                f"{method:7s} k={self.k_opt[method]:3d}  "
                f"(A) weighted {scores['weighted']:.1f} dB   "
                f"(B) all pixels {scores['all_px']:.1f} dB   "
                f"outside ref. region {scores['outside']:.1f} dB")
        lines.append("paper  (airPCR, k=46): (A) 15.9 dB   (B) 16.2 / 16.5 dB")
        return "\n".join(lines)


def _select_k(sweep, score_fn, k_sweep, override: Optional[int]) -> int:
    if override is not None:
        return override
    scores = [score_fn(sweep[k].w_k) for k in k_sweep]
    return k_sweep[int(np.argmax(scores))]


def run(ds: Dataset, cfg: Config = Config(), verbose: bool = True) -> Result:
    d, r, c, p = ds.shape
    shots = ds.shots
    k_max = max(cfg.k_sweep) + 1

    calib = fit_calibration(ds, cfg)
    if verbose:
        print(f"calibration pool: {calib.noise_pool.shape}  PCA-{cfg.k_latent}, FA-{cfg.k_fa}")

    model, history = train_dictionary(calib, cfg, verbose=verbose)
    atoms, lam = model.numpy_params()

    def score_airpcr(w_k):
        return npr_airpcr(ds, calib.pc, w_k, cfg)[0]

    def score_airtd(w_k):
        return npr_airtd(ds, atoms, lam, calib.inv_psi, w_k, cfg)[0]

    # k_opt is chosen with the same metric that gets reported, as in the paper.
    # A coarse sweep keeps only w_k; full residuals are refit once at k_opt.
    if cfg.k_opt_airpcr is None or cfg.k_opt_airtd is None:
        if verbose:
            print("sweeping k over the full per-shot dataset ...")
    k_opt = {}
    finals = {}
    for name, sweep_fn, score_fn, override in (
        ("airPCR", lambda km, ko, full: airpcr_sweep(shots, calib.pc, km, ko,
                                                     cfg.airpcr_c, full), score_airpcr,
         cfg.k_opt_airpcr),
        ("airTD", lambda km, ko, full: airtd_sweep(shots, atoms, lam, calib.inv_psi,
                                                   km, ko, cfg.airpcr_c, full), score_airtd,
         cfg.k_opt_airtd),
    ):
        if override is None:
            coarse = sweep_fn(k_max, cfg.k_sweep, False)
            k = _select_k(coarse, score_fn, cfg.k_sweep, None)
        else:
            k = override
        k_opt[name] = k
        finals[name] = sweep_fn(k + 1, [k], True)[k]
        if verbose:
            print(f"{name}: k_opt={k}")

    od_maps = {"Raw": ds.mean_map_od}
    npr_px, npr_db = {}, {}
    outside = outside_edge_mask(ds.energy_ev, cfg)
    for name, iterate in finals.items():
        od_maps[f"{name} (k={k_opt[name]})"] = shots_to_delay_map(iterate, (d, r * c, p))
        if name == "airPCR":
            weighted, px = npr_airpcr(ds, calib.pc, iterate.w_k, cfg)
        else:
            weighted, px = npr_airtd(ds, atoms, lam, calib.inv_psi, iterate.w_k, cfg)
        npr_px[name] = px
        npr_db[name] = {"weighted": weighted,
                        "all_px": plain_db(px),
                        "outside": plain_db(px, outside)}

    return Result(calib=calib, model=model, history=history, k_opt=k_opt,
                  od_maps=od_maps, npr_px=npr_px, npr_db=npr_db)
