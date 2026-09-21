"""Loading of the XUV transient-absorption dataset."""
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .config import DEFAULT_DATA_DIR, GDRIVE_IDS


@dataclass
class Dataset:
    energy_ev: np.ndarray      # (P,)
    delays_ps: np.ndarray      # (D,)
    measure: np.ndarray        # (D, R, C, P) per-shot OD
    spectra_off: np.ndarray    # (D, R, C, P) raw UV-off spectra

    @property
    def shape(self):
        return self.measure.shape

    @property
    def mean_map_od(self) -> np.ndarray:
        """Grand-mean OD map, (D, P)."""
        return self.measure.mean(axis=(1, 2))

    @property
    def shots(self) -> np.ndarray:
        """All individual shots flattened to (D*R*C, P)."""
        d, r, c, p = self.measure.shape
        return self.measure.reshape(d * r * c, p).astype(np.float32)


def _resolve(data_dir: Path, name: str, download: bool) -> Path:
    path = data_dir / name
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"{path} not found (pass download=True to fetch it)")
    import gdown
    data_dir.mkdir(parents=True, exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_IDS[name]}", str(path), quiet=False)
    return path


def load_dataset(data_dir: Path = DEFAULT_DATA_DIR, download: bool = False) -> Dataset:
    data_dir = Path(data_dir)

    with h5py.File(_resolve(data_dir, "calibration.h5", download), "r") as f:
        energy_ev = f["energy_vs_pix"][()].astype(np.float32)

    with h5py.File(_resolve(data_dir, "log_transients.h5", download), "r") as f:
        delays_ps = -f["delays"][()].astype(np.float32)
        measure = f["measure"][()].astype(np.float32)

    with h5py.File(_resolve(data_dir, "spectra_on_spectra_off.h5", download), "r") as f:
        spectra_off = f["spectra_off"][()].astype(np.float32)

    return Dataset(energy_ev=energy_ev, delays_ps=delays_ps,
                   measure=measure, spectra_off=spectra_off)
