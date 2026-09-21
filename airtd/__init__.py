"""airTD — task-trained dictionary denoising for XUV transient-absorption data."""
from .calibration import Calibration, fit_calibration
from .config import Config
from .data import Dataset, load_dataset
from .dictionary import TaskDict, ridge_denoise
from .inference import airpcr_sweep, airtd_sweep, shots_to_delay_map
from .metrics import npr_airpcr, npr_airtd, outside_edge_mask, plain_db
from .pipeline import Result, run
from .training import train_dictionary

__all__ = [
    "Calibration", "Config", "Dataset", "Result", "TaskDict",
    "airpcr_sweep", "airtd_sweep", "fit_calibration", "load_dataset",
    "npr_airpcr", "npr_airtd", "outside_edge_mask", "plain_db",
    "ridge_denoise", "run", "shots_to_delay_map", "train_dictionary",
]
