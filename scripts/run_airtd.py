#!/usr/bin/env python3
"""Run the full airTD pipeline and write the final OD map."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airtd import Config, load_dataset, run
from airtd.config import DEFAULT_DATA_DIR
from airtd.plotting import plot_od_maps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--download", action="store_true",
                    help="fetch missing HDF5 files from Google Drive")
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--n-atoms", type=int, default=Config.n_atoms)
    ap.add_argument("--k-fa", type=int, default=Config.k_fa)
    ap.add_argument("--auto-k", action="store_true",
                    help="select k_opt by argmax instead of the pinned values")
    ap.add_argument("--out", type=Path, default=Path("outputs/od_maps.png"))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cfg = Config(epochs=args.epochs, n_atoms=args.n_atoms, k_fa=args.k_fa)
    if args.auto_k:
        cfg.k_opt_airpcr = cfg.k_opt_airtd = None

    ds = load_dataset(args.data_dir, download=args.download)
    print(f"measure {ds.shape}  delays {ds.delays_ps[0]:.1f}..{ds.delays_ps[-1]:.1f} ps")

    result = run(ds, cfg)
    print("\n" + result.report())

    plot_od_maps(ds.energy_ev, ds.delays_ps, result.od_maps, cfg,
                 save_to=args.out, show=not args.no_show)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
