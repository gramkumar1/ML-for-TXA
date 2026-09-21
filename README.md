# airTD — task-trained dictionary denoising for XUV transient absorption

airTD denoises XUV transient-absorption spectra by replacing the PCA basis of
the reference **airPCR** algorithm (Faccialà, Toulson & Gessner, *Opt. Express*
29, 2021) with a dictionary trained end-to-end **through the deployment
operator** — a differentiable weighted ridge solve — rather than by plain
reconstruction. Everything else about the pipeline, including the iterative
discriminator and the NPR metrics, matches the paper so the two methods can be
compared numerically on the same dataset.

## Method

1. **Calibration pool** (`airtd/calibration.py`) — noise samples from *disjoint*
   pairs of consecutive UV-off spectra. A PCA basis (q = 35, the paper's value)
   and a factor analysis (100 components) are fit on it; the factor-analysis
   noise variance `Psi` becomes a per-pixel reliability weight `1/Psi`, capped
   at 1.0 so it can suppress unreliable pixels but never amplify trust above
   baseline. No whitening at any stage.

2. **Dictionary** (`airtd/dictionary.py`) — unit-norm atoms `D` plus a learned
   per-atom Tikhonov shrinkage. Denoising a spectrum `x` solves

   ```
   z = argmin_z  Σ_p w_p (x_p − (Dᵀz)_p)²  +  Σ_n λ_n z_n²
   ```

   in closed form, so the whole operator is differentiable.

3. **Training** (`airtd/training.py`) — each step samples a soft-edged random
   mask (`w = 0` over bands of pixels that might contain signal) and backprops
   plain MSE through that same solve.

4. **Deployment** (`airtd/inference.py`) — the paper's iterative discriminator,
   `w_{k+1} = exp(−α_k d_k / ‖d_k‖)` with `α_k = (1+c)^k`, run on the full
   per-shot dataset. The same routine drives the PCA basis, giving a like-for-like
   airPCR baseline.

5. **Scoring** (`airtd/metrics.py`) — both of the paper's NPR definitions: the
   Eq. 6 `(1−w_k)`-weighted convergence diagnostic and the plain unweighted
   headline mean over the fixed edge-pixel referencing region.

Several variants were tried and reverted along the way: a full learned prior
precision matrix in place of the scalar per-atom shrinkage, an unrolled
discriminator trained against synthetic injected signal, blind-spot
leave-band-out inference, and an explicitly upweighted masked-region loss term.
None improved NPR; the last two made it worse.

## Results

| Metric | airPCR | airTD | Paper (airPCR, k=46) |
|---|---|---|---|
| (A) Eq. 6, weighted | 15.9 dB | **17.2 dB** | 15.9 dB |
| (B) all pixels | 16.1 dB | **17.9 dB** | 16.2 dB |
| (B) outside ref. region | 16.6 dB | **17.6 dB** | 16.5 dB |

airPCR reproduces the paper's own numbers, confirming the pipeline; airTD reads
1–1.8 dB higher across every metric.

## Usage

```bash
pip install -r requirements.txt
python scripts/run_airtd.py                 # data in transient_data/
python scripts/run_airtd.py --download      # fetch the HDF5 files first
python scripts/run_airtd.py --epochs 30 --no-show
```

Or from Python:

```python
from airtd import Config, load_dataset, run
from airtd.plotting import plot_od_maps

ds = load_dataset()
result = run(ds, Config())
print(result.report())
plot_od_maps(ds.energy_ev, ds.delays_ps, result.od_maps, Config())
```

## Data

Three HDF5 files in `transient_data/` (gitignored, ~3.7 GB; `--download` fetches
them from Google Drive):

| File | Contents |
|---|---|
| `log_transients.h5` | `measure`, the `(delay, repeat, cycle, pixel)` OD tensor |
| `spectra_on_spectra_off.h5` | raw per-shot on/off spectra, for the calibration pool |
| `calibration.h5` | energy-per-pixel calibration curve |

## Layout

```
airtd/
  config.py       hyperparameters, device, data paths
  data.py         HDF5 loading
  calibration.py  noise pool, PCA basis, reliability prior
  dictionary.py   TaskDict, the ridge solve, mask sampling
  training.py     backprop through the deployment operator
  inference.py    the iterative discriminator
  metrics.py      both NPR definitions
  pipeline.py     run() — calibration to scored maps
  plotting.py     final OD map figure
scripts/
  run_airtd.py    CLI entry point
```
