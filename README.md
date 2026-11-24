# spac-ann-3c-inversion

Physics-guided 3-component **SPAC / F-J**, **RayDec / DFA HVSR**, **PyTorch ADsurf** mode-number-free surface-wave dispersion determinants, **probabilistic deep learning** (`MDNInversionNet`, `JointNetDispFormer`, `ConditionalNormalizingFlow3C`), and **Parallel-Tempering trans-dimensional rj-MCMC** toolkit for broadband surface-wave characterization and bedrock depth estimation.

## Key Features

- **3-Component Array Processing (`spac_ann.spac_3c`)**
  - Near-DC / thermal drift suppression diagnostics + Sleeman 3-station self-noise & array-wide incoherent Noise-to-Signal Ratio (NSR) estimator.
  - True 3-component `E/N/Z -> R/T/Z` azimuth rotation, isotropic `Z-R / R-Z` real-part anti-coherence, and decoupled **Rayleigh (`J_0 - J_2`) vs. Love (`J_0 + J_2`) Frequency-Bessel (F-J)** spectra with adaptive quadrature weights and dynamic-programming ridge tracking.
- **HVSR & Rayleigh Ellipticity (`spac_ann.raydec_dfa`)**
  - Numba-accelerated elastic **Diffuse-Field Assumption (DFA)** Green's-function HVSR forward solver (PSV + SH contributions + Love fraction, converging cleanly to the analytical Sanchez-Sesma Poisson half-space asymptote `1.344`).
  - **Hobiger et al. (2009) RayDec** Random Decrement directional Rayleigh ellipticity estimator with Love-wave rejection.
- **Differentiable Physics & ADsurf (`spac_ann.adsurf_det`)**
  - Exact, pole-free **Plucker compound Rayleigh determinant** `D_R(f, c)` and sinc/tanh-regularized **Love determinant** `D_L(f, c)` in PyTorch `float64`.
  - **Mode-number-free Newton velocity distance** (`|D(f, c)| / |dD/dc|`) supporting fundamental and coherent higher-mode picks without mode-index assignment.
  - Two-stage joint inversion: Stage-1 multi-start Adam autodiff + Stage-2 Powell DFA/RayDec/direct-SPAC polish.
- **Probabilistic Deep Learning (`spac_ann.models_ann`)**
  - `MDNInversionNet`: heteroscedastic Mixture Density Network outputting multimodal 1D Vs posteriors.
  - `JointNetDispFormer`: dual-branch CNN + Transformer encoder predicting full Vs profiles and engineering depth/velocity metrics (`Vs30`, `Z1.0`, `Z760`, `f0_SH`).
  - `ConditionalNormalizingFlow3C`: RealNVP conditional normalizing flow (`log_prob` + exact invertible posterior sampling capturing full depth-covariance structure).
- **Trans-Dimensional Bayesian Sampling (`spac_ann.rjmcmc`)**
  - Birth/death/move/velocity trans-dimensional reversible-jump MCMC with exact detailed balance, 4-temperature **Parallel Tempering**, and multi-chain Gelman-Rubin `R-hat` diagnostics.
- **SAC I/O & Basin Physics (`spac_ann.sac_io`, `spac_ann.physics`, `spac_ann.plotting`)**
  - Binary SAC header/trace reader, auto-UTM projection (`pyproj`), Brocher (2005) `Vp/rho` relations, 1D viscoelastic SH transfer resonance, and synthetic basin generator.

## Repository Layout

```text
spac-ann-3c-inversion/
|-- spac_ann/
|   |-- __init__.py       # Public API exports
|   |-- sac_io.py         # Binary SAC reader, windowing, geodesic/UTM utilities
|   |-- physics.py        # Brocher relations, SH f0, profile metrics, synthetic basin generator
|   |-- spac_3c.py        # Drift filter, Sleeman/array NSR, 3C coherencies, decoupled 3C F-J
|   |-- raydec_dfa.py     # Numba elastic DFA HVSR forward & Hobiger RayDec ellipticity
|   |-- adsurf_det.py     # PyTorch Plucker Rayleigh & Love determinants + ADsurf joint inversion
|   |-- models_ann.py     # MDNInversionNet, JointNetDispFormer, ConditionalNormalizingFlow3C
|   |-- rjmcmc.py         # Parallel-Tempering trans-dimensional rj-MCMC & Gelman-Rubin R-hat
|   `-- plotting.py       # Classic ring SPAC fit, posterior Vs uncertainty & dispersion plots
|-- examples/
|   `-- synthetic_end_to_end_demo.py
|-- tests/
|   `-- test_suite.py
|-- requirements.txt
|-- pyproject.toml
`-- LICENSE
```

## Installation

```bash
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Run the self-contained synthetic end-to-end demonstration (synthesizes a 1D sedimentary basin profile, decouples synthetic 3C SPAC/F-J wavefields, runs RayDec + DFA HVSR, checks PyTorch ADsurf determinants, runs CNF + ADsurf inversion + Parallel-Tempering rj-MCMC, and writes summary figures into `./demo_outputs/`):

```bash
python examples/synthetic_end_to_end_demo.py
```

Run the automated analytical, physics, differentiability, and bijection verification suite:

```bash
python tests/test_suite.py
```

## License

Released under the [MIT License](LICENSE).
