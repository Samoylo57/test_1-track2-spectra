# Scattering Spectra numerical core

The `scatspectra/models`, `layers`, `description` and `utils` directories are
unchanged files from RudyMorel/scattering_spectra version 2.0.3, commit
`48920143c4b702c220b22dd2486b9eb0e852592e`, under the included MIT license.
SHA-256 hashes are recorded in `scatspectra-manifest.json`.

The root initializer is replaced with an empty initializer. The upstream
initializer imports `data/__init__.py`, which eagerly reads bundled S&P/VIX
pickle datasets. Those datasets, the frontend, tutorials and standard model
datasets are excluded here. Forecasts use only the requested unit's fit prefix.

`scattering_generator.py` imports this core explicitly and supplies a bounded
L-BFGS-B driver. It optimizes trajectory values, retaining the authors' wavelet
filters, description, modulus, correlations and normalization. Batched synthesis
minimizes each path's coefficient mismatch, rather than matching batch averages.
Histogram extensions are disabled, as in the authors' `snp_generation.py` default.
