"""Bounded microcanonical synthesis using the authors' Scattering Spectra core.

Only trajectory values are optimized; Battle-Lemarie filters have fixed weights.
The original numerical core is vendored without the eager-loading S&P datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import sys
import time

import numpy as np
from numpy.typing import ArrayLike

SOURCE_COMMIT = "48920143c4b702c220b22dd2486b9eb0e852592e"
VENDOR = Path(__file__).resolve().parent / "vendor"


def load_backend():
    """Import the unchanged core without executing the upstream dataset frontend."""
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    # Containers have a read-only root; all optional caches belong in /tmp.
    import tempfile
    cache = Path(os.environ.get("PYTORCH_KERNEL_CACHE_PATH", str(Path(tempfile.gettempdir()) / "track2-torch-kernels")))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", str(cache))
    loaded = sys.modules.get("scatspectra")
    if loaded is not None and Path(loaded.__file__).resolve().parent != VENDOR / "scatspectra":
        raise ValueError("a different scatspectra package was already imported; start a clean process")
    import torch
    from scatspectra.models import Model
    return torch, Model


@dataclass
class ScatteringSpectraGenerator:
    """Phi1-Phi4 constraints, Gaussian starts, L-BFGS-B, then contiguous snippets.

    The default increment domain follows the authors' snp_generation.py script
    (gen_log_returns=True). domain='levels' applies the same core to integrated
    increments, the x/log-price convention used in the paper's equations.
    In strict mode each path must meet the tolerance independently. Non-strict
    mode retains finite approximations and discloses errors and convergence.
    Generation is bounded; no alternative generator is substituted.
    A fixed synthesis seed keeps the bank unchanged across candidate-draw seeds.
    """

    bank_size: int = 64
    bank_length: int = 1024
    max_path_length: int = 512
    batch_size: int = 64
    max_iterations: int = 600
    tolerance: float = .006
    seed: int = 12345
    device: str = "cpu"
    domain: str = "increments"
    strict_tolerance: bool = True
    _history: np.ndarray | None = field(default=None, init=False, repr=False)
    _bank: np.ndarray | None = field(default=None, init=False, repr=False)
    _details: dict = field(default_factory=dict, init=False, repr=False)
    _mean: float = field(default=0, init=False)
    _scale: float = field(default=0, init=False)

    def __post_init__(self):
        if min(self.bank_size, self.batch_size, self.max_iterations, self.max_path_length) < 1:
            raise ValueError("scattering bank/batch/path sizes and iteration limit must be positive")
        if self.bank_length < self.max_path_length + 64:
            raise ValueError("scattering bank_length must exceed max_path_length by at least 64")
        if not np.isfinite(self.tolerance) or not 0 < self.tolerance < 1:
            raise ValueError("scattering tolerance must be finite and in (0,1)")
        if self.device not in {"auto", "cpu", "cuda"} or self.domain not in {"increments", "levels"}:
            raise ValueError("scattering device is auto/cpu/cuda; domain is increments/levels")

    def fit(self, increments: ArrayLike):
        values = np.asarray(increments, dtype=float)
        if values.ndim != 1 or len(values) < 64 or not np.isfinite(values).all():
            raise ValueError("Scattering Spectra needs at least 64 finite univariate increments")
        if self._history is not None and np.array_equal(values, self._history):
            return self
        self._history = values.copy()
        self._mean, self._scale = float(values.mean()), float(values.std(ddof=1))
        self._bank = None
        self._details = {}
        return self

    def _synthesize(self):
        from scipy.optimize import minimize
        from threadpoolctl import threadpool_limits
        torch, Model = load_backend()
        use_cuda = torch.cuda.is_available() if self.device == "auto" else self.device == "cuda"
        if use_cuda and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; use --ss-device cpu")
        device = torch.device("cuda" if use_cuda else "cpu")
        # All normalization and statistics depend only on the current fit prefix.
        standardized = (self._history - self._mean) / self._scale
        signal = (standardized if self.domain == "increments"
                  else np.r_[0.0, standardized.cumsum()])
        target = torch.as_tensor(signal[None, None], dtype=torch.float32, device=device)
        j = min(int(np.log2(len(signal))) - 3, int(np.log2(self.bank_length)) - 3)
        margin = 2 ** j
        if self.bank_length - 2 * margin < self.max_path_length:
            raise ValueError("scattering bank too short after excluding wavelet edge regions")

        def make_model(length, model_type, sigma2=None, r=2):
            # nchunks is a chunk size in this pinned core, despite its name.
            model = Model(
                model_type=model_type, coeff_types=None, T=length, r=r, J=j, Q=1,
                wav_type="battle_lemarie", wav_norm="l1", high_freq=.425,
                A=None, rpad=True, N=1, multivariate_model=False,
                estim_operator=None, qs=None, dtype=torch.float32,
                skew_redundance=True, histogram_moments=False,
                sigma2=sigma2, norm_on_the_fly=False, histogram_norm=None,
                nchunks=self.batch_size, gen_log_returns=self.domain == "increments",
            )
            # Module.float() leaves complex128 buffers unchanged in this core.
            model._apply(lambda t: t.to(torch.complex64) if t.is_complex() else t)
            return model.to(device)

        with torch.no_grad():
            wavelet = make_model(len(signal), None, r=1)
            wx = wavelet.compute_scattering_coefficients(target[:, :, None, None], None)[0]
            sigma2 = wx.abs().square().mean(-1)[:, :, :, 0]
            if torch.any(sigma2 <= 1e-12) or not torch.isfinite(sigma2).all():
                raise ValueError("degenerate scattering spectrum; no surrogate model was used")
            target_model = make_model(len(signal), "scat_spectra", sigma2)
            target_y = target_model(target).y.detach()
            target_energy = target_y.abs().square().mean()
        model_length = self.bank_length + (self.domain == "levels")
        model = make_model(model_length, "scat_spectra", sigma2)
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("Scattering core unexpectedly contains trainable model weights")
        descriptions = target_model.df
        coefficient_counts = descriptions["coeff_type"].value_counts().to_dict()
        rng = np.random.default_rng(self.seed)
        banks, optimizations = [], []
        started = time.perf_counter()
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with threadpool_limits(limits=1):
                for offset in range(0, self.bank_size, self.batch_size):
                    size = min(self.batch_size, self.bank_size - offset)
                    initial = rng.normal(size=(size, 1, self.bank_length))
                    if self.domain == "levels":
                        initial = np.concatenate([np.zeros((size, 1, 1)), initial.cumsum(-1)], -1)
                        initial -= initial.mean(-1, keepdims=True)
                        initial /= initial.std(-1, keepdims=True)
                        initial = signal.mean() + signal.std() * initial
                    shape = initial.shape
                    latest = {}
                    counts = {"calls": 0, "iterations": 0}

                    def objective(flat):
                        counts["calls"] += 1
                        x = torch.as_tensor(flat.reshape(shape), dtype=torch.float32,
                                            device=device).requires_grad_(True)
                        coefficients = model(x).y
                        gaps = coefficients - target_y
                        per_path = gaps.abs().square().mean((1, 2)) / target_energy
                        loss = per_path.mean()
                        gradient, = torch.autograd.grad(loss, x)
                        errors = per_path.detach().sqrt().cpu().numpy()
                        latest.update(errors=errors, loss=float(loss.detach().cpu()))
                        return latest["loss"], gradient.detach().cpu().numpy().astype(float).ravel()

                    class Matched(Exception):
                        pass

                    def callback(flat):
                        counts["iterations"] += 1
                        if np.max(latest["errors"]) <= self.tolerance:
                            raise Matched(flat.copy())

                    initial_loss, _ = objective(initial.ravel())
                    calls, iterations, status = 0, 0, "tolerance"
                    try:
                        result = minimize(objective, initial.ravel(), method="L-BFGS-B", jac=True,
                                          callback=callback, options={"maxiter": self.max_iterations,
                                          "ftol": 1e-15, "gtol": 1e-12, "maxls": 30})
                        final, calls, iterations = result.x, int(result.nfev), int(result.nit)
                        status = str(result.message)
                    except Matched as matched:
                        final = matched.args[0]
                        calls, iterations = counts["calls"], counts["iterations"]
                    objective(final)
                    errors = latest["errors"].copy()
                    if not np.isfinite(final).all() or not np.isfinite(errors).all():
                        raise ValueError('Scattering synthesis produced non-finite paths or matching errors')
                    matched_tolerance = bool(np.max(errors) <= self.tolerance * (1 + 1e-5))
                    if not matched_tolerance and self.strict_tolerance:
                        raise ValueError(f"Scattering synthesis did not match tolerance {self.tolerance}: "
                                         f"worst error={np.max(errors):.5g}; {status}")
                    values = final.reshape(shape)[:, 0]
                    if self.domain == "levels":
                        values = np.diff(values, axis=-1)
                    banks.append(values * self._scale + self._mean)
                    optimizations.append({"batch": offset // self.batch_size,
                        "initial_relative_error": float(np.sqrt(initial_loss)),
                        "final_relative_errors": errors.tolist(), "status": status,
                        "iterations": iterations, "function_calls": calls,
                        "matched_tolerance": matched_tolerance})
        finally:
            torch.set_num_threads(previous_threads)
        self._bank = np.concatenate(banks)
        self._details = {"source_commit": SOURCE_COMMIT, "backend": "authors' fixed numerical core",
            "torch_version": torch.__version__, "device": str(device), "domain": self.domain,
            "scales_J": j, "wavelet": "battle_lemarie", "layers": 2,
            "coefficient_counts": {str(k): int(v) for k, v in coefficient_counts.items()},
            "histogram_moments": False, "bank_size": self.bank_size, "bank_length": self.bank_length,
            "max_path_length": self.max_path_length, "edge_margin": margin,
            "synthesis_seed": self.seed, "tolerance": self.tolerance,
            "strict_tolerance": self.strict_tolerance,
            "matched_tolerance": all(o['matched_tolerance'] for o in optimizations),
            "synthesis_seconds": time.perf_counter() - started, "optimizations": optimizations}

    def sample(self, n_paths, n_steps, rng):
        if self._history is None or n_paths < 1 or not 1 <= n_steps <= self.max_path_length:
            raise ValueError("fit Scattering Spectra first; use positive paths and steps <= ss-max-path-length")
        if self._scale <= max(1.0, abs(self._mean)) * 1e-12:
            self._details = {"source_commit": SOURCE_COMMIT, "constant_history": True}
            return np.full((n_paths, n_steps), self._mean)
        if self._bank is None:
            self._synthesize()
        margin = self._details["edge_margin"]
        starts_per_path = self.bank_length - 2 * margin - self.max_path_length + 1
        population = self.bank_size * starts_per_path
        ids = rng.choice(population, size=n_paths, replace=n_paths > population)
        # Use a fixed maximum snippet span so requested horizons preserve prefixes.
        bank_ids, starts = ids // starts_per_path, margin + ids % starts_per_path
        self._details.update(candidate_window_population=population,
                             unique_candidate_windows=int(np.unique(ids).size))
        return self._bank[bank_ids[:, None], starts[:, None] + np.arange(n_steps)]

    def diagnostics(self):
        return self._details.copy()
