"""Generate increment paths independently of the conditioning method.

The baseline resamples contiguous history blocks, restarting at random times.
Past and future are generated in one call so their boundary retains dependence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
GENERATOR_NAMES = (
    "stationary-bootstrap", "iid-bootstrap", "student-t", "garch-bootstrap", "quantile-gbdt",
    "scattering-spectra",
)


class PathGenerator(Protocol):
    """Replacement generators only need fit() and sample()."""

    def fit(self, increments: ArrayLike) -> PathGenerator: ...

    def sample(
        self, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> FloatArray: ...


@dataclass
class StationaryBootstrapGenerator:
    """Bootstrap blocks of geometrically distributed length, without end wrapping.

    At each step, continue with the next historical increment with probability
    1 - 1/mean_block_length. Otherwise start at a uniformly selected increment.
    Reaching the history's end forces a restart instead of inventing an adjacency
    between the last and first observation. The caller supplies gap-free history.
    """

    mean_block_length: float = 20.0
    _history: FloatArray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean_block_length) or self.mean_block_length < 1:
            raise ValueError("mean_block_length must be finite and >= 1")

    def fit(self, increments: ArrayLike) -> StationaryBootstrapGenerator:
        history = np.asarray(increments, dtype=np.float64)
        if history.ndim != 1 or history.size < 2 or not np.isfinite(history).all():
            raise ValueError("fit requires at least two finite univariate increments")
        self._history = history.copy()
        return self

    def sample(
        self, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> FloatArray:
        if self._history is None:
            raise ValueError("fit the generator before sampling")
        if n_paths < 1 or n_steps < 1:
            raise ValueError("n_paths and n_steps must be positive")
        history = self._history
        indices = rng.integers(history.size, size=n_paths)
        paths = np.empty((n_paths, n_steps), dtype=np.float64)
        paths[:, 0] = history[indices]
        for t in range(1, n_steps):
            restart = (rng.random(n_paths) < 1.0 / self.mean_block_length) | (
                indices + 1 >= history.size
            )
            indices += 1
            indices[restart] = rng.integers(history.size, size=int(restart.sum()))
            paths[:, t] = history[indices]
        return paths


def _history(increments: ArrayLike, minimum: int = 30) -> FloatArray:
    values = np.asarray(increments, dtype=np.float64)
    if values.ndim != 1 or len(values) < minimum or not np.isfinite(values).all():
        raise ValueError(f"require at least {minimum} finite univariate increments")
    return values.copy()


@dataclass
class StudentTRandomWalkGenerator:
    """IID joint Student-t increments with sample mean and sample covariance.

    A common chi-square divisor per date couples all assets' tail shocks. The
    shape matrix is covariance * (df-2)/df, so df does not inflate fitted variance.
    A vector input returns [path, time]; a matrix input returns [path, time, asset].
    Time steps are independent. PSMC cannot add population-level temporal memory.
    """

    df: float = 5.0
    _mean: FloatArray | None = field(default=None, init=False, repr=False)
    _root: FloatArray | None = field(default=None, init=False, repr=False)
    _covariance: FloatArray | None = field(default=None, init=False, repr=False)
    _univariate: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.df) or self.df <= 2:
            raise ValueError("Student-t df must be finite and > 2 for a finite covariance")

    def fit(self, increments: ArrayLike) -> StudentTRandomWalkGenerator:
        values = np.asarray(increments, dtype=float)
        if values.ndim not in (1, 2) or len(values) < 30 or not np.isfinite(values).all():
            raise ValueError("Student-t fit needs >= 30 finite rows, vector or [time, asset]")
        self._univariate = values.ndim == 1
        values = values[:, None] if self._univariate else values
        if values.shape[1] == 0:
            raise ValueError("provide at least one asset")
        self._mean = values.mean(axis=0)
        self._covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
        eigenvalues, vectors = np.linalg.eigh(self._covariance)
        self._root = vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0) * (self.df - 2) / self.df))
        return self

    def sample(self, n_paths: int, n_steps: int, rng: np.random.Generator) -> FloatArray:
        if self._mean is None or self._root is None or n_paths < 1 or n_steps < 1:
            raise ValueError("fit Student-t first and use positive path/step counts")
        paths = np.empty((n_paths, n_steps, len(self._mean)))
        # Draw one date at a time so adding later horizons preserves earlier paths.
        for t in range(n_steps):
            gaussian = rng.standard_normal((n_paths, len(self._mean))) @ self._root.T
            divisor = np.sqrt(rng.chisquare(self.df, size=n_paths) / self.df)
            paths[:, t] = self._mean + gaussian / divisor[:, None]
        return paths[:, :, 0] if self._univariate else paths

    def diagnostics(self) -> dict:
        return {"df": self.df, "mean": self._mean.tolist(),
                "covariance": self._covariance.tolist(), "independent_time_steps": True}


@dataclass
class GARCHBootstrapGenerator:
    """Constant mean + GARCH(1,1), Gaussian QMLE, bootstrapped standardized shocks.

    Fitting rescales increments for numerical stability and constrains alpha+beta
    below .995. Candidate paths start from randomly selected fitted variance states,
    then evolve continuously through the synthetic past and future. This is an
    unconditional candidate generator, not a direct forecast from the last state.
    """

    _history: FloatArray | None = field(default=None, init=False, repr=False)
    _mean: float = field(default=0, init=False)
    _scale: float = field(default=0, init=False)
    _parameters: FloatArray | None = field(default=None, init=False, repr=False)
    _variances: FloatArray | None = field(default=None, init=False, repr=False)
    _residuals: FloatArray | None = field(default=None, init=False, repr=False)

    def fit(self, increments: ArrayLike) -> GARCHBootstrapGenerator:
        values = _history(increments)
        if self._history is not None and np.array_equal(values, self._history):
            return self
        from scipy.optimize import minimize
        from scipy.signal import lfilter

        self._mean = float(values.mean())
        self._scale = float(values.std(ddof=1))
        if self._scale <= max(1.0, abs(self._mean)) * 1e-12:
            self._scale = 0.0
            self._parameters = np.array([0.0, 0.0, 0.0])
            self._variances = np.zeros(len(values))
            self._residuals = np.zeros(len(values))
            self._history = values
            return self
        errors = (values - self._mean) / self._scale

        def variances(parameters):
            omega, alpha, beta = parameters
            following, _ = lfilter([1.0], [1.0, -beta], omega + alpha * errors[:-1] ** 2, zi=[beta])
            return np.r_[1.0, following]

        def objective(parameters):
            variance = variances(parameters)
            if np.any(variance <= 0) or not np.isfinite(variance).all():
                return 1e100
            return float(np.mean(np.log(variance) + errors**2 / variance))

        fits = [minimize(
            objective, [1 - a - b, a, b], method="SLSQP",
            bounds=[(1e-6, 10), (0, .994), (0, .994)],
            constraints=[{"type": "ineq", "fun": lambda p: .995 - p[1] - p[2]}],
            options={"maxiter": 300, "ftol": 1e-9},
        ) for a, b in ((.05, .9), (.15, .75))]
        accepted = [fit for fit in fits if fit.success and np.isfinite(fit.fun)
                    and fit.x[1] + fit.x[2] <= .995 + 1e-8]
        if not accepted:
            raise ValueError("GARCH QMLE did not converge; no fallback generator was substituted")
        fit = min(accepted, key=lambda result: result.fun)
        self._parameters = fit.x
        self._variances = variances(fit.x)
        residuals = errors / np.sqrt(self._variances)
        residuals -= residuals.mean()
        self._residuals = residuals / np.sqrt(np.mean(residuals**2))
        self._history = values
        return self

    def sample(self, n_paths: int, n_steps: int, rng: np.random.Generator) -> FloatArray:
        if self._history is None or n_paths < 1 or n_steps < 1:
            raise ValueError("fit GARCH first and use positive path/step counts")
        if self._scale == 0:
            return np.full((n_paths, n_steps), self._mean)
        omega, alpha, beta = self._parameters
        state = rng.choice(self._variances, size=n_paths)
        paths = np.empty((n_paths, n_steps))
        for t in range(n_steps):
            shock = rng.choice(self._residuals, size=n_paths)
            error = np.sqrt(state) * shock
            paths[:, t] = self._mean + self._scale * error
            state = omega + alpha * error**2 + beta * state
        return paths

    def diagnostics(self) -> dict:
        omega, alpha, beta = self._parameters
        return {"mean": self._mean, "omega": float(omega * self._scale**2),
                "alpha": float(alpha), "beta": float(beta), "persistence": float(alpha + beta),
                "residual_count": len(self._residuals), "initial_state": "random fitted historical variance"}


@dataclass
class QuantileGBDTGenerator:
    """Conditional increment quantiles from small histogram boosted-tree models.

    Features: last increment, trailing five-step mean, trailing twenty-step SD.
    Feature-bin boundaries are learned from the fit prefix only. Models are fitted
    on those bin indices; predictions for every possible bin combination are cached
    exactly. Recursive simulation then performs table lookups, not repeated tree
    inference. Nine quantiles define an interpolated inverse CDF. Linear end
    extrapolation defines the finite 0/1 endpoints; no pretrained weights are used.
    """

    quantiles: tuple[float, ...] = (.01, .05, .1, .25, .5, .75, .9, .95, .99)
    max_iter: int = 40
    _history: FloatArray | None = field(default=None, init=False, repr=False)
    _mean: float = field(default=0, init=False)
    _scale: float = field(default=0, init=False)
    _cuts: list[FloatArray] = field(default_factory=list, init=False, repr=False)
    _shape: tuple[int, ...] = field(default=(), init=False, repr=False)
    _table: FloatArray | None = field(default=None, init=False, repr=False)
    _models: list = field(default_factory=list, init=False, repr=False)

    def fit(self, increments: ArrayLike) -> QuantileGBDTGenerator:
        values = _history(increments, minimum=60)
        if self._history is not None and np.array_equal(values, self._history):
            return self
        if (len(self.quantiles) < 2 or not 0 < self.quantiles[0] < self.quantiles[-1] < 1
                or np.any(np.diff(self.quantiles) <= 0) or self.max_iter < 1):
            raise ValueError("use increasing interior quantiles and positive max_iter")
        self._mean, self._scale = float(values.mean()), float(values.std(ddof=1))
        if self._scale <= max(1.0, abs(self._mean)) * 1e-12:
            self._scale = 0.0
            self._history = values
            self._table = None
            self._models = []
            return self
        from sklearn.ensemble import HistGradientBoostingRegressor
        from threadpoolctl import threadpool_limits

        normalized = (values - self._mean) / self._scale
        past = np.lib.stride_tricks.sliding_window_view(normalized, 20)[:-1]
        features = np.column_stack([past[:, -1], past[:, -5:].mean(axis=1), past.std(axis=1)])
        self._cuts = [np.unique(np.quantile(features[:, i], np.arange(1, n) / n))
                      for i, n in enumerate((12, 12, 8))]
        binned = np.column_stack([np.searchsorted(cut, features[:, i], side="right")
                                 for i, cut in enumerate(self._cuts)])
        self._shape = tuple(len(cut) + 1 for cut in self._cuts)
        states = np.indices(self._shape).reshape(3, -1).T
        self._models = []
        predictions = []
        with threadpool_limits(limits=1):
            for quantile in self.quantiles:
                model = HistGradientBoostingRegressor(
                    loss="quantile", quantile=quantile, max_iter=self.max_iter,
                    learning_rate=.08, max_leaf_nodes=7, max_depth=3,
                    min_samples_leaf=20, l2_regularization=1.0,
                    early_stopping=False, categorical_features=None, random_state=0,
                ).fit(binned, normalized[20:])
                self._models.append(model)
                predictions.append(model.predict(states))
        # Monotone rearrangement resolves quantile crossing before simulation.
        self._table = np.sort(np.column_stack(predictions), axis=1)
        self._history = values
        return self

    def sample(self, n_paths: int, n_steps: int, rng: np.random.Generator) -> FloatArray:
        if self._history is None or n_paths < 1 or n_steps < 1:
            raise ValueError("fit quantile GBDT first and use positive path/step counts")
        if self._scale == 0:
            return np.full((n_paths, n_steps), self._mean)
        normalized = (self._history - self._mean) / self._scale
        starts = rng.integers(0, len(normalized) - 19, size=n_paths)
        buffer = normalized[starts[:, None] + np.arange(20)]
        # Ring buffers support lagged mean and volatility in O(n_paths) per step.
        total, squares = buffer.sum(axis=1), (buffer**2).sum(axis=1)
        recent = buffer[:, -5:].sum(axis=1)
        last = buffer[:, -1].copy()
        paths = np.empty((n_paths, n_steps))
        levels = np.asarray(self.quantiles)
        for t in range(n_steps):
            volatility = np.sqrt(np.maximum(squares / 20 - (total / 20)**2, 0))
            features = (last, recent / 5, volatility)
            indices = [np.searchsorted(cut, feature, side="right")
                       for cut, feature in zip(self._cuts, features)]
            flat = np.ravel_multi_index(tuple(indices), self._shape)
            quantiles = self._table[flat]
            uniforms = rng.random(n_paths)
            lower = np.clip(np.searchsorted(levels, uniforms, side="right") - 1, 0, len(levels) - 2)
            fraction = (uniforms - levels[lower]) / (levels[lower + 1] - levels[lower])
            rows = np.arange(n_paths)
            new = quantiles[rows, lower] + fraction * (quantiles[rows, lower + 1] - quantiles[rows, lower])
            paths[:, t] = self._mean + self._scale * new
            old = buffer[:, t % 20].copy()
            old_recent = buffer[:, (t - 5) % 20].copy()
            total += new - old
            squares += new**2 - old**2
            recent += new - old_recent
            buffer[:, t % 20] = new
            last = new
        return paths

    def diagnostics(self) -> dict:
        return {"quantiles": list(self.quantiles), "max_iter": self.max_iter,
                "features": ["last_increment", "mean_5", "sd_20"],
                "feature_bins": list(self._shape), "constant_history": self._scale == 0,
                "cdf_tails": "linear extrapolation to finite 0/1 endpoints"}


def make_generator(
    name: str, mean_block_length: float = 20.0, *, student_df: float = 5.0,
    scattering_options: dict | None = None,
) -> PathGenerator:
    """The iid variant is an ablation: it discards all temporal dependence."""
    if name == "stationary-bootstrap":
        return StationaryBootstrapGenerator(mean_block_length)
    if name == "iid-bootstrap":
        return StationaryBootstrapGenerator(mean_block_length=1.0)
    if name == "student-t":
        return StudentTRandomWalkGenerator(student_df)
    if name == "garch-bootstrap":
        return GARCHBootstrapGenerator()
    if name == "quantile-gbdt":
        return QuantileGBDTGenerator()
    if name == "scattering-spectra":
        from scattering_generator import ScatteringSpectraGenerator
        return ScatteringSpectraGenerator(**(scattering_options or {}))
    raise ValueError(f"unknown generator: {name}")
