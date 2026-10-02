"""Condition generated paths on an observed past using Path Shadowing Monte Carlo.

Only past prefixes determine neighbours and weights. Whole future continuations
are resampled together, preserving dependence between forecast horizons.
This is a small NumPy implementation of the identity-embedding PSMC baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from embeddings import foveal_features, foveal_lags

FloatArray = NDArray[np.float64]


def effective_sample_size(weights: FloatArray) -> float:
    """Number of equally weighted neighbours with the same weight concentration."""
    return float(1.0 / np.dot(weights, weights))


def _kernel_weights(energies: FloatArray, temperature: float) -> FloatArray:
    shifted = energies - energies.min()
    weights = np.exp(-shifted / temperature)
    return weights / weights.sum()


def _adaptive_weights(
    energies: FloatArray, target_ess: float
) -> tuple[FloatArray, float | None]:
    """Choose the narrowest Gaussian kernel attaining the requested ESS.

    Kernel adaptation uses past distances only; no future or outcome is read.
    Tied distances and constant histories have a well-defined uniform limit.
    """
    spread = float(np.ptp(energies))
    if spread == 0 or target_ess >= energies.size:
        return np.full(energies.size, 1.0 / energies.size), None
    normalized = (energies - energies.min()) / spread
    low, high = np.finfo(float).eps, 1.0
    while effective_sample_size(_kernel_weights(normalized, high)) < target_ess:
        high *= 2.0
    for _ in range(64):
        middle = (low + high) / 2.0
        if effective_sample_size(_kernel_weights(normalized, middle)) < target_ess:
            low = middle
        else:
            high = middle
    return _kernel_weights(normalized, high), float(np.sqrt(high * spread / 2.0))


@dataclass(frozen=True)
class ShadowingResult:
    future_steps: FloatArray
    neighbour_indices: NDArray[np.int64]
    distances: FloatArray
    weights: FloatArray
    sampled_indices: NDArray[np.int64]
    bandwidth: float | None
    n_candidates: int
    embedding: str = "identity"
    embedding_dimension: int = 0

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_candidates": self.n_candidates,
            "n_neighbours": int(self.weights.size),
            "effective_sample_size": effective_sample_size(self.weights),
            "bandwidth": self.bandwidth,
            "distance_min": float(self.distances.min()),
            "distance_median": float(np.median(self.distances)),
            "distance_max": float(self.distances.max()),
            "max_weight": float(self.weights.max()),
            "unique_sampled_paths": int(np.unique(self.sampled_indices).size),
            "embedding": self.embedding,
            "embedding_dimension": self.embedding_dimension,
        }


@dataclass(frozen=True)
class PSMC:
    """Nearest pasts + Gaussian kernel + weighted future resampling.

    Distance is ||generated_past - observed_past|| / ||observed_past||,
    as in the authors' RelativeMSE identity baseline. For multiple assets,
    normalize each asset separately, then flatten the time/asset dimensions.
    A resampled path keeps every asset and horizon together. For an all-zero observed
    past, use the generated past RMS norm, then a numerical floor.
    bandwidth=None adapts the kernel to ess_fraction * top_k neighbours.
    """

    top_k: int = 512
    ess_fraction: float = 0.5
    bandwidth: float | None = None
    embedding: str = "identity"
    foveal_alpha: float = 1.15
    foveal_beta: float = .9

    def __post_init__(self) -> None:
        if self.embedding not in {"identity", "foveal"}:
            raise ValueError("embedding must be identity or foveal")
        if self.embedding == "foveal":
            foveal_lags(126, self.foveal_alpha)
            if not np.isfinite(self.foveal_beta) or self.foveal_beta < 0:
                raise ValueError("Foveal beta must be finite and >=0")
        if self.top_k < 1 or not 0 < self.ess_fraction <= 1:
            raise ValueError("top_k must be positive and 0 < ess_fraction <= 1")
        if self.bandwidth is not None and (
            not np.isfinite(self.bandwidth) or self.bandwidth <= 0
        ):
            raise ValueError("bandwidth must be positive and finite")

    def condition(
        self,
        observed_past: ArrayLike,
        generated_paths: ArrayLike,
        n_draws: int,
        rng: np.random.Generator,
    ) -> ShadowingResult:
        observed = np.asarray(observed_past, dtype=np.float64)
        paths = np.asarray(generated_paths, dtype=np.float64)
        if observed.ndim not in (1, 2) or observed.size == 0 or not np.isfinite(observed).all():
            raise ValueError("observed_past must be finite [time] or [time, asset]")
        if (
            paths.ndim != observed.ndim + 1
            or paths.shape[0] < 1
            or paths.shape[1] <= observed.shape[0]
            or paths.shape[2:] != observed.shape[1:]
            or not np.isfinite(paths).all()
            or n_draws < 1
        ):
            raise ValueError("provide finite [path, past+future] paths and positive n_draws")
        past = paths[:, : observed.shape[0]]
        features = observed
        if self.embedding == "foveal":
            features = foveal_features(observed, 0, self.foveal_alpha, self.foveal_beta)
            past = foveal_features(past, 1, self.foveal_alpha, self.foveal_beta)
        scale = np.linalg.norm(features, axis=0)
        fallback = np.sqrt(np.mean(np.sum(past**2, axis=1), axis=0))
        scale = np.where(scale > 0, scale, np.maximum(fallback, 1e-12))
        # Standardize each asset by its observed past norm. Currency quote units
        # must not determine which multivariate pasts are nearest.
        differences = ((past - features) / scale).reshape(len(paths), -1)
        distances = np.linalg.norm(differences, axis=1)
        if not np.isfinite(distances).all():
            raise ValueError("past distances overflowed; rescale the increments")
        k = min(self.top_k, paths.shape[0])
        # Stable sorting makes equal-distance neighbours reproducible.
        neighbours = np.argsort(distances, kind="stable")[:k]
        distances = distances[neighbours]
        energies = distances**2
        if not np.isfinite(energies).all():
            raise ValueError("past squared distances overflowed; rescale the increments")
        if self.bandwidth is None:
            weights, bandwidth = _adaptive_weights(energies, max(1.0, k * self.ess_fraction))
        else:
            # Taking the square root before dividing avoids overflow/underflow
            # of bandwidth**2. The closest neighbour always has log weight zero.
            with np.errstate(over="ignore", under="ignore", divide="ignore"):
                scaled = np.sqrt(energies - energies.min()) / self.bandwidth / np.sqrt(2.0)
                weights = np.exp(-(scaled**2))
            weights /= weights.sum()
            bandwidth = self.bandwidth
        selected = neighbours[rng.choice(k, size=n_draws, replace=True, p=weights)]
        return ShadowingResult(
            future_steps=paths[selected, observed.shape[0] :].copy(),
            neighbour_indices=neighbours,
            distances=distances,
            weights=weights,
            sampled_indices=selected,
            bandwidth=bandwidth,
            n_candidates=paths.shape[0],
            embedding=self.embedding,
            embedding_dimension=features.size,
        )
