"""Causal past embeddings, including equation (13) of the PSMC paper."""

import numpy as np


def foveal_lags(context_length, alpha=1.15):
    if not np.isfinite(alpha) or alpha <= 1 or context_length < 2:
        raise ValueError("Foveal alpha must exceed 1 and context must contain >=2 increments")
    dimension = int(np.floor(np.log(context_length) / np.log(alpha)))
    if not 1 <= dimension <= 10000:
        raise ValueError("Foveal alpha creates an empty or excessively large embedding")
    # Repeated short lags are intentional; both paper and authors' code keep them.
    return np.floor(alpha ** np.arange(1, dimension + 1)).astype(int)


def foveal_features(increments, time_axis=0, alpha=1.15, beta=.9):
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("Foveal beta must be finite and nonnegative")
    values = np.moveaxis(np.asarray(increments, dtype=float), time_axis, -1)
    lags = foveal_lags(values.shape[-1], alpha)
    past_changes = np.cumsum(values[..., ::-1], axis=-1)
    features = past_changes[..., lags - 1] / lags ** beta
    return np.moveaxis(features, -1, time_axis)
