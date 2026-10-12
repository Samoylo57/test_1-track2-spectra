"""Portable non-neural centre correction of joint Spectra draws.

Inference reads only this invocation's past panel and a disclosed JSON tree
ensemble. No labels, historical panels, unit IDs or fitted answer table are
packaged. Dates gate all learned corrections, including parameter selection.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from center_features import features, FEATURE_NAMES


@lru_cache(maxsize=1)
def artifact():
    path = Path(__file__).resolve().parent / 'models/center.json'
    data = path.read_bytes()
    model = json.loads(data)
    if model['schema_version'] != 1 or tuple(model['features']) != FEATURE_NAMES:
        raise ValueError('unsupported centre model schema or feature order')
    if len(model['trees']) != 60 or model['training_rows'] != 952:
        raise ValueError('unexpected centre ensemble')
    return model, hashlib.sha256(data).hexdigest()


def predict_trees(model, X):
    """sklearn HistGradientBoosting raw numeric splits, with scaled leaf values."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or not np.isfinite(X).all():
        raise ValueError('finite aligned centre features required')
    result = np.full(len(X), model['baseline'], dtype=float)
    for tree in model['trees']:
        for row_index, row in enumerate(X):
            i = 0
            for _ in range(4):
                # value, feature index, threshold, left, right, leaf
                node = tree[i]
                if node[5]:
                    result[row_index] += node[0]
                    break
                i = node[3] if row[node[1]] <= node[2] else node[4]
            else:
                raise ValueError('centre tree exceeds supported depth')
    return result


def correct_draws(draws, values, horizons, assets, target_type, asof,
                  frequency='daily', transfer=False, scales=5):
    """Positive affine transforms preserve draw identities and joint ranks."""
    model, digest = artifact()
    x = np.asarray(draws, float)
    single = x.ndim == 2
    if single:
        x = x[:, :, None]
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError('finite joint forecast draws required')
    v = np.asarray(values, float)
    if v.ndim == 1:
        v = v[:, None]
    median = np.median(x, axis=0)
    sd = np.std(x, axis=0, ddof=1)
    delta_sd = np.zeros(median.shape)
    dates_allowed = asof > max(model['training_cutoff'], model['selection_cutoff'])
    width = model['width'] if dates_allowed else 1.
    reason = 'applied'
    if not dates_allowed:
        reason = 'training or parameter-selection data do not precede task cutoff'
    elif frequency != 'daily' or scales != 5 or len(v) < 257 or transfer:
        reason = 'centre fallback: daily contiguous J5 history of at least 257 observations required'
    else:
        X, _, _ = features(v, x, horizons, assets, target_type)
        raw = predict_trees(model, X).reshape(median.shape)
        delta_sd = np.clip(model['strength'] * raw, -1., 1.)
    delta = sd * delta_sd
    adjusted = median + delta + width * (x - median) if dates_allowed else x.copy()
    info = dict(model='Spectra normalized residual centre HistGradientBoosting',
        revision='sha256:' + digest, training_rows=model['training_rows'],
        training_cutoff=model['training_cutoff'], selection_cutoff=model['selection_cutoff'],
        parameters=model['parameters'], strength=model['strength'],
        width_factor=width, requested_width_factor=model['width'],
        center_applied=reason == 'applied', reason=reason,
        delta_sd=delta_sd.tolist(), delta=delta.tolist(),
        original_median=median.tolist(), original_sd=sd.tolist(),
        original_quantiles=np.quantile(x, [.05,.5,.95], axis=0).tolist(),
        joint_draw_order_preserved=True, text_used=False)
    return (adjusted[:, :, 0] if single else adjusted), info
