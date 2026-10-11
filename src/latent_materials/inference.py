"""Checkpoint-safe target inverse transformation.

Use the fitted transformer from the original training split, including
its standardization parameters. Never reconstruct it from all labels.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted


def inverse_target_transform(values, fitted_transformer):
    if fitted_transformer is None:
        raise ValueError("Original fitted target transformer is required")
    try:
        check_is_fitted(fitted_transformer)
        if getattr(fitted_transformer, "standardize", False):
            check_is_fitted(fitted_transformer._scaler)
    except (AttributeError, TypeError, ValueError) as e:
        raise ValueError("Target transformer lacks fitted training state") from e
    a = np.asarray(values, float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[1] != 1 or not np.isfinite(a).all():
        raise ValueError("Expected finite single-target predictions")
    if not a.size:
        return np.array([], float)
    return fitted_transformer.inverse_transform(a).ravel()
