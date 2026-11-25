"""Deterministic decoder-to-anchor numerical retrieval, not chemical similarity.

The query uses the first 2048 decoder components and L2 normalization.
No Morgan-fingerprint semantics are assigned to the decoder output.
"""

import numpy as np


def anchor_indices(decoded, fingerprint_matrix):
    d, f = np.asarray(decoded, float), np.asarray(fingerprint_matrix, float)
    if (
        d.ndim != 2
        or f.ndim != 2
        or d.shape[1] < 2048
        or f.shape[1] != 2048
        or not len(f)
    ):
        raise ValueError(
            "Expected decoded (N, >=2048) and nonempty fingerprints (M, 2048)"
        )
    if not np.isfinite(d).all() or not np.isfinite(f).all():
        raise ValueError("Retrieval inputs must be finite")
    q = d[:, :2048]
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    scores = q @ f.T
    idx = np.argmax(scores, axis=1)  # deterministic first index on ties
    return idx, scores[np.arange(len(q)), idx]
