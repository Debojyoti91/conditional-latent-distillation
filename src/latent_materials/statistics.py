"""Numerical controls following the supplied ablation and comparison formulas.

These helpers make RNG state and input validation explicit. Experiment implementations preserve the reported random-number order.
"""

import numpy as np
from scipy.stats import mannwhitneyu


def _vector(x):
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all():
        raise ValueError("Expected a nonempty finite one-dimensional array")
    return a


def latent_metrics(truth, prediction):
    t, p = np.asarray(truth, float), np.asarray(prediction, float)
    if t.ndim != 2 or t.shape != p.shape or not t.size:
        raise ValueError("Latent arrays must be nonempty matrices of identical shape")
    if not np.isfinite(t).all() or not np.isfinite(p).all():
        raise ValueError("Latents must be finite")
    error = p - t
    mse = float(np.mean(error**2))
    mol_rmse = np.sqrt(np.mean(error**2, axis=1))
    r = (
        float(np.corrcoef(t.ravel(), p.ravel())[0, 1])
        if np.std(t) and np.std(p)
        else np.nan
    )
    return dict(
        test_mse=mse,
        test_rmse=float(np.sqrt(mse)),
        pearson_r=r,
        molecule_rmse_mean=float(mol_rmse.mean()),
        molecule_rmse=mol_rmse,
    )


def paired_bootstrap_improvement(better, comparator, n_bootstrap=10000, seed=12345):
    a, b = _vector(better), _vector(comparator)
    if a.shape != b.shape:
        raise ValueError("Paired errors must have identical shape")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    d = b - a
    rng = np.random.default_rng(seed)
    draws = np.array(
        [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_bootstrap)]
    )
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return dict(
        mean_RMSE_improvement=float(d.mean()),
        CI95_low=float(lo),
        CI95_high=float(hi),
        bootstrap_fraction_improvement_le_0=float(np.mean(draws <= 0)),
    )


def corrected_empirical_p(null_values, observed):
    """Lower errors are better: (count(null <= observed)+1)/(m+1)."""
    null = _vector(null_values)
    if not np.isfinite(observed):
        raise ValueError("Observed value must be finite")
    return float((np.count_nonzero(null <= observed) + 1) / (len(null) + 1))


def bh_fdr(p):
    a = np.asarray(p, float)
    if a.ndim != 1 or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError("p values must be finite and lie in [0, 1]")
    if not a.size:
        return a.copy()
    order = np.argsort(a, kind="stable")
    q = np.minimum.accumulate((a[order] * len(a) / np.arange(1, len(a) + 1))[::-1])[
        ::-1
    ]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def cliffs_delta(bo, nn):
    b, a = _vector(bo), _vector(nn)
    u, p = mannwhitneyu(b, a, alternative="two-sided")
    return float(2 * u / (len(b) * len(a)) - 1), float(p)


def permutation_p(bo, nn, n=10000, seed=20260911):
    b, a = _vector(bo), _vector(nn)
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    obs = abs(b.mean() - a.mean())
    pooled = np.concatenate([b, a])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        x = rng.permutation(pooled)
        count += abs(x[: len(b)].mean() - x[len(b) :].mean()) >= obs
    return float((count + 1) / (n + 1))


def shuffle_properties_within_partitions(properties, partitions, seed):
    """Shuffle whole property pairs independently within disjoint partitions."""
    a = np.asarray(properties, float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Expected finite matrix")
    parts = [np.asarray(i) for i in partitions]
    if any(i.ndim != 1 or i.dtype.kind not in "iu" for i in parts):
        raise ValueError("Partition indices must be integer vectors")
    flat = np.concatenate(parts) if parts else np.array([], int)
    if len(np.unique(flat)) != len(flat) or np.any(flat < 0) or np.any(flat >= len(a)):
        raise ValueError("Partitions must be disjoint and in range")
    rng = np.random.default_rng(seed)
    out = a.copy()
    for idx in parts:
        out[idx] = a[rng.permutation(idx)]
    return out
