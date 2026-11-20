"""Kenyon-cell (KC) coding used throughout the pilot.

Two projection types (08_plan.md, section 2):
  - gaussian: W in R^{m x d}, i.i.d. N(0, 1)                         (the case covered by Lemma 1)
  - sparse:   W in {0,1}^{m x d}, each KC samples `fan_in` inputs     (the thesis design, D4.2)
k-WTA keeps the top-k activations (k = round(frac * m)) and returns a binary code.
"""
import numpy as np


def make_W(kind, m, d, rng, fan_in=(6, 10)):
    if kind == 'gaussian':
        return rng.standard_normal((m, d))
    if kind == 'sparse':
        W = np.zeros((m, d))
        lo, hi = fan_in
        for r in range(m):
            n = rng.integers(lo, hi + 1)
            W[r, rng.choice(d, size=n, replace=False)] = 1.0
        return W
    raise ValueError(kind)


def kwta(Z, k):
    """Binary top-k code per column of Z (shape m x n)."""
    idx = np.argpartition(-Z, k - 1, axis=0)[:k]
    H = np.zeros_like(Z, dtype=bool)
    np.put_along_axis(H, idx, True, axis=0)
    return H


def codes(W, X, frac=0.05):
    """X: n x d inputs -> n x m boolean KC codes."""
    m = W.shape[0]
    k = max(1, int(round(frac * m)))
    return kwta(W @ X.T, k).T


def overlap(Ha, Hb):
    """Fraction of shared active KCs (|a & b| / k) for paired rows."""
    k = Ha.sum(1)
    return (Ha & Hb).sum(1) / k


def random_rotation(d, rng):
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))
