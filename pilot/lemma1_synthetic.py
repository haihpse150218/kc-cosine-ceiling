"""Numerical check of Lemma 1 on synthetic vectors (no face data needed).

Lemma 1: with i.i.d. Gaussian W and k-WTA, the joint law of KC codes depends only on the cosine matrix of the
inputs. Testable consequences checked here:
  (a) rotation invariance: overlap(x, y) has the same distribution when (x, y) -> (Qx, Qy);
  (b) scale invariance:   overlap(x, y) unchanged when x -> c x (c > 0);
  (c) cosine-only:        two pairs with the SAME cosine but very different geometry (one pair inside a dense
                          'face-like' cluster, the other made of isotropic random vectors) give the same overlap law.
For contrast, the same checks are run with the sparse binary W of the thesis, for which Lemma 1 does not apply.
Output: results/lemma1_synthetic.json (+ printed table). Deterministic seeds; takes well under a minute on CPU.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import ks_2samp

from kc import make_W, codes, overlap, random_rotation

d, m, frac = 512, 2000, 0.05
N_W, N_PAIRS = 1, 400  # N_W unused: one fresh W per pair
OUT = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT, exist_ok=True)


def pair_with_cosine(base, c, rng):
    """Return y with cos(base, y) = c exactly (direction chosen at random orthogonal to base)."""
    u = base / np.linalg.norm(base)
    v = rng.standard_normal(base.shape)
    v -= v @ u * u
    v /= np.linalg.norm(v)
    return np.linalg.norm(base) * (c * u + np.sqrt(1 - c * c) * v)


def make_pairs(geom, c, n, rng):
    if geom == 'isotropic':
        X = rng.standard_normal((n, d))
    else:  # 'clustered': nonnegative, low-rank-ish vectors, like ReLU / face embeddings with a common mean
        mu = np.abs(rng.standard_normal(d)) * 2
        X = mu + 0.3 * rng.standard_normal((n, d)) @ np.diag(np.linspace(1, 0.05, d))
    Y = np.stack([pair_with_cosine(x, c, rng) for x in X])
    return X, Y


def overlaps(kind, X, Y, seed, transform=None):
    """One independent W per pair, so samples are i.i.d. draws of the per-pair overlap law (what Lemma 1
    speaks about). Sharing one W across many pairs makes samples dependent through W and through the
    cross-pair cosines, which a KS test would wrongly read as a violation."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(X))
    for i in range(len(X)):
        W = make_W(kind, m, d, rng)
        x, y = X[i:i + 1], Y[i:i + 1]
        if transform is not None:
            x, y = transform(x, y, rng)
        out[i] = overlap(codes(W, x, frac), codes(W, y, frac))[0]
    return out


def main():
    rng = np.random.default_rng(0)
    rows = []
    for kind in ['gaussian', 'sparse']:
        for c in [0.3, 0.6, 0.9]:
            Xi, Yi = make_pairs('isotropic', c, N_PAIRS, rng)
            Xc, Yc = make_pairs('clustered', c, N_PAIRS, rng)
            if kind == 'sparse':  # sparse binary W reads raw (nonnegative) inputs; keep same pairs
                pass
            base = overlaps(kind, Xc, Yc, 1)
            rot = overlaps(kind, Xc, Yc, 1, lambda X, Y, r: (lambda Q: (X @ Q.T, Y @ Q.T))(random_rotation(d, r)))
            scale = overlaps(kind, Xc, Yc, 1, lambda X, Y, r: (3.7 * X, Y))
            iso = overlaps(kind, Xi, Yi, 1)
            row = {
                'W': kind, 'cos': c,
                'mean_overlap_clustered': float(base.mean()),
                'mean_overlap_isotropic': float(iso.mean()),
                'p_rotation': float(ks_2samp(base, rot).pvalue),
                'p_scale': float(ks_2samp(base, scale).pvalue),
                'p_cosine_only': float(ks_2samp(base, iso).pvalue),
            }
            rows.append(row)
            print(f"{kind:8s} cos={c:.1f}  overlap clustered={row['mean_overlap_clustered']:.3f} "
                  f"isotropic={row['mean_overlap_isotropic']:.3f}  KS p: rot={row['p_rotation']:.3f} "
                  f"scale={row['p_scale']:.3f} cos-only={row['p_cosine_only']:.3g}", flush=True)
    with open(os.path.join(OUT, 'lemma1_synthetic.json'), 'w') as f:
        json.dump({'d': d, 'm': m, 'frac': frac, 'n_W': N_W, 'n_pairs': N_PAIRS, 'rows': rows}, f, indent=1)


if __name__ == '__main__':
    sys.exit(main())
