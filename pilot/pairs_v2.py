"""Real-pair overlap checks, v2 (after the second review, B7/B8).

For each backbone: 600 same-person and 600 stranger pairs of TEST identities. Pairs 0-299 are scored without
rotation and pairs 300-599 with a fresh Haar rotation, so the two KS samples are independent. One fresh W per
pair (Gaussian and sparse binary). Residual = overlap - kappa(cos) with the closed form (Owen's T), valid for
negative cosines too. Binned means (by cosine, width 0.1) of per-pair overlaps are saved for Fig. 1a, so the
plot compares like with like (no Jensen effect from averaging cosines first).
Output: results/pairs_v2.json (aggregates only).
"""
import json
import os
import sys

import numpy as np
from scipy.special import owens_t
from scipy.stats import ks_2samp, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ceiling_kc as ck  # noqa: E402
from kc import make_W, codes, overlap, random_rotation  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
M, FRAC, N, SEED = 2000, 0.05, 600, 20261122
T = norm.ppf(1 - FRAC)


def kappa(c):
    c = np.clip(c, -0.999999, 0.999999)
    return (1 - norm.cdf(T) - 2 * owens_t(T, np.sqrt((1 - c) / (1 + c)))) / FRAC


def ov(kind, x, y, rng, rot):
    if rot:
        Q = random_rotation(len(x), rng)
        x, y = Q @ x, Q @ y
    W = make_W(kind, M, len(x), rng)
    return float(overlap(codes(W, x[None], FRAC), codes(W, y[None], FRAC))[0])


def main():
    rng = np.random.default_rng(SEED)
    out = {'closed_form_check': {str(c): float(kappa(c)) for c in [-0.5, 0.0, 0.5, 0.9]}}
    bins = np.round(np.arange(-0.3, 1.0001, 0.1), 2)
    for bb in ['facenet', 'dinov2']:
        E, pid, split = ck.load(bb)
        t_idx = np.where(split == 'test')[0]
        by = {}
        for i in t_idx:
            by.setdefault(int(pid[i]), []).append(i)
        ids = list(by)
        same = [tuple(rng.choice(by[ids[j]], 2, replace=False)) for j in rng.choice(len(ids), N)]
        diff = []
        while len(diff) < N:
            a, b = rng.choice(t_idx, 2, replace=False)
            if pid[a] != pid[b]:
                diff.append((a, b))
        r = {}
        allc, allo = {'gaussian': [], 'sparse': []}, {'gaussian': [], 'sparse': []}
        for kind in ['gaussian', 'sparse']:
            for name, pairs in [('same', same), ('stranger', diff)]:
                cs = np.array([E[a] @ E[b] for a, b in pairs])
                o = np.array([ov(kind, E[a], E[b], rng, rot=(i >= N // 2)) for i, (a, b) in enumerate(pairs)])
                h = N // 2
                res = o - kappa(cs)
                r[f'{kind}_{name}'] = {'p_ks_rotation_independent': float(ks_2samp(o[:h], o[h:]).pvalue),
                                       'mean_overlap_norot': float(o[:h].mean()), 'mean_overlap_rot': float(o[h:].mean()),
                                       'mean_residual': float(res.mean()), 'se_residual': float(res.std() / np.sqrt(N)),
                                       'mean_residual_norot': float(res[:h].mean()), 'mean_residual_rot': float(res[h:].mean())}
                allc[kind] += list(cs)
                allo[kind] += list(o)
                print(bb, kind, name, {k: round(v, 4) for k, v in r[f'{kind}_{name}'].items()}, flush=True)
        r['binned'] = {}
        for kind in ['gaussian', 'sparse']:
            c, o = np.array(allc[kind]), np.array(allo[kind])
            rows = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                s = (c >= lo) & (c < hi)
                if s.sum() >= 20:
                    rows.append({'cos_mid': float((lo + hi) / 2), 'n': int(s.sum()), 'mean_overlap': float(o[s].mean()),
                                 'se': float(o[s].std() / np.sqrt(s.sum())), 'mean_kappa': float(kappa(c[s]).mean())})
            r['binned'][kind] = rows
        out[bb] = r
        json.dump(out, open(os.path.join(RES, 'pairs_v2.json'), 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
