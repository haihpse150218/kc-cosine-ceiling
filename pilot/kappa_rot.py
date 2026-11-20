"""Rest of P1 + inputs for L-Kernel (08_plan.md T2 day 2) + the likelihood-ratio ceiling of Corollary 1.

(a) kappa(c): expected KC code overlap as a function of input cosine, Gaussian W (m=2000, k=5%), measured on
    synthetic pairs with exact cosine c (Lemma 1 => cosine-only), one fresh W per pair; compared with the
    large-m approximation P(Z1>t, Z2>t; rho=c)/frac, t = Phi^-1(1-frac). Saved as a table for L-Kernel.
(b) Rotation KS on REAL embedding pairs (same-person and stranger pairs from the TEST identities), one fresh W
    per pair, overlap with and without a random rotation Q; Gaussian and sparse W. Also the residual of real
    Gaussian overlaps from kappa(cos) (cosine-only on real data).
(c) Corollary 1 ceiling = AUC of the likelihood-ratio test on the cosine score, not of raw cosine (equal only
    when the LR is monotone in cosine). Estimated by 2-fold cross-fitted binned LR (20 quantile bins, split by
    person) for the v1 (image-level look-alikes) and v2 (identity-level) tasks.
Output: results/kappa_rot.json. Usage: .venv/Scripts/python.exe pilot/kappa_rot.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import ks_2samp, multivariate_normal, norm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ceiling_kc as ck  # noqa: E402
from kc import make_W, codes, overlap, random_rotation  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
M, FRAC, SEED = 2000, 0.05, 20261124
GRID = np.round(np.arange(0.0, 1.0, 0.05), 2).tolist() + [0.97, 0.99]
N_GRID, N_PAIRS = 200, 300


def pair_with_cos(x, c, rng):
    u = x / np.linalg.norm(x)
    v = rng.standard_normal(x.shape); v -= v @ u * u; v /= np.linalg.norm(v)
    return c * u + np.sqrt(1 - c * c) * v


def ov(kind, x, y, rng, rot=False):
    if rot:
        Q = random_rotation(len(x), rng); x, y = Q @ x, Q @ y
    W = make_W(kind, M, len(x), rng)
    return float(overlap(codes(W, x[None], FRAC), codes(W, y[None], FRAC))[0])


def kappa_curve(rng, d=384):
    t = norm.ppf(1 - FRAC)
    rows = []
    for c in GRID:
        o = [ov('gaussian', x, pair_with_cos(x, c, rng), rng) for x in rng.standard_normal((N_GRID, d))]
        approx = multivariate_normal([0, 0], [[1, c], [c, 1]]).cdf([-t, -t]) / FRAC  # P(Z1>t,Z2>t) by symmetry
        rows.append({'cos': c, 'mean': float(np.mean(o)), 'sd': float(np.std(o)), 'approx_large_m': float(approx)})
        print(f'kappa cos={c:.2f} mean={np.mean(o):.4f} approx={approx:.4f}', flush=True)
    return rows


def real_pairs(E, pid, split, rng, n):
    t_idx = np.where(split == 'test')[0]
    by = {}
    for i in t_idx:
        by.setdefault(pid[i], []).append(i)
    ids = list(by)
    same = [tuple(rng.choice(by[ids[j]], 2, replace=False)) for j in rng.choice(len(ids), n)]
    diff = [tuple(rng.choice(t_idx, 2, replace=False)) for _ in range(n)]
    return same, diff


def lr_ceiling(score, y, person, rng, bins=20):
    folds = rng.integers(0, 2, person.max() + 1)[person]
    out = np.empty(len(y))
    for f in (0, 1):
        tr, te = folds != f, folds == f
        edges = np.unique(np.quantile(score[tr], np.linspace(0, 1, bins + 1)))
        b_tr = np.clip(np.searchsorted(edges, score[tr]) - 1, 0, len(edges) - 2)
        b_te = np.clip(np.searchsorted(edges, score[te]) - 1, 0, len(edges) - 2)
        p = np.array([(y[tr][b_tr == b].sum() + 0.5) / ((b_tr == b).sum() + 1) for b in range(len(edges) - 1)])
        out[te] = p[b_te]
    return float(roc_auc_score(y, out))


def main():
    rng = np.random.default_rng(SEED)
    res = {'m': M, 'frac': FRAC, 'kappa_gaussian': kappa_curve(rng)}
    res['real'] = {}
    for bb in ['facenet', 'dinov2']:
        E, pid, split = ck.load(bb)
        r = {}
        same, diff = real_pairs(E, pid, split, rng, N_PAIRS)
        kap = res['kappa_gaussian']
        for kind in ['gaussian', 'sparse']:
            for name, pairs in [('same', same), ('stranger', diff)]:
                base = np.array([ov(kind, E[a], E[b], rng) for a, b in pairs])
                rot = np.array([ov(kind, E[a], E[b], rng, rot=True) for a, b in pairs])
                cs = np.array([E[a] @ E[b] for a, b in pairs])
                pred = np.interp(cs, [k['cos'] for k in kap], [k['mean'] for k in kap])
                r[f'{kind}_{name}'] = {'mean_overlap': float(base.mean()), 'mean_overlap_rot': float(rot.mean()),
                                       'p_ks_rotation': float(ks_2samp(base, rot).pvalue),
                                       'mean_cos': float(cs.mean()),
                                       'mean_resid_vs_kappa': float((base - pred).mean()),
                                       'sd_resid_vs_kappa': float((base - pred).std())}
                print(bb, kind, name, {k: round(v, 4) for k, v in r[f'{kind}_{name}'].items()}, flush=True)
        for look in ['image', 'identity']:
            ck.LOOK = look
            enrol, qp, qi, ql, qk, _ = ck.build_task(E, pid, split, np.random.default_rng(ck.SEED))
            cos_q = np.einsum('qd,qkd->qk', E[qi], E[enrol[qp]]).max(1)
            m = (qk == 'pos') | (qk == 'look')
            r[f'task_{look}'] = {'auc_raw_cos': float(roc_auc_score(ql[m], cos_q[m])),
                                 'auc_lr_ceiling_crossfit': lr_ceiling(cos_q[m], ql[m], qp[m], rng)}
            print(bb, look, r[f'task_{look}'], flush=True)
        res['real'][bb] = r
        with open(os.path.join(RES, 'kappa_rot.json'), 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    sys.exit(main())
