"""Checks requested by the independent proof review (letter/submission/proof_review.md, M2 and M3).

M2  Cross-draw AUC. Corollary 1 bounds AUC(S) for a score whose W is drawn independently per trial, i.e. the
    probability that a positive scored with one draw beats a negative scored with an INDEPENDENT draw. The
    tables reported the mean of per-draw AUCs (both members of a pair share one W), which the Corollary does not
    cover. Here we recompute, with exactly the draws of ceiling_kc.py (same seeds), both quantities:
      per-draw AUC  = mean_r AUC(S^(r)_pos, S^(r)_neg)
      cross-draw AUC = mean_{r != r'} AUC(S^(r)_pos, S^(r')_neg)          (ties count 1/2)
M3  Sparse binary rows with n ones have E[w^T u] = (n/d) 1^T u, so sparse W also sees each embedding's component
    along the all-ones direction. We report 1^T u / sqrt(d) (the cosine with the all-ones direction) and repeat
    the sparse (no rotation) and Gaussian runs on MEAN-CENTRED embeddings (global mean of the pool removed, then
    renormalised); raw cosine is recomputed on the centred vectors as the reference.
Output: results/crossdraw.json. Task, split and look-alike rule = v2 (identity-level), as in ceiling_kc.py.
Usage: .venv/Scripts/python.exe pilot/crossdraw.py [facenet|dinov2 ...]
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ceiling_kc as ck  # noqa: E402
from kc import make_W, codes, random_rotation  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def auc2(pos, neg):
    """P(pos > neg) + 1/2 P(pos = neg) via ranks."""
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def kc_scores(E, enrol, qi, qp, kind, rot, draw):
    need = np.unique(np.concatenate([enrol.ravel(), qi]))
    pos_of = {v: i for i, v in enumerate(need)}
    r = np.random.default_rng(ck.SEED + 1000 * draw + (7 if rot else 0) + (13 if kind == 'sparse' else 0))
    X = E[need]
    if rot:
        X = X @ random_rotation(X.shape[1], r).T
    W = make_W(kind, ck.M_KC, X.shape[1], r)
    H = codes(W, X, ck.FRAC)
    k = H[0].sum()
    hq = H[[pos_of[i] for i in qi]]
    he = H[[[pos_of[i] for i in row] for row in enrol[qp]]]
    return (hq[:, None, :] & he).sum(2).max(1) / k


def run(E, enrol, qp, qi, ql, qk, cells):
    m = (qk == 'pos') | (qk == 'look')
    y = ql[m]
    cos_q = np.einsum('qd,qkd->qk', E[qi], E[enrol[qp]]).max(1)
    out = {'auc_raw_cos': float(roc_auc_score(y, cos_q[m]))}
    for kind, rot in cells:
        S = np.array([kc_scores(E, enrol, qi[m], qp[m], kind, rot, d) for d in range(ck.N_DRAWS)])
        P, N = S[:, y == 1], S[:, y == 0]
        per = [auc2(P[r], N[r]) for r in range(len(S))]
        cross = [auc2(P[r], N[s]) for r in range(len(S)) for s in range(len(S)) if r != s]
        out[f'{kind}_rot{int(rot)}'] = dict(per_draw_mean=float(np.mean(per)), per_draw_sd=float(np.std(per)),
                                            cross_draw=float(np.mean(cross)),
                                            diff_cross_minus_per=float(np.mean(cross) - np.mean(per)))
        print(f'  {kind} rot{int(rot)}: per-draw {np.mean(per):.4f}  cross-draw {np.mean(cross):.4f}', flush=True)
    return out


def main():
    ck.LOOK = 'identity'
    res = {}
    for bb in sys.argv[1:] or ['facenet', 'dinov2']:
        E, pid, split = ck.load(bb)
        enrol, qp, qi, ql, qk, _ = ck.build_task(E, pid, split, np.random.default_rng(ck.SEED))
        ones = E @ (np.ones(E.shape[1]) / np.sqrt(E.shape[1]))
        r = {'cos_with_all_ones': {'mean': float(ones.mean()), 'sd': float(ones.std()),
                                   'p5': float(np.percentile(ones, 5)), 'p95': float(np.percentile(ones, 95))}}
        print(bb, 'cos(u, 1/sqrt d):', r['cos_with_all_ones'], flush=True)
        print(' original embeddings', flush=True)
        r['original'] = run(E, enrol, qp, qi, ql, qk,
                            [('gaussian', False), ('gaussian', True), ('sparse', False), ('sparse', True)])
        Ec = E - E[split == 'pool'].mean(0)
        Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
        print(' mean-centred embeddings', flush=True)
        r['centred'] = run(Ec, enrol, qp, qi, ql, qk, [('gaussian', False), ('sparse', False)])
        res[bb] = r
        json.dump(res, open(os.path.join(RES, 'crossdraw.json'), 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
