"""Uncertainty for Table 1 after the third review (letter/submission/proof_review_round3.md, M2, m3, m5).

Uses the cached per-trial KC scores of ceiling_v3.py (results/v3/*.npy, not released) and rebuilds the same v3 task.
  * KC cross-draw AUC and (KC - raw cosine): total SE = sqrt(SE_identity^2 + SE_draw^2), where SE_identity is a
    bootstrap over test identities (draws fixed, B = 500) and SE_draw a leave-one-draw-out jackknife over the 10
    draws; 95% CI = estimate +/- 1.96 SE. Also the difference between the two Gaussian runs (identical in law) and
    between sparse no-rot and sparse rot, with the same total SE.
  * Ceiling lower bounds minus raw cosine (logistic score on C; binned LR on max-cosine): paired bootstrap over
    identities of one cross-fit (fold seed fixed), B = 500.
  * Sparse W with FIXED fan-in 8 on real pairs: is the small excess of sparse overlaps over kappa (pairs_v2) due to
    unequal fan-in (6-10)?
Output: results/ceiling_v3_ci.json (aggregates only).
"""
import json
import os
import sys

import numpy as np
from scipy.special import owens_t
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ceiling_kc as ck  # noqa: E402
from ceiling_v3 import build, features, auc, cross_auc, CELL, RES  # noqa: E402
from kc import make_W, codes, overlap  # noqa: E402

B, SEED = 500, 20261125


def by_person(person):
    persons = np.unique(person)
    return [np.where(person == p)[0] for p in persons]


def boot_idx(groups, rng):
    return np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def jack_se(vals):
    v = np.asarray(vals)
    n = len(v)
    return float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))


def crossfit_scores(F, raw, y, person, seed):
    fold = np.random.default_rng(seed).integers(0, 2, person.max() + 1)[person]
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fz = (F - mu) / sd
    lg, lr = np.empty(len(y)), np.empty(len(y))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        lg[te] = LogisticRegression(max_iter=2000).fit(Fz[tr], y[tr]).decision_function(Fz[te])
        edges = np.unique(np.quantile(raw[tr], np.linspace(0, 1, 21)))
        btr = np.clip(np.searchsorted(edges, raw[tr]) - 1, 0, len(edges) - 2)
        bte = np.clip(np.searchsorted(edges, raw[te]) - 1, 0, len(edges) - 2)
        p = np.array([(y[tr][btr == b].sum() + .5) / ((btr == b).sum() + 1) for b in range(len(edges) - 1)])
        lr[te] = p[bte]
    return lg, lr


def fanin_check(E, pid, split, rng, n=600):
    t = norm.ppf(0.95)
    kap = lambda c: (1 - norm.cdf(t) - 2 * owens_t(t, np.sqrt((1 - np.clip(c, -.999999, .999999)) / (1 + np.clip(c, -.999999, .999999))))) / 0.05
    t_idx = np.where(split == 'test')[0]
    by = {}
    for i in t_idx:
        by.setdefault(int(pid[i]), []).append(i)
    ids = list(by)
    pairs = [tuple(rng.choice(by[ids[j]], 2, replace=False)) for j in rng.choice(len(ids), n)]
    out = {}
    for fan in [(6, 10), (8, 8)]:
        res = []
        for a, b in pairs:
            W = make_W('sparse', 2000, E.shape[1], rng, fan_in=fan)
            o = overlap(codes(W, E[a][None], 0.05), codes(W, E[b][None], 0.05))[0]
            res.append(o - kap(E[a] @ E[b]))
        res = np.array(res)
        out[f'fan_in_{fan[0]}_{fan[1]}'] = {'mean_residual': float(res.mean()), 'se': float(res.std() / np.sqrt(n))}
    return out


def main():
    rng = np.random.default_rng(SEED)
    Ef, pid, split = ck.load('facenet')
    Ed, _, _ = ck.load('dinov2')
    E = {'facenet': Ef, 'dinov2': Ed}
    other = {'facenet': 'dinov2', 'dinov2': 'facenet'}
    out = {}
    for bb in ['facenet', 'dinov2']:
        enrol, qp, qi, ql, qk = build(E[other[bb]], pid, split, ck.SEED)
        m = (qk == 'pos') | (qk == 'look')
        y, person = ql[m], qp[m]
        raw, F = features(E[bb], enrol, qi[m], qp[m])
        groups = by_person(person)
        S = {k: np.load(os.path.join(CELL, f'{bb}_{k}.npy')) for k in
             ['gaussian_rot0', 'gaussian_rot1', 'sparse_rot0', 'sparse_rot1']}
        a_raw = auc(raw[y == 1], raw[y == 0])
        est = {k: cross_auc(v, y) for k, v in S.items()}
        # identity bootstrap (draws fixed)
        bt = {k: [] for k in list(S) + ['raw']}
        idxs = [boot_idx(groups, rng) for _ in range(B)]
        for idx in idxs:
            yy = y[idx]
            bt['raw'].append(auc(raw[idx][yy == 1], raw[idx][yy == 0]))
            for k, v in S.items():
                bt[k].append(cross_auc(v[:, idx], yy))
        bt = {k: np.array(v) for k, v in bt.items()}
        # draw jackknife (identities fixed)
        jk = {k: [cross_auc(np.delete(v, d, axis=0), y) for d in range(len(v))] for k, v in S.items()}

        def total(dv_boot, jk_vals):
            se_i = float(np.std(dv_boot))
            se_d = jack_se(jk_vals) if jk_vals is not None else 0.0
            return se_i, se_d, float(np.sqrt(se_i ** 2 + se_d ** 2))

        r = {'auc_raw': a_raw, 'auc_raw_ci_identity': [float(np.percentile(bt['raw'], 2.5)), float(np.percentile(bt['raw'], 97.5))], 'kc': {}}
        for k in S:
            se_i, se_d, se = total(bt[k], jk[k])
            d = est[k] - a_raw
            se_i2, _, _ = total(bt[k] - bt['raw'], None)
            se2 = float(np.sqrt(se_i2 ** 2 + se_d ** 2))
            r['kc'][k] = {'cross_draw': est[k], 'se_identity': se_i, 'se_draw': se_d, 'ci': [est[k] - 1.96 * se, est[k] + 1.96 * se],
                          'minus_raw': d, 'minus_raw_se': se2, 'minus_raw_ci': [d - 1.96 * se2, d + 1.96 * se2]}
        for a, b in [('gaussian_rot0', 'gaussian_rot1'), ('sparse_rot0', 'sparse_rot1'), ('sparse_rot0', 'gaussian_rot0')]:
            d = est[a] - est[b]
            se_i = float(np.std(bt[a] - bt[b]))
            se_d = float(np.sqrt(jack_se(jk[a]) ** 2 + jack_se(jk[b]) ** 2))  # independent draws per cell
            se = float(np.sqrt(se_i ** 2 + se_d ** 2))
            r[f'{a}_minus_{b}'] = {'diff': d, 'se': se, 'ci': [d - 1.96 * se, d + 1.96 * se]}
        lg, lr = crossfit_scores(F, raw, y, person, SEED)
        a_lg, a_lr = auc(lg[y == 1], lg[y == 0]), auc(lr[y == 1], lr[y == 0])
        dl = [auc(lg[i][y[i] == 1], lg[i][y[i] == 0]) - auc(raw[i][y[i] == 1], raw[i][y[i] == 0]) for i in idxs]
        dr = [auc(lr[i][y[i] == 1], lr[i][y[i] == 0]) - auc(raw[i][y[i] == 1], raw[i][y[i] == 0]) for i in idxs]
        r['logistic_C'] = {'auc': a_lg, 'minus_raw': a_lg - a_raw, 'minus_raw_ci': [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))]}
        r['binned_lr'] = {'auc': a_lr, 'minus_raw': a_lr - a_raw, 'minus_raw_ci': [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]}
        r['fan_in_check'] = fanin_check(E[bb], pid, split, np.random.default_rng(SEED + 7))
        out[bb] = r
        print(bb, json.dumps(r, indent=None)[:3000], flush=True)
        json.dump(out, open(os.path.join(RES, 'ceiling_v3_ci.json'), 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
