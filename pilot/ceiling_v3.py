"""Main analysis of the Letter, v3 (after the second independent review, letter/submission/proof_review_round2.md).

Changes from v2 (ceiling_kc.py + crossdraw.py):
  * Look-alike strangers are selected with the OTHER backbone (identity level: rank pool identities by the cosine
    between the person's enrolled prototype and the identity's half-A prototype, in the selector embedding), so
    negatives are never selected by the score under test. Each look-alike image is used by at most one test
    identity. Enrolled, positive and random-stranger images are identical for both columns.
  * KC AUCs are cross-draw (positive and negative scored with different, independent draws of W);
    cross = (R^2 * AUC(pooled over draws) - sum_r AUC_rr) / (R (R - 1)).
  * 95% CIs for every AUC and for KC - raw cosine: bootstrap over test identities (B = 500) with the 10 draws held
    fixed; the cross-draw AUC is recomputed inside each resample.
  * Lower bounds on the LR ceiling: cross-fitted binned LR on max-cosine, and a cross-fitted logistic score on
    several entries of C (sorted query-enrolled cosines, mean/min enrolled-enrolled cosine, square of the max).
  * Control for the all-ones mechanism of sparse W: project each embedding off 1/sqrt(d) and renormalise.
  * The v1 selection artefact (look-alikes = pool IMAGES most similar in the evaluated embedding) is recomputed
    on DINOv2 within one run (raw cosine, binned LR, KC Gaussian cross-draw).
Output: results/ceiling_v3.json (aggregate numbers only). Resumable per cell (results/v3/*.json).
Usage: .venv/Scripts/python.exe pilot/ceiling_v3.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ceiling_kc as ck  # noqa: E402
from crossdraw import kc_scores  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CELL = os.path.join(RES, 'v3')
N_BOOT, SEED = 500, 20261121
K, NP, NL, NR = ck.K_ENROL, ck.N_POS, ck.N_LOOK, ck.N_RAND


def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cross_auc(S, y):
    """S: (R, n) scores of the same n trials under R independent draws; y: labels."""
    R = len(S)
    P, N = S[:, y == 1], S[:, y == 0]
    pooled = auc(P.ravel(), N.ravel())
    diag = sum(auc(P[r], N[r]) for r in range(R))
    return (R * R * pooled - diag) / (R * (R - 1))


def build(Esel, pid, split, rng_seed):
    """Shared enrol/pos/rand; look-alikes selected in Esel (identity level, image reuse forbidden)."""
    rng = np.random.default_rng(rng_seed)
    test_ids = np.unique(pid[split == 'test'])
    pool_ids = np.unique(pid[split == 'pool'])
    poolA, poolB = [], []
    for s in pool_ids:
        idx = rng.permutation(np.where(pid == s)[0])
        h = len(idx) // 2
        poolA.append(idx[:h])
        poolB.append(list(idx[h:]))
    pool_idx = np.where(split == 'pool')[0]
    enrol, pos, rnd = [], [], []
    for person in test_ids:
        idx = rng.permutation(np.where(pid == person)[0])
        enrol.append(idx[:K])
        pos.append(idx[K:K + NP])
        rnd.append(rng.choice(pool_idx, NR, replace=False))
    A = np.array([Esel[a].mean(0) for a in poolA])
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    used, look = set(), []
    for e in enrol:
        proto = Esel[e].mean(0)
        proto /= np.linalg.norm(proto)
        got = []
        for j in np.argsort(-(A @ proto)):
            free = [i for i in poolB[j] if i not in used]
            if free:
                used.add(free[0])
                got.append(free[0])
            if len(got) == NL:
                break
        look.append(got)
    qp, qi, ql, qk = [], [], [], []
    for p in range(len(test_ids)):
        for i, lab, kind in [(i, 1, 'pos') for i in pos[p]] + [(i, 0, 'look') for i in look[p]] + \
                             [(i, 0, 'rand') for i in rnd[p]]:
            qp.append(p); qi.append(i); ql.append(lab); qk.append(kind)
    return np.array(enrol), np.array(qp), np.array(qi), np.array(ql), np.array(qk)


def features(E, enrol, qi, qp):
    cq = np.sort(np.einsum('qd,qkd->qk', E[qi], E[enrol[qp]]), axis=1)[:, ::-1]
    ee = np.einsum('pkd,pjd->pkj', E[enrol], E[enrol])
    iu = np.triu_indices(enrol.shape[1], 1)
    eev = ee[:, iu[0], iu[1]]
    F = np.column_stack([cq, eev.mean(1)[qp], eev.min(1)[qp], cq[:, 0] ** 2, cq[:, 0] * eev.mean(1)[qp]])
    return cq[:, 0], F


def crossfit(score_fn, y, person, seeds=5):
    vals = []
    for s in range(seeds):
        fold = np.random.default_rng(SEED + s).integers(0, 2, person.max() + 1)[person]
        out = np.empty(len(y))
        for f in (0, 1):
            tr, te = fold != f, fold == f
            out[te] = score_fn(tr, te)
        vals.append(auc(out[y == 1], out[y == 0]))
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def binned_lr(x, y, person, bins=20):
    def fn(tr, te):
        edges = np.unique(np.quantile(x[tr], np.linspace(0, 1, bins + 1)))
        btr = np.clip(np.searchsorted(edges, x[tr]) - 1, 0, len(edges) - 2)
        bte = np.clip(np.searchsorted(edges, x[te]) - 1, 0, len(edges) - 2)
        p = np.array([(y[tr][btr == b].sum() + .5) / ((btr == b).sum() + 1) for b in range(len(edges) - 1)])
        return p[bte]
    return crossfit(fn, y, person)


def logistic_lr(F, y, person):
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fz = (F - mu) / sd

    def fn(tr, te):
        return LogisticRegression(max_iter=2000).fit(Fz[tr], y[tr]).decision_function(Fz[te])
    return crossfit(fn, y, person)


def lr_monotone_table(x, y, bins=10):
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(edges, x) - 1, 0, bins - 1)
    return [float(y[b == i].mean()) for i in range(bins)]


def boot(cells_scores, raw, y, person, rng):
    """CI for each AUC and each (KC - raw) by bootstrap over identities with draws fixed."""
    persons = np.unique(person)
    by = [np.where(person == p)[0] for p in persons]
    res = {k: [] for k in ['raw'] + list(cells_scores)}
    diffs = {k: [] for k in cells_scores}
    for _ in range(N_BOOT):
        idx = np.concatenate([by[i] for i in rng.integers(0, len(persons), len(persons))])
        yy = y[idx]
        a_raw = auc(raw[idx][yy == 1], raw[idx][yy == 0])
        res['raw'].append(a_raw)
        for k, S in cells_scores.items():
            a = cross_auc(S[:, idx], yy)
            res[k].append(a)
            diffs[k].append(a - a_raw)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {k: q(v) for k, v in res.items()}, {k: q(v) for k, v in diffs.items()}


def cell(name, E, enrol, qp, qi, kind, rot):
    path = os.path.join(CELL, name + '.npy')
    if os.path.exists(path):
        return np.load(path)
    S = np.array([kc_scores(E, enrol, qi, qp, kind, rot, d) for d in range(ck.N_DRAWS)], dtype=np.float32)
    np.save(path + '.tmp.npy', S)
    os.replace(path + '.tmp.npy', path)
    return S


def main():
    os.makedirs(CELL, exist_ok=True)
    Ef, pid, split = ck.load('facenet')
    Ed, pid2, split2 = ck.load('dinov2')
    assert (pid == pid2).all() and (split == split2).all()
    E = {'facenet': Ef, 'dinov2': Ed}
    other = {'facenet': 'dinov2', 'dinov2': 'facenet'}
    out = {}
    for bb in ['facenet', 'dinov2']:
        enrol, qp, qi, ql, qk = build(E[other[bb]], pid, split, ck.SEED)
        m = (qk == 'pos') | (qk == 'look')
        y, person, Eb = ql[m], qp[m], E[bb]
        raw, F = features(Eb, enrol, qi[m], qp[m])
        r = {'selector': other[bb], 'n_trials': int(m.sum()), 'n_identities': int(len(np.unique(person))),
             'auc_raw': auc(raw[y == 1], raw[y == 0]),
             'lr_binned_maxcos': binned_lr(raw, y, person),
             'lr_logistic_C': logistic_lr(F, y, person),
             'P_y1_by_maxcos_decile': lr_monotone_table(raw, y)}
        print(bb, 'raw', round(r['auc_raw'], 4), 'binnedLR', r['lr_binned_maxcos'], 'logisticC', r['lr_logistic_C'], flush=True)
        print('  P(Y=1 | decile)', [round(v, 3) for v in r['P_y1_by_maxcos_decile']], flush=True)
        cells = {}
        for kind in ['gaussian', 'sparse']:
            for rot in [False, True]:
                S = cell(f'{bb}_{kind}_rot{int(rot)}', Eb, enrol, qp[m], qi[m], kind, rot)
                cells[f'{kind}_rot{int(rot)}'] = S
                print(f'  {kind} rot{int(rot)} cross-draw {cross_auc(S, y):.4f}', flush=True)
        ci, dci = boot(cells, raw, y, person, np.random.default_rng(SEED))
        r['auc_raw_ci'] = ci['raw']
        r['kc'] = {k: {'cross_draw': cross_auc(S, y), 'ci': ci[k], 'minus_raw': cross_auc(S, y) - r['auc_raw'],
                       'minus_raw_ci': dci[k],
                       'per_draw_sd': float(np.std([auc(S[d][y == 1], S[d][y == 0]) for d in range(len(S))]))}
                   for k, S in cells.items()}
        for k, v in r['kc'].items():
            print(f"  {k}: {v['cross_draw']:.4f} CI {np.round(v['ci'], 4)}  minus raw {v['minus_raw']:+.4f} CI {np.round(v['minus_raw_ci'], 4)}", flush=True)
        # all-ones control: project off 1/sqrt(d)
        one = np.ones(Eb.shape[1]) / np.sqrt(Eb.shape[1])
        Ep = Eb - np.outer(Eb @ one, one)
        Ep /= np.linalg.norm(Ep, axis=1, keepdims=True)
        rawp, _ = features(Ep, enrol, qi[m], qp[m])
        pc = {k: cell(f'{bb}_proj1_{k}', Ep, enrol, qp[m], qi[m], k.split('_')[0], False)
              for k in ['gaussian_rot0', 'sparse_rot0']}
        cip, dcip = boot(pc, rawp, y, person, np.random.default_rng(SEED + 1))
        r['project_out_ones'] = {'auc_raw': auc(rawp[y == 1], rawp[y == 0]), 'auc_raw_ci': cip['raw'],
                                 **{k: {'cross_draw': cross_auc(S, y), 'minus_raw': cross_auc(S, y) - auc(rawp[y == 1], rawp[y == 0]),
                                        'minus_raw_ci': dcip[k]} for k, S in pc.items()}}
        print('  project-out-1:', json.dumps(r['project_out_ones']), flush=True)
        out[bb] = r
        json.dump(out, open(os.path.join(RES, 'ceiling_v3.json'), 'w'), indent=1)
    # v1 artefact on DINOv2 in one run
    ck.LOOK = 'image'
    enrol, qp, qi, ql, qk, _ = ck.build_task(Ed, pid, split, np.random.default_rng(ck.SEED))
    m = (qk == 'pos') | (qk == 'look')
    y, person = ql[m], qp[m]
    raw, _ = features(Ed, enrol, qi[m], qp[m])
    S = cell('dinov2_v1image_gaussian_rot0', Ed, enrol, qp[m], qi[m], 'gaussian', False)
    out['v1_artefact_dinov2'] = {'auc_raw': auc(raw[y == 1], raw[y == 0]), 'lr_binned_maxcos': binned_lr(raw, y, person),
                                 'kc_gaussian_cross_draw': cross_auc(S, y)}
    print('v1 artefact dinov2', out['v1_artefact_dinov2'], flush=True)
    json.dump(out, open(os.path.join(RES, 'ceiling_v3.json'), 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
