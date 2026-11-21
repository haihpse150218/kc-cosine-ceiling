"""Pilot week 2 (08_plan.md 3.1, T2 days 1-3): cosine ceiling, overlap statistic O, P1 and P2 on real faces.

Task (per backbone, on the TEST identities; look-alikes come from the disjoint STRANGER POOL):
  enrol K images of a known person; a query is either a new image of that person (positive) or an image of a
  stranger from the pool (negative). Two kinds of strangers: 'lookalike' = the pool images most similar to the
  person's enrolled prototype (hardest case, the seed of wrong spill-over), and 'random'.
  Score = max similarity between the query and the K enrolled items.
    cosine score  : max cos(query, enrolled)                          -> its AUC is the ceiling (Lemma 1, Cor. 1)
    KC score      : max |h(query) & h(enrolled)| / k  for W in {gaussian, sparse}, 10 independent draws of W
  O = share of test persons whose top-1 look-alike stranger is more similar to their prototype than the 5th
      percentile of positive (same-person) similarities.
  P1: MB-gaussian AUC <= ceiling + 0.005 (mean over draws).
  P2: Delta = AUC(MB-sparse) - ceiling >= 0.01 with 95% bootstrap CI (over persons) excluding 0, on both backbones;
      reported with and without a random rotation Q of the embedding (rotation control).
Results are written per (backbone, W, rotation, draw) to results/kc/*.json and skipped on restart.
Usage: .venv/Scripts/python.exe pilot/ceiling_kc.py [facenet|dinov2 ...]
"""
import glob
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kc import make_W, codes, random_rotation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(HERE, '..', 'data', 'emb')
RES = os.path.join(HERE, 'results')
K_ENROL, N_POS, N_LOOK, N_RAND = 5, 5, 5, 5
M_KC, FRAC, N_DRAWS, N_BOOT, SEED = 2000, 0.05, 10, 1000, 20261123
LOOK = os.environ.get('LOOK', 'identity')  # 'identity' (v2, default) | 'image' (v1, biased)
VER = 'v2' if LOOK == 'identity' else 'v1'


def load(bb):
    files = sorted(glob.glob(os.path.join(EMB, bb, 'shard_*.npz')))
    if not files:
        sys.exit(f'no shards for {bb}')
    parts = [np.load(f) for f in files]
    E = np.concatenate([p['emb'] for p in parts]).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    pid = np.concatenate([p['pid'] for p in parts])
    split = np.concatenate([p['split'] for p in parts])
    return E, pid, split


def build_task(E, pid, split, rng):
    """Returns index arrays: enrol (P x K), queries (person, idx, label, kind).

    v2 (LOOK='identity', default): look-alike strangers are chosen at the IDENTITY level. Each pool identity's
    images are split at random into a ranking half A and a query half B; strangers are ranked by
    cos(person prototype, stranger prototype from A) and the query is a random image from B. The query image
    therefore never took part in its own selection. v1 (LOOK='image') picked the pool IMAGES most similar to
    the prototype in the same embedding: the negatives are then selected on the very score being evaluated,
    so cosine is adversarially biased (winner's curse) and the 'ceiling' is not a ceiling (pilot log 15/11).
    """
    test_ids = np.unique(pid[split == 'test'])
    pool_ids = np.unique(pid[split == 'pool'])
    A_proto, B_imgs = [], []
    for s_id in pool_ids:
        idx = rng.permutation(np.where(pid == s_id)[0])
        h = len(idx) // 2
        v = E[idx[:h]].mean(0)
        A_proto.append(v / np.linalg.norm(v))
        B_imgs.append(idx[h:])
    A_proto = np.array(A_proto)
    pool_idx = np.where(split == 'pool')[0]
    Ep = E[pool_idx]
    enrol, q_person, q_idx, q_label, q_kind = [], [], [], [], []
    for p, person in enumerate(test_ids):
        idx = rng.permutation(np.where(pid == person)[0])
        enrol.append(idx[:K_ENROL])
        pos = idx[K_ENROL:K_ENROL + N_POS]
        proto = E[idx[:K_ENROL]].mean(0)
        proto /= np.linalg.norm(proto)
        if LOOK == 'image':
            order = np.argsort(-(Ep @ proto))
            look, seen = [], set()
            for j in order:
                if pid[pool_idx[j]] not in seen:
                    seen.add(pid[pool_idx[j]])
                    look.append(pool_idx[j])
                if len(look) == N_LOOK:
                    break
        else:
            top = np.argsort(-(A_proto @ proto))[:N_LOOK]
            look = [rng.choice(B_imgs[t]) for t in top]
        rnd = rng.choice(pool_idx, N_RAND, replace=False)
        for i, lab, kind in [(i, 1, 'pos') for i in pos] + [(i, 0, 'look') for i in look] + [(i, 0, 'rand') for i in rnd]:
            q_person.append(p); q_idx.append(i); q_label.append(lab); q_kind.append(kind)
    return (np.array(enrol), np.array(q_person), np.array(q_idx), np.array(q_label), np.array(q_kind), test_ids)


def boot_auc_diff(s_a, s_b, labels, person, rng, n=N_BOOT):
    """Paired bootstrap over persons of AUC(s_a) - AUC(s_b)."""
    persons = np.unique(person)
    by_p = [np.where(person == p)[0] for p in persons]
    out = []
    for _ in range(n):
        idx = np.concatenate([by_p[i] for i in rng.integers(0, len(persons), len(persons))])
        y = labels[idx]
        if y.min() == y.max():
            continue
        out.append(roc_auc_score(y, s_a[idx]) - roc_auc_score(y, s_b[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    backbones = sys.argv[1:] or ['facenet', 'dinov2']
    os.makedirs(os.path.join(RES, 'kc_' + VER), exist_ok=True)
    summary = {}
    for bb in backbones:
        t0 = time.time()
        E, pid, split = load(bb)
        rng = np.random.default_rng(SEED)
        enrol, qp, qi, ql, qk, test_ids = build_task(E, pid, split, rng)
        # --- cosine ceiling ---
        cos_q = np.einsum('qd,qkd->qk', E[qi], E[enrol[qp]]).max(1)
        res = {'n_images': int(len(E)), 'n_test_persons': int(len(test_ids)), 'K': K_ENROL}
        for neg in ['look', 'rand']:
            m = (qk == 'pos') | (qk == neg)
            res[f'auc_cos_pos_vs_{neg}'] = float(roc_auc_score(ql[m], cos_q[m]))
        res['cos_quantiles'] = {k: [float(x) for x in np.percentile(cos_q[qk == k], [5, 25, 50, 75, 95])] for k in ['pos', 'look', 'rand']}
        protos = E[enrol].mean(1)
        protos /= np.linalg.norm(protos, axis=1, keepdims=True)
        pos_sim = np.array([E[qi[j]] @ protos[qp[j]] for j in np.where(qk == 'pos')[0]])
        top_look = np.array([max(E[qi[j]] @ protos[qp[j]] for j in np.where((qk == 'look') & (qp == p))[0]) for p in range(len(test_ids))])
        thr = np.percentile(pos_sim, 5)
        res['O_share_persons_top1_lookalike_above_p5_pos'] = float(np.mean(top_look > thr))
        hist_bins = np.linspace(-0.2, 1.0, 61)
        res['hist'] = {'bins': hist_bins.tolist(), **{k: np.histogram(cos_q[qk == k], hist_bins)[0].tolist() for k in ['pos', 'look', 'rand']}}
        print(f"[{bb}] ceiling AUC pos-vs-lookalike {res['auc_cos_pos_vs_look']:.4f}  pos-vs-random {res['auc_cos_pos_vs_rand']:.4f}  O={res['O_share_persons_top1_lookalike_above_p5_pos']:.3f}", flush=True)
        # --- KC codes (P1, P2) ---
        m_look = (qk == 'pos') | (qk == 'look')
        need = np.unique(np.concatenate([enrol.ravel(), qi]))
        pos_of = {v: i for i, v in enumerate(need)}
        Xn = E[need]
        for kind in ['gaussian', 'sparse']:
            for rot in ([False] if kind == 'gaussian' else [False, True]) + ([True] if kind == 'gaussian' else []):
                aucs = []
                for draw in range(N_DRAWS):
                    tag = f'{bb}_{kind}_rot{int(rot)}_d{draw}'
                    path = os.path.join(RES, 'kc_' + VER, tag + '.json')
                    if os.path.exists(path):
                        aucs.append(json.load(open(path))['auc_pos_vs_look'])
                        continue
                    r = np.random.default_rng(SEED + 1000 * draw + (7 if rot else 0) + (13 if kind == 'sparse' else 0))
                    X = Xn @ random_rotation(Xn.shape[1], r).T if rot else Xn
                    W = make_W(kind, M_KC, X.shape[1], r)
                    H = codes(W, X, FRAC)
                    k = H[0].sum()
                    hq = H[[pos_of[i] for i in qi]]
                    he = H[[[pos_of[i] for i in row] for row in enrol[qp]]]
                    kc_q = (hq[:, None, :] & he).sum(2).max(1) / k
                    auc = float(roc_auc_score(ql[m_look], kc_q[m_look]))
                    lo, hi = boot_auc_diff(kc_q[m_look], cos_q[m_look], ql[m_look], qp[m_look], r, 300)
                    json.dump({'auc_pos_vs_look': auc, 'delta_vs_ceiling': auc - res['auc_cos_pos_vs_look'],
                               'delta_ci95': [lo, hi]}, open(path + '.tmp', 'w'))
                    os.replace(path + '.tmp', path)
                    aucs.append(auc)
                    print(f'  {tag}: AUC {auc:.4f}  delta {auc - res["auc_cos_pos_vs_look"]:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]', flush=True)
                res[f'kc_{kind}_rot{int(rot)}'] = {'auc_mean': float(np.mean(aucs)), 'auc_sd': float(np.std(aucs)),
                                                    'delta_mean_vs_ceiling': float(np.mean(aucs) - res['auc_cos_pos_vs_look'])}
        res['seconds'] = round(time.time() - t0)
        summary[bb] = res
        json.dump(summary, open(os.path.join(RES, f'ceiling_kc_summary_{VER}.json'), 'w'), indent=1)
    print(json.dumps({bb: {k: v for k, v in r.items() if k not in ('hist', 'cos_quantiles')} for bb, r in summary.items()}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
