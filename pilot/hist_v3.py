"""Histogram of max cosine (positives / look-alikes / random strangers) for the v3 task, facenet column
(look-alikes selected with DINOv2). Aggregate counts only -> results/hist_v3.json. Used by letter/make_fig1.py."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ceiling_kc as ck  # noqa: E402
from ceiling_v3 import build  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def main():
    Ef, pid, split = ck.load('facenet')
    Ed, _, _ = ck.load('dinov2')
    enrol, qp, qi, ql, qk = build(Ed, pid, split, ck.SEED)
    mx = np.einsum('qd,qkd->qk', Ef[qi], Ef[enrol[qp]]).max(1)
    bins = np.linspace(-0.2, 1.0, 61)
    out = {'backbone': 'facenet', 'selector': 'dinov2', 'bins': bins.tolist(),
           **{k: np.histogram(mx[qk == k], bins)[0].tolist() for k in ['pos', 'look', 'rand']}}
    json.dump(out, open(os.path.join(RES, 'hist_v3.json'), 'w'))
    print({k: int(np.sum(out[k])) for k in ['pos', 'look', 'rand']})


if __name__ == '__main__':
    sys.exit(main())
