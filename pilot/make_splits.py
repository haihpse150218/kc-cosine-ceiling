"""Count images per identity in CelebA and make identity-disjoint splits (08_plan.md, T1 day 2).

Input : data/celeba/identity_CelebA.txt   (lines "000001.jpg 2880")
Output: data/celeba/splits.json            (LOCAL ONLY — per-image identity indices may not be redistributed,
                                            see LICENSES.md; data/ is git-ignored)
        pilot/results/splits_summary.json  (aggregate counts only; safe to commit)
Split : identities with >= MIN_IMGS images -> dev / test / stranger pool (identity-disjoint), seeded.
"""
import collections
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'celeba')
RES = os.path.join(HERE, 'results')
MIN_IMGS, SEED = 15, 20261116
FRAC = {'dev': 0.2, 'test': 0.4, 'pool': 0.4}
MD5_IDENTITY = '32bd1bd63d3c78cd57e08160ec5ed1e2'  # from torchvision celeba.py (LICENSES.md)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    path = os.path.join(DATA, 'identity_CelebA.txt')
    got = md5(path)
    print('identity_CelebA.txt md5', got, 'OK' if got == MD5_IDENTITY else 'MISMATCH')
    by_id = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            img, pid = line.split()
            by_id[int(pid)].append(img)
    counts = np.array([len(v) for v in by_id.values()])
    keep = sorted(pid for pid, v in by_id.items() if len(v) >= MIN_IMGS)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(keep)
    n_dev, n_test = int(FRAC['dev'] * len(perm)), int(FRAC['test'] * len(perm))
    split = {'dev': perm[:n_dev], 'test': perm[n_dev:n_dev + n_test], 'pool': perm[n_dev + n_test:]}
    out = {k: {str(int(pid)): by_id[int(pid)] for pid in v} for k, v in split.items()}
    with open(os.path.join(DATA, 'splits.json'), 'w') as f:
        json.dump({'seed': SEED, 'min_imgs': MIN_IMGS, 'splits': out}, f)
    summary = {
        'md5_identity_ok': got == MD5_IDENTITY,
        'n_images': int(counts.sum()), 'n_identities': int(len(counts)),
        'imgs_per_identity': {'min': int(counts.min()), 'median': float(np.median(counts)),
                              'mean': float(counts.mean()), 'max': int(counts.max())},
        'hist_imgs_per_identity': {str(k): int(v) for k, v in sorted(collections.Counter(counts.tolist()).items())},
        'min_imgs': MIN_IMGS, 'seed': SEED, 'n_identities_kept': len(keep),
        'n_per_split': {k: int(len(v)) for k, v in split.items()},
        'n_images_per_split': {k: int(sum(len(by_id[int(p)]) for p in v)) for k, v in split.items()},
    }
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, 'splits_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ['n_images', 'n_identities', 'imgs_per_identity', 'n_identities_kept', 'n_per_split', 'n_images_per_split']}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
