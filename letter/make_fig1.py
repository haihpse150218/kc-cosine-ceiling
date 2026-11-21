"""Figure 1 of the Letter from pilot results (no new computation on images).
(a) closed form kappa(c) (Owen's T), Monte-Carlo means (kappa_rot.json), and real face pairs binned by cosine
    (pairs_v2.json: mean of per-pair overlaps within each cosine bin, so no Jensen effect).
(b) max cosine of a query to the enrolled images, facenet, v3 task (look-alikes selected with DINOv2; hist_v3.json).
"""
import json
import os

import matplotlib
import numpy as np
from scipy.special import owens_t
from scipy.stats import norm

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(H, '..', 'pilot', 'results')
k = json.load(open(os.path.join(R, 'kappa_rot.json')))
pv = json.load(open(os.path.join(R, 'pairs_v2.json')))
hv = json.load(open(os.path.join(R, 'hist_v3.json')))
P = 0.05
T = norm.ppf(1 - P)


def kappa(c):
    c = np.clip(c, -0.999999, 0.999999)
    return (1 - norm.cdf(T) - 2 * owens_t(T, np.sqrt((1 - c) / (1 + c)))) / P


fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
cc = np.linspace(-0.3, 0.995, 300)
ax[0].plot(cc, kappa(cc), 'k-', lw=1.2, label=r'closed form $\kappa(c)$')
g = k['kappa_gaussian']
c = np.array([r['cos'] for r in g]); m = np.array([r['mean'] for r in g]); sd = np.array([r['sd'] for r in g])
ax[0].errorbar(c, m, yerr=sd, fmt='o', ms=3, color='C0', elinewidth=0.6, capsize=0, label='Monte-Carlo (Gaussian $W$)')
for bb, mk, name, dx in [('facenet', 's', 'facenet', -0.012), ('dinov2', '^', 'DINOv2', 0.012)]:
    for kind, col, fill, kn in [('gaussian', 'C1', True, 'Gaussian'), ('sparse', 'C2', False, 'sparse')]:
        rows = pv[bb]['binned'][kind]
        ax[0].plot([r['cos_mid'] + dx for r in rows], [r['mean_overlap'] for r in rows], mk, color=col, ms=4,
                   mfc=col if fill else 'none', label=f'{name}, {kn} $W$')
ax[0].set_xlabel('cosine $c$'); ax[0].set_ylabel('expected code overlap'); ax[0].set_title('(a)', loc='left', fontsize=9)
ax[0].legend(fontsize=6, frameon=False, ncol=1)
b = np.array(hv['bins']); ctr = (b[:-1] + b[1:]) / 2
for key, lab, col in [('pos', 'same person', 'C0'), ('look', 'look-alike stranger', 'C3'), ('rand', 'random stranger', '0.6')]:
    y = np.array(hv[key], float); y /= y.sum() * (b[1] - b[0])
    ax[1].plot(ctr, y, color=col, lw=1.2, label=lab)
ax[1].set_xlabel('max cosine to enrolled images (facenet)'); ax[1].set_ylabel('density'); ax[1].set_title('(b)', loc='left', fontsize=9)
ax[1].legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(H, 'fig1.pdf')); fig.savefig(os.path.join(H, 'fig1.png'), dpi=150)
print('ok; max |MC - closed form| =', round(float(np.max(np.abs(m - kappa(c)))), 4))
