"""Figure 1 of the Letter from pilot results (no new computation on images)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(H, '..', 'pilot', 'results')
k = json.load(open(os.path.join(R, 'kappa_rot.json')))
s = json.load(open(os.path.join(R, 'ceiling_kc_summary_v2.json')))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
g = k['kappa_gaussian']
c = np.array([r['cos'] for r in g]); m = np.array([r['mean'] for r in g]); sd = np.array([r['sd'] for r in g])
ap = np.array([r['approx_large_m'] for r in g])
o = np.argsort(c)
ax[0].plot(c[o], ap[o], 'k-', lw=1.2, label=r'closed form $\kappa(c)$')
ax[0].errorbar(c, m, yerr=sd, fmt='o', ms=3, color='C0', elinewidth=0.6, capsize=0, label='Monte-Carlo (Gaussian $W$)')
for bb, mk in [('facenet', 's'), ('dinov2', '^')]:
    for kind, col in [('gaussian', 'C1'), ('sparse', 'C2')]:
        for pair in ['same', 'stranger']:
            r = k['real'][bb][f'{kind}_{pair}']
            ax[0].plot(r['mean_cos'], r['mean_overlap'], mk, color=col, ms=5, mfc='none',
                       label=f'faces, {kind} $W$' if (bb == 'facenet' and pair == 'same') else None)
ax[0].set_xlabel('cosine $c$'); ax[0].set_ylabel('expected code overlap'); ax[0].set_title('(a)', loc='left', fontsize=9)
ax[0].legend(fontsize=6.5, frameon=False)
h = s['facenet']['hist']; b = np.array(h['bins']); ctr = (b[:-1] + b[1:]) / 2
for key, lab, col in [('pos', 'same person', 'C0'), ('look', 'look-alike stranger', 'C3'), ('rand', 'random stranger', '0.6')]:
    y = np.array(h[key], float); y /= y.sum() * (b[1] - b[0])
    ax[1].plot(ctr, y[:len(ctr)], color=col, lw=1.2, label=lab)
ax[1].set_xlabel('max cosine to enrolled images (facenet)'); ax[1].set_ylabel('density'); ax[1].set_title('(b)', loc='left', fontsize=9)
ax[1].legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(H, 'fig1.pdf')); fig.savefig(os.path.join(H, 'fig1.png'), dpi=150)
print('ok', len(c), 'cosines; max |MC - closed form| =', round(float(np.max(np.abs(m - ap))), 4))
