# The cosine ceiling

Code and summary results for the Letter

> P. H. Hoang, *The cosine ceiling: what random-projection Kenyon-cell codes can add to frozen embeddings* (manuscript, 2026).

A Kenyon-cell (KC) code is a random projection `W` followed by k-winners-take-all. With i.i.d. Gaussian `W` the joint law of the codes of any set of inputs depends only on their cosine matrix (Observation 1), so any readout of the codes is a garbling of that matrix and, in expectation over `W`, cannot beat the likelihood-ratio test on cosines (Corollary 1). The scripts check this on synthetic vectors and on CelebA face embeddings from two frozen backbones (facenet / VGGFace2, 512-d; DINOv2 ViT-S/14, 384-d), with Gaussian and sparse binary `W`, with and without a random rotation of the embedding.

Archived on Zenodo: [10.5281/zenodo.24412129](https://doi.org/10.5281/zenodo.24412129) (this version; all versions: [10.5281/zenodo.24411196](https://doi.org/10.5281/zenodo.24411196)).

## Layout

| Path | Content |
|---|---|
| `pilot/kc.py` | KC code: `make_W` (Gaussian or sparse binary), k-WTA `codes`, `overlap`, `random_rotation` |
| `pilot/lemma1_synthetic.py` | Observation 1 on synthetic vectors: rotation, scale and cosine-only checks (no face data needed) |
| `pilot/make_splits.py` | Identity-disjoint dev / test / stranger-pool splits of CelebA (seed 20261116, ≥ 15 images per identity), with an md5 check of `identity_CelebA.txt` |
| `pilot/embed.py` | Embeddings from the two frozen backbones; resumable shards |
| `pilot/ceiling_v3.py` | **Main experiment of the Letter (Table 1)**: look-alikes selected with the other backbone, cross-draw AUC of KC codes vs cosine, bootstrap CIs over identities, LR lower bounds, all-ones projection control, v1 selection artefact |
| `pilot/ceiling_v3_ci.py` | Table 1 uncertainty: identity bootstrap + jackknife over draws, ceiling lower bounds minus cosine, fixed-fan-in check |
| `pilot/pairs_v2.py`, `pilot/hist_v3.py` | Real-pair overlaps vs closed-form kappa (KS on independent halves, residuals, binned means) and the Fig. 1b histogram |
| `pilot/crossdraw.py`, `pilot/ceiling_kc.py` | Earlier task versions (v2 / v1), kept for the record |
| `pilot/kappa_rot.py` | Closed-form κ(c) vs Monte-Carlo, rotation KS test on real pairs, cross-fitted likelihood-ratio ceiling |
| `pilot/results/` | Summary results only (per-setting AUCs and CIs, κ table, KS statistics, split counts) and run logs |
| `pilot/LICENSES.md` | Licence review of the data, code and weights used (Vietnamese) |
| `letter/` | Manuscript sources (`main.tex`, `refs.bib`, `highlights.txt`); `make_fig1.py` draws Figure 1 from `pilot/results/` |

## Reproducing

Python 3.13 with the packages in `requirements.txt`; MiKTeX or TeX Live for the manuscript. Run every script from the repository root.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # bin/ instead of Scripts/ on Linux/macOS

# 1. No face data needed
.venv/Scripts/python pilot/lemma1_synthetic.py        # -> pilot/results/lemma1_synthetic.json
.venv/Scripts/python letter/make_fig1.py              # Figure 1 from the shipped summary results

# 2. Face experiments (needs CelebA, see below)
.venv/Scripts/python pilot/make_splits.py             # -> data/celeba/splits.json (local only)
.venv/Scripts/python pilot/embed.py facenet
.venv/Scripts/python pilot/embed.py dinov2
.venv/Scripts/python pilot/ceiling_v3.py
.venv/Scripts/python pilot/ceiling_v3_ci.py
.venv/Scripts/python pilot/pairs_v2.py
.venv/Scripts/python pilot/hist_v3.py
.venv/Scripts/python pilot/kappa_rot.py
```

### Data and weights you must fetch yourself

The CelebA agreement does not allow redistribution, so this repository contains **no** CelebA images, identity labels, per-image splits or per-image embeddings. After accepting the agreement on the [CelebA page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), put

- `data/celeba/img_align_celeba/` (aligned images) and
- `data/celeba/identity_CelebA.txt` (md5 `32bd1bd63d3c78cd57e08160ec5ed1e2`).

Weights go in `weights/`:

- `20180402-114759-vggface2.pt` from the [facenet-pytorch v2.2.9 release](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt) (ported from David Sandberg's facenet, trained on VGGFace2);
- `dinov2_vits14_pretrain.pth` from [DINOv2](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) (Apache-2.0).

## Licence

Code: MIT (see `LICENSE`). Summary results and text: CC BY 4.0. CelebA, VGGFace2 and the backbone weights keep their own terms (`pilot/LICENSES.md`).
