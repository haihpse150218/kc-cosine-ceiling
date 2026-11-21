"""Embed the CelebA images in splits.json with two frozen backbones (08_plan.md, T1 days 3-4).

  facenet : InceptionResnetV1 (VGGFace2 weights), 512-d, input 160x160. CelebA 'img_align_celeba' faces are
            already aligned; MTCNN is tried first and, if it finds no face, we fall back to a centre crop.
            The MTCNN failure rate is recorded (V12).
  dinov2  : ViT-S/14 (Apache-2.0), CLS token, 384-d, input 224x224, ImageNet normalisation.

Resumable: images are processed in shards of SHARD images; a shard is written atomically
(data/emb/<backbone>/shard_XXXX.npz) and skipped on restart, so a power cut loses at most one shard.
Usage: .venv/Scripts/python.exe pilot/embed.py facenet|dinov2 [--limit N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..')
DATA = os.path.join(BASE, 'data', 'celeba')
WEIGHTS = os.path.join(BASE, 'weights')
SHARD = int(os.environ.get("SHARD", 2000))


def image_list():
    with open(os.path.join(DATA, 'splits.json')) as f:
        sp = json.load(f)['splits']
    items = []
    for split, ids in sp.items():
        for pid, imgs in ids.items():
            items += [(img, int(pid), split) for img in imgs]
    items.sort()
    return items


def img_root():
    for cand in ['img_align_celeba', os.path.join('img_align_celeba', 'img_align_celeba')]:
        p = os.path.join(DATA, cand)
        if os.path.isdir(p):
            return p
    sys.exit('img_align_celeba/ not found under data/celeba — unzip img_align_celeba.zip there')


def build(backbone, dev):
    if backbone == 'facenet':
        from facenet_pytorch import InceptionResnetV1, MTCNN
        net = InceptionResnetV1(pretrained=None, classify=False)
        sd = torch.load(os.path.join(WEIGHTS, '20180402-114759-vggface2.pt'), map_location='cpu')
        sd = {k: v for k, v in sd.items() if not k.startswith('logits')}
        net.load_state_dict(sd)
        mtcnn = MTCNN(image_size=160, margin=14, post_process=True, device=dev)

        def prep(im):
            face = mtcnn(im)
            if face is not None:
                return face, True
            w, h = im.size
            s = min(w, h)
            crop = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((160, 160))
            t = torch.from_numpy(np.asarray(crop, dtype=np.float32)).permute(2, 0, 1)
            return (t - 127.5) / 128.0, False
        return net.eval().to(dev), prep
    if backbone == 'dinov2':
        # pinned commit; the embeddings of 15/11/2026 were made from a 'main' cache verified byte-identical
        # to this commit (30 model/layer/hub files compared, PILOT_LOG 15/11)
        net = torch.hub.load('facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8', 'dinov2_vits14',
                             pretrained=False, trust_repo=True)
        net.load_state_dict(torch.load(os.path.join(WEIGHTS, 'dinov2_vits14_pretrain.pth'), map_location='cpu'))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        def prep(im):
            t = torch.from_numpy(np.asarray(im.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0)
            return (t.permute(2, 0, 1) - mean) / std, True
        return net.eval().to(dev), prep
    raise ValueError(backbone)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('backbone', choices=['facenet', 'dinov2'])
    ap.add_argument('--limit', type=int, default=0, help='only the first N images (smoke test)')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    items = image_list()
    if a.limit:
        items = items[:a.limit]
    root = img_root()
    out = os.path.join(BASE, 'data', 'emb', a.backbone + ('_smoke' if a.limit else ''))
    os.makedirs(out, exist_ok=True)
    net, prep = build(a.backbone, dev)
    n_sh = (len(items) + SHARD - 1) // SHARD
    t0 = time.time()
    for s in range(n_sh):
        path = os.path.join(out, f'shard_{s:04d}.npz')
        if os.path.exists(path):
            continue
        chunk = items[s * SHARD:(s + 1) * SHARD]
        embs, ok = [], []
        batch = []
        for img, pid, split in chunk:
            x, good = prep(Image.open(os.path.join(root, img)).convert('RGB'))
            batch.append(x)
            ok.append(good)
            if len(batch) == 64:
                with torch.no_grad():
                    embs.append(net(torch.stack(batch).to(dev)).float().cpu().numpy())
                batch = []
        if batch:
            with torch.no_grad():
                embs.append(net(torch.stack(batch).to(dev)).float().cpu().numpy())
        E = np.concatenate(embs)
        tmp = path + '.tmp.npz'
        np.savez(tmp, emb=E, img=np.array([c[0] for c in chunk]), pid=np.array([c[1] for c in chunk]),
                 split=np.array([c[2] for c in chunk]), detected=np.array(ok))
        os.replace(tmp, path)
        print(f'{a.backbone} shard {s + 1}/{n_sh}  {len(chunk)} imgs  detected {np.mean(ok):.3f}  '
              f'{time.time() - t0:.0f}s', flush=True)
    print('done', a.backbone, len(items), 'images')


if __name__ == '__main__':
    sys.exit(main())
