"""Extract frozen self-supervised (XLS-R) features once and cache them.
For each clip: centre 4 s window -> all hidden layers -> mean over time -> [layers, dim] (float16).

Run:  python extract_ssl.py                              (XLS-R 300M, ~1.2 GB download first time)
      python extract_ssl.py --model facebook/wav2vec2-base   (smaller, faster alternative)
GPU is used automatically if present. Safe to stop and re-run.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2Model

from common import fix_len, load_prep, prep_path, read_split


def feat_path(path, tag, aug=False):
    p = prep_path(path, aug).replace(os.path.join("data", "prep"), os.path.join("data", f"feat_{tag}"), 1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--tag", default="xlsr")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    model = Wav2Vec2Model.from_pretrained(args.model, dtype=dtype).to(device).eval()
    print(f"Model {args.model} on {device}")

    splits = {n: read_split(n) for n in ["train", "dev", "eval_seen", "eval_unseen", "eval_ood"]
              if os.path.exists(os.path.join("meta", f"split_{n}.csv"))}
    clean = pd.concat(splits.values()).path.unique().tolist()
    augd = pd.concat([splits["train"], splits["dev"]]).path.unique().tolist()
    jobs = [(p, False) for p in clean] + [(p, True) for p in augd]
    jobs = [j for j in jobs if not os.path.exists(feat_path(j[0], args.tag, j[1]))]
    print(f"To extract: {len(jobs)} clips")

    t0 = time.time()
    for b in range(0, len(jobs), args.bs):
        batch = jobs[b:b + args.bs]
        wavs = []
        for p, a in batch:
            w = fix_len(load_prep(p, a))
            wavs.append((w - w.mean()) / (w.std() + 1e-7))
        x = torch.from_numpy(np.stack(wavs)).to(device=device, dtype=dtype)
        with torch.no_grad():
            hs = model(x, output_hidden_states=True).hidden_states   # tuple of [B, T, D]
        feats = torch.stack(hs, dim=1).mean(dim=2).float().cpu().numpy()  # [B, L, D]
        for (p, a), f in zip(batch, feats):
            out = feat_path(p, args.tag, a)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tmp = out[:-4] + ".tmp.npy"
            np.save(tmp, f.astype(np.float16))
            os.replace(tmp, out)      # atomic: never leaves a half-written .npy
        done = b + len(batch)
        if done % (args.bs * 25) < args.bs or done == len(jobs):
            el = time.time() - t0
            print(f"{done}/{len(jobs)}  elapsed {el/60:.1f} min  ETA {el/done*(len(jobs)-done)/60:.1f} min")
    print("Done")


if __name__ == "__main__":
    main()
