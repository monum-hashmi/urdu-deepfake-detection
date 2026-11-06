"""Frozen Whisper encoder features (the model family behind the best result of Owais et al., IEEE Access 2026).
Same recipe as extract_ssl.py: centre 4 s window -> every encoder layer -> mean over the frames that
cover the real audio (Whisper pads input to 30 s; those padded frames are excluded) -> [layers, dim].

Run:  python extract_whisper.py                     (openai/whisper-small, ~1 GB download first time)
Then: python train_ssl.py --seed 0 --tag whisper
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

from common import SEG, SR, fix_len, load_prep, read_split, split_names
from extract_ssl import feat_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--tag", default="whisper")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--clean-only", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    enc = WhisperModel.from_pretrained(args.model, dtype=dtype).get_encoder().to(device).eval()
    # encoder frame rate: 100 mel frames/s, conv stride 2 -> 50 frames/s
    n_keep = int(SEG / SR * 50)
    print(f"Model {args.model} on {device}; pooling first {n_keep} encoder frames (4 s)")

    names = split_names()
    splits = {n: read_split(n) for n in names}
    clean = pd.concat(splits.values()).path.unique().tolist()
    augd = pd.concat([splits["train"], splits["dev"]]).path.unique().tolist()
    jobs = [(p, False) for p in clean] + ([] if args.clean_only else [(p, True) for p in augd])
    jobs = [j for j in jobs if not os.path.exists(feat_path(j[0], args.tag, j[1]))]
    print(f"To extract: {len(jobs)} clips")

    t0 = time.time()
    for b in range(0, len(jobs), args.bs):
        batch = jobs[b:b + args.bs]
        wavs = [fix_len(load_prep(p, a)) for p, a in batch]
        inp = fe(wavs, sampling_rate=SR, return_tensors="pt").input_features.to(device=device, dtype=dtype)
        with torch.no_grad():
            hs = enc(inp, output_hidden_states=True).hidden_states          # tuple of [B, 1500, D]
        feats = torch.stack(hs, dim=1)[:, :, :n_keep].mean(dim=2).float().cpu().numpy()  # [B, L, D]
        for (p, a), f in zip(batch, feats):
            out = feat_path(p, args.tag, a)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tmp = out[:-4] + ".tmp.npy"
            np.save(tmp, f.astype(np.float16))
            os.replace(tmp, out)
        done = b + len(batch)
        if done % (args.bs * 25) < args.bs or done == len(jobs):
            el = time.time() - t0
            print(f"{done}/{len(jobs)}  elapsed {el/60:.1f} min  ETA {el/done*(len(jobs)-done)/60:.1f} min")
    print("Done")


if __name__ == "__main__":
    main()
