"""Baseline 2: AASIST-L (Jung et al., ICASSP 2022), official implementation from github.com/clovaai/aasist.

Two settings:
  --mode pretrained   official AASIST-L weights trained on ASVspoof 2019 LA, applied to Urdu
                      with NO Urdu training (the setting of the ACL 2024 Urdu dataset paper)
  --mode train        AASIST-L trained from scratch on our speaker-disjoint Urdu train split

Setup once:  git clone https://github.com/clovaai/aasist.git
Run:         python train_aasist.py --mode pretrained
             python train_aasist.py --mode train --seed 0 [--aug]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import eer, load_prep, read_split, save_results

AASIST_DIR = "aasist"
NB_SAMP = 64600          # AASIST's native input length (~4.04 s)


def build_model():
    if not os.path.isdir(AASIST_DIR):
        raise SystemExit("Run first:  git clone https://github.com/clovaai/aasist.git")
    sys.path.insert(0, AASIST_DIR)
    from models.AASIST import Model
    with open(os.path.join(AASIST_DIR, "config", "AASIST-L.conf")) as f:
        cfg = json.load(f)["model_config"]
    return Model(cfg)


def fix_len(wav, train, rng):
    n = len(wav)
    if n >= NB_SAMP:
        start = int(rng.integers(0, n - NB_SAMP + 1)) if train else (n - NB_SAMP) // 2
        return wav[start:start + NB_SAMP]
    return np.tile(wav, int(np.ceil(NB_SAMP / n)))[:NB_SAMP]


class WavSet(Dataset):
    def __init__(self, df, train, aug, seed):
        self.paths, self.ys = df.path.tolist(), df.y.tolist()
        self.train, self.aug = train, aug
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        use_aug = self.train and self.aug and self.rng.random() < 0.5
        wav = fix_len(load_prep(self.paths[i], aug=use_aug), self.train, self.rng)
        if self.train and self.aug and self.rng.random() < 0.3:
            snr = self.rng.uniform(10, 40)
            p = np.mean(wav ** 2) + 1e-8
            wav = wav + self.rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(wav)).astype(np.float32)
        return torch.from_numpy(wav.astype(np.float32)), self.ys[i]


def score(model, df, device, bs=8):
    """Score = bonafide logit (AASIST convention: class 1 = bonafide)."""
    dl = DataLoader(WavSet(df, False, False, 0), batch_size=bs, num_workers=0)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            _, logits = model(x.to(device))
            out.append(logits[:, 1].float().cpu().numpy())
    return np.concatenate(out)


def evaluate(model, run, device):
    parts = []
    for name in ["eval_seen", "eval_unseen"]:
        df = read_split(name)
        df["score"] = score(model, df, device)
        parts.append(df)
    save_results(pd.concat(parts), run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pretrained", "train"], default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    os.makedirs("checkpoints", exist_ok=True)

    if args.mode == "pretrained":
        run = "aasistL-pretrained_s0"
        sd = torch.load(os.path.join(AASIST_DIR, "models", "weights", "AASIST-L.pth"), map_location=device)
        model.load_state_dict(sd)
        torch.save(model.state_dict(), os.path.join("checkpoints", f"{run}.pt"))
        print(f"Run {run} on {device} (ASVspoof 2019 weights, no Urdu training)")
        evaluate(model, run, device)
        return

    run = f"aasistL_{'aug' if args.aug else 'noaug'}_s{args.seed}"
    print(f"Run {run} on {device}")
    train, dev = read_split("train"), read_split("dev")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=5e-6)
    w = torch.tensor([1.0, (train.y == 0).sum() / max(1, (train.y == 1).sum())], dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w.to(device))
    dl = DataLoader(WavSet(train, True, args.aug, args.seed), batch_size=args.bs,
                    shuffle=True, num_workers=args.workers, drop_last=True)
    ckpt = os.path.join("checkpoints", f"{run}.pt")
    best, bad = 1e9, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            _, logits = model(x)
            loss = lossf(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        dev_eer = eer(score(model, dev, device), dev.y)
        print(f"ep {ep:02d} loss {tot/len(dl):.4f}  dev EER {dev_eer:.2f}%  ({time.time()-t0:.0f}s)")
        if dev_eer < best:
            best, bad = dev_eer, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stop")
                break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    evaluate(model, run, device)


if __name__ == "__main__":
    main()
