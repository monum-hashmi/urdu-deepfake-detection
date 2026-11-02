"""Proposed model: frozen XLS-R features + learned layer weighting + small MLP classifier.
Trains on cached features from extract_ssl.py, so it runs in minutes even on CPU.

Run:  python train_ssl.py --seed 0            (no augmentation)
      python train_ssl.py --seed 0 --tag whisper  (frozen Whisper-small encoder features)
      python train_ssl.py --seed 0 --train_gen vits   (leave-one-generator-out)
      python train_ssl.py --seed 0 --aug      (codec augmentation: 50% of train clips use MP3 copy)
Results -> results/eer_ssl-xlsr_*.csv  and the learned layer weights in results/layers_*.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import eer, read_split, save_results
from extract_ssl import feat_path


class LayerMLP(nn.Module):
    def __init__(self, n_layers, dim, hidden=256, drop=0.3):
        super().__init__()
        self.lw = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(hidden, 2))

    def forward(self, x):                                   # x: [B, L, D]
        w = torch.softmax(self.lw, dim=0)
        return self.mlp(self.norm((x * w[None, :, None]).sum(1)))


def load_feats(df, tag, aug=False):
    return np.stack([np.load(feat_path(p, tag, aug)).astype(np.float32) for p in df.path])


def scores(model, X, device):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 512):
            lo = model(torch.from_numpy(X[b:b + 512]).to(device))
            out.append((lo[:, 1] - lo[:, 0]).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--tag", default="xlsr")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--train_gen", choices=["vits", "tacotron"], default=None,
                    help="leave-one-generator-out: train (and select) on this spoof generator only")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lo = f"_tr{args.train_gen}" if args.train_gen else ""
    run = f"ssl-{args.tag}_{'aug' if args.aug else 'noaug'}{lo}_s{args.seed}"
    print(f"Run {run} on {device}")

    train, dev = read_split("train"), read_split("dev")
    if args.train_gen:
        keep = lambda d: d[(d.y == 1) | (d.generator == args.train_gen)].reset_index(drop=True)
        train, dev = keep(train), keep(dev)
        print(f"Leave-one-generator-out: spoof training data = {args.train_gen} only")
    Xtr = load_feats(train, args.tag)
    Xtr_aug = load_feats(train, args.tag, aug=True) if args.aug else None
    Xdev = load_feats(dev, args.tag)
    ytr = torch.tensor(train.y.values)
    model = LayerMLP(Xtr.shape[1], Xtr.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    w = torch.tensor([1.0, (train.y == 0).sum() / max(1, (train.y == 1).sum())], dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w.to(device))

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = os.path.join("checkpoints", f"{run}.pt")
    best, bad = 1e9, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        X = Xtr
        if args.aug:
            pick = rng.random(len(Xtr)) < 0.5
            X = np.where(pick[:, None, None], Xtr_aug, Xtr)
        perm = rng.permutation(len(X))
        tot = 0.0
        for b in range(0, len(perm), args.bs):
            idx = perm[b:b + args.bs]
            x = torch.from_numpy(X[idx]).to(device)
            loss = lossf(model(x), ytr[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        dev_eer = eer(scores(model, Xdev, device), dev.y)
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:02d} loss {tot/len(X):.4f}  dev EER {dev_eer:.2f}%")
        if dev_eer < best - 1e-6:
            best, bad = dev_eer, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at ep {ep}, best dev EER {best:.2f}%")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    lw = torch.softmax(model.lw.detach().cpu(), 0).numpy()
    os.makedirs("results", exist_ok=True)
    pd.DataFrame({"layer": range(len(lw)), "weight": lw}).to_csv(f"results/layers_{run}.csv", index=False)

    parts = []
    for name in ["eval_seen", "eval_unseen"]:
        df = read_split(name)
        df["score"] = scores(model, load_feats(df, args.tag), device)
        parts.append(df)
    save_results(pd.concat(parts), run)


if __name__ == "__main__":
    main()
