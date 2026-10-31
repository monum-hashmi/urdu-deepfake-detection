"""Baseline: LFCC front-end + LCNN back-end (ASVspoof 2019 official baseline design).

Run:  python train_lcnn.py --seed 0            (no augmentation)
      python train_lcnn.py --seed 0 --aug      (codec + noise augmentation)
Results -> results/eer_lcnn_*.csv
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import SEG, eer, fix_len, load_prep, read_split, save_results


# ------------------------------------------------------------------ front-end
def deltas(x):
    p = F.pad(x, (2, 2), mode="replicate")
    return (2 * (p[..., 4:] - p[..., :-4]) + (p[..., 3:-1] - p[..., 1:-3])) / 10


class LFCC(nn.Module):
    """20 linear-filterbank cepstra + delta + delta-delta = 60 x frames (20 ms win, 10 ms hop)."""

    def __init__(self, n_fft=512, win=320, hop=160, n_filt=20, n_ceps=20):
        super().__init__()
        self.n_fft, self.win, self.hop = n_fft, win, hop
        self.register_buffer("window", torch.hann_window(win))
        nb = n_fft // 2 + 1
        edges = torch.linspace(0, nb - 1, n_filt + 2)
        bins = torch.arange(nb).float()
        fb = torch.zeros(n_filt, nb)
        for m in range(n_filt):
            l, c, r = edges[m], edges[m + 1], edges[m + 2]
            fb[m] = torch.clamp(torch.minimum((bins - l) / (c - l), (r - bins) / (r - c)), min=0)
        self.register_buffer("fb", fb)
        n = torch.arange(n_filt).float()
        k = torch.arange(n_ceps).float()[:, None]
        dct = torch.cos(math.pi / n_filt * (n + 0.5) * k) * math.sqrt(2 / n_filt)
        dct[0] /= math.sqrt(2)
        self.register_buffer("dct", dct)

    def forward(self, x):                                    # x: [B, T]
        X = torch.stft(x, self.n_fft, self.hop, self.win, self.window,
                       return_complex=True).abs() ** 2       # [B, F, frames]
        c = self.dct @ torch.log(self.fb @ X + 1e-8)         # [B, 20, frames]
        d = deltas(c)
        return torch.cat([c, d, deltas(d)], dim=1)           # [B, 60, frames]


# ------------------------------------------------------------------ LCNN
class MFM(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


def conv(cin, cout, k, p=0):
    return nn.Sequential(nn.Conv2d(cin, 2 * cout, k, 1, p), MFM())


class LCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = LFCC()
        self.inorm = nn.BatchNorm2d(1)
        self.net = nn.Sequential(
            conv(1, 32, 5, 2), nn.MaxPool2d(2),
            conv(32, 32, 1), nn.BatchNorm2d(32), conv(32, 48, 3, 1), nn.MaxPool2d(2), nn.BatchNorm2d(48),
            conv(48, 48, 1), nn.BatchNorm2d(48), conv(48, 64, 3, 1), nn.MaxPool2d(2),
            conv(64, 64, 1), nn.BatchNorm2d(64), conv(64, 32, 3, 1), nn.BatchNorm2d(32),
            conv(32, 32, 1), nn.BatchNorm2d(32), conv(32, 32, 3, 1), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Dropout(0.7), nn.Linear(32, 160), MFM(), nn.Linear(80, 2))

    def forward(self, wav):
        x = self.inorm(self.fe(wav).unsqueeze(1))
        x = self.net(x).mean(dim=(2, 3))
        return self.head(x)


# ------------------------------------------------------------------ data
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
        if self.train and self.aug and self.rng.random() < 0.3:      # additive noise 10-40 dB SNR
            snr = self.rng.uniform(10, 40)
            p = np.mean(wav ** 2) + 1e-8
            wav = wav + self.rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(wav)).astype(np.float32)
        return torch.from_numpy(wav.astype(np.float32)), self.ys[i]


def score(model, df, device, bs):
    dl = DataLoader(WavSet(df, False, False, 0), batch_size=bs, num_workers=0)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            logits = model(x.to(device))
            out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = f"lcnn_{'aug' if args.aug else 'noaug'}_s{args.seed}"
    print(f"Run {run} on {device}")

    train, dev = read_split("train"), read_split("dev")
    model = LCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    w = torch.tensor([1.0, (train.y == 0).sum() / max(1, (train.y == 1).sum())], dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w.to(device))
    dl = DataLoader(WavSet(train, True, args.aug, args.seed), batch_size=args.bs,
                    shuffle=True, num_workers=args.workers, drop_last=True)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = os.path.join("checkpoints", f"{run}.pt")
    best, bad = 1e9, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = lossf(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        dev_eer = eer(score(model, dev, device, args.bs), dev.y)
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
    parts = []
    for name in ["eval_seen", "eval_unseen"]:
        df = read_split(name)
        df["score"] = score(model, df, device, args.bs)
        parts.append(df)
    save_results(pd.concat(parts), run)


if __name__ == "__main__":
    main()
