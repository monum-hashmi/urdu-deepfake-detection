"""Statistical checks on the existing scores (no GPU, no retraining).

1. 95% bootstrap confidence interval of EER for every run (utterances resampled within each class).
2. Paired bootstrap test: is the reference model (default: ssl-xlsr_noaug) better than each other
   setting on the SAME test utterances? Seeds are paired by index (s0 vs s0, ...).
Run:  python bootstrap.py                      -> results/bootstrap_ci.csv, results/bootstrap_paired.csv
      python bootstrap.py --ref ssl-whisper_noaug --n 1000
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def eer_fast(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2 * 100


def strat_idx(y, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def load_runs():
    runs = {}
    for f in sorted(glob.glob(os.path.join("results", "scores_*.csv"))):
        name = os.path.basename(f)[len("scores_"):-4]
        if name.startswith(("ood_", "extra_")):
            continue
        runs[name] = pd.read_csv(f)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ref", default="ssl-xlsr_noaug")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    runs = load_runs()
    if not runs:
        raise SystemExit("No results/scores_*.csv found.")

    # ---- 1. per-run CIs
    rows = []
    for name, df in runs.items():
        for cond, d in df.groupby("condition"):
            y, s = d.y.values, d.score.values
            boots = [eer_fast(s[i], y[i]) for i in (strat_idx(y, rng) for _ in range(args.n))]
            rows.append(dict(run=name, setting=re.sub(r"_s\d+$", "", name), condition=cond,
                             eer=eer_fast(s, y), ci_low=np.percentile(boots, 2.5),
                             ci_high=np.percentile(boots, 97.5)))
    ci = pd.DataFrame(rows)
    ci.to_csv("results/bootstrap_ci.csv", index=False)
    summ = ci.groupby(["setting", "condition"])[["eer", "ci_low", "ci_high"]].mean().round(2)
    print("=== EER with 95% bootstrap CI (averaged over seeds) ===")
    print(summ.to_string())

    # ---- 2. paired test vs reference
    ref_runs = sorted(k for k in runs if re.sub(r"_s\d+$", "", k) == args.ref)
    if not ref_runs:
        print(f"\nNo runs for reference setting {args.ref}; skipping paired test.")
        return
    settings = sorted({re.sub(r"_s\d+$", "", k) for k in runs} - {args.ref})
    def arrays(run, cond):
        d = runs[run][runs[run].condition == cond].sort_values("path").reset_index(drop=True)
        return d.path.values, d.y.values, d.score.values

    prow = []
    for st in settings:
        other = sorted(k for k in runs if re.sub(r"_s\d+$", "", k) == st)
        pairs = list(zip(ref_runs, other if len(other) > 1 else other * len(ref_runs)))
        for cond in ["seen_generators", "unseen_generators"]:
            cache = []
            for r, o in pairs:
                pa, ya, sa = arrays(r, cond)
                pb, yb, sb = arrays(o, cond)
                assert len(pa) == len(pb) and (pa == pb).all(), "test sets differ"
                cache.append((ya, sa, yb, sb))
            diffs = []
            for i in range(args.n):
                idx = strat_idx(cache[0][0], np.random.default_rng(i))   # same resample for both models
                diffs.append(np.mean([eer_fast(sb[idx], yb[idx]) - eer_fast(sa[idx], ya[idx])
                                      for ya, sa, yb, sb in cache]))
            diffs = np.array(diffs)
            prow.append(dict(reference=args.ref, other=st, condition=cond,
                             mean_eer_gain=round(float(diffs.mean()), 2),
                             ci_low=round(float(np.percentile(diffs, 2.5)), 2),
                             ci_high=round(float(np.percentile(diffs, 97.5)), 2),
                             p_value=float((diffs <= 0).mean())))
    paired = pd.DataFrame(prow)
    paired.to_csv("results/bootstrap_paired.csv", index=False)
    print(f"\n=== Paired bootstrap: EER(other) - EER({args.ref}); positive = reference better ===")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
