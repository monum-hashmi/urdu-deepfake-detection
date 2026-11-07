"""Operating point fixed on the DEVELOPMENT set (review round 2, notes 9, 41, 55, 94).

For every checkpoint: score the dev split, take the threshold that gives EER on dev, and apply that
fixed threshold to the seen test set, the clean unseen test set and FLEURS.

Run (needs the cached features / preprocessed audio, i.e. on the GPU machine):
      python score_dev.py               -> results/review2_dev_threshold.csv (+ summary printed)
      python score_dev.py --only ssl    (skip LCNN / AASIST, ~1 min)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from common import read_split
from score_extra import score_df


def eer_thr(s, y):
    fpr, tpr, thr = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2 * 100, thr[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "ssl"], default="all")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = read_split("dev")
    rows = []
    for ckpt in sorted(glob.glob(os.path.join("checkpoints", "*.pt"))):
        run = os.path.splitext(os.path.basename(ckpt))[0]
        if args.only == "ssl" and not run.startswith("ssl-"):
            continue
        f_test, f_ood = f"results/scores_{run}.csv", f"results/scores_ood_{run}.csv"
        if not (os.path.exists(f_test) and os.path.exists(f_ood)):
            continue
        s = score_df(run, ckpt, dev, device)
        if s is None:
            continue
        dev_eer, t = eer_thr(np.asarray(s), dev.y.values)
        test, ood = pd.read_csv(f_test), pd.read_csv(f_ood)
        seen = test[test.condition == "seen_generators"]
        uns = test[(test.condition == "unseen_generators") & (test.generator != "real_mp3")]
        fleurs = ood[ood.y == 1]
        rows.append(dict(
            run=run, dev_eer=dev_eer,
            fa_seen_real=(seen[seen.y == 1].score < t).mean() * 100,          # bona fide test speech rejected
            miss_seen_fake=(seen[seen.y == 0].score >= t).mean() * 100,       # seen fakes accepted
            miss_unseen_fake=(uns[uns.y == 0].score >= t).mean() * 100,       # unseen fakes accepted
            fa_fleurs=(fleurs.score < t).mean() * 100))                       # FLEURS speech rejected
        print(f"{run}: dev EER {dev_eer:.2f}  FA seen {rows[-1]['fa_seen_real']:.2f}  "
              f"FA FLEURS {rows[-1]['fa_fleurs']:.2f}  miss unseen {rows[-1]['miss_unseen_fake']:.2f}")
    df = pd.DataFrame(rows)
    df.round(2).to_csv("results/review2_dev_threshold.csv", index=False)
    df["setting"] = df.run.str.replace(r"_s\d+$", "", regex=True)
    print("\n=== Threshold fixed on dev: mean over seeds (%) ===")
    print(df.groupby("setting")[["dev_eer", "fa_seen_real", "fa_fleurs", "miss_seen_fake", "miss_unseen_fake"]]
          .agg(lambda x: f"{x.mean():.2f} ± {x.std():.2f}" if len(x) > 1 else f"{x.mean():.2f}").to_string())


if __name__ == "__main__":
    main()
