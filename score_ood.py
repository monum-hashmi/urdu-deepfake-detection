"""Score every trained checkpoint on the out-of-domain real-speech control. No retraining.

For each run it reports:
  * fa_ood      : % of genuine FLEURS clips flagged FAKE, at the threshold that gives EER on
                  the in-domain test set (eval_seen). This is the key validity number.
  * fa_indomain : same threshold, % of in-domain real test clips flagged fake (~= EER by design)
  * eer_ood     : EER with FLEURS as the only real class, vs each spoof generator and ALL
Run:  python score_ood.py      -> results/ood_<run>.csv and results/ood_summary.csv
"""
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from common import eer, read_split


def eer_threshold(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return thr[i]


def score_run(ckpt, device):
    run = os.path.splitext(os.path.basename(ckpt))[0]
    seen, ood = read_split("eval_seen"), read_split("eval_ood")
    if run.startswith("ssl-"):
        from train_ssl import LayerMLP, load_feats, scores
        tag = run.split("_")[0][len("ssl-"):]
        Xs, Xo = load_feats(seen, tag), load_feats(ood, tag)
        model = LayerMLP(Xs.shape[1], Xs.shape[2]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        s_seen, s_ood = scores(model, Xs, device), scores(model, Xo, device)
    elif run.startswith("lcnn"):
        from train_lcnn import LCNN, score
        model = LCNN().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        s_seen, s_ood = score(model, seen, device, 32), score(model, ood, device, 32)
    elif run.startswith("aasist"):
        from train_aasist import build_model, score
        model = build_model().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        s_seen, s_ood = score(model, seen, device), score(model, ood, device)
    else:
        return None

    thr = eer_threshold(s_seen, seen.y)
    ood = ood.assign(score=s_ood)
    bona = ood[ood.y == 1]
    rows = [dict(run=run, metric="fa_ood", generator="real_fleurs",
                 value=float((bona.score < thr).mean() * 100)),
            dict(run=run, metric="fa_indomain", generator="real",
                 value=float((s_seen[seen.y.values == 1] < thr).mean() * 100)),
            dict(run=run, metric="eer_ood", generator="ALL", value=eer(ood.score, ood.y))]
    for g, s in ood[ood.y == 0].groupby("generator"):
        sub = pd.concat([bona, s])
        rows.append(dict(run=run, metric="eer_ood", generator=g, value=eer(sub.score, sub.y)))
    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv(os.path.join("results", f"ood_{run}.csv"), index=False)
    ood.to_csv(os.path.join("results", f"scores_ood_{run}.csv"), index=False)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(glob.glob(os.path.join("checkpoints", "*.pt")))
    if not ckpts:
        raise SystemExit("No checkpoints found.")
    parts = []
    for c in ckpts:
        print("Scoring", os.path.basename(c))
        r = score_run(c, device)
        if r is not None:
            parts.append(r)
    df = pd.concat(parts)
    df["setting"] = df.run.str.replace(r"_s\d+$", "", regex=True)
    summ = df.groupby(["setting", "metric", "generator"]).value.agg(["mean", "std", "count"]).reset_index().round(2)
    summ.to_csv("results/ood_summary.csv", index=False)
    summ["val"] = summ.apply(lambda r: f"{r['mean']:.2f} ± {0 if pd.isna(r['std']) else r['std']:.2f} (n={int(r['count'])})", axis=1)
    print("\n=== Out-of-domain real speech control (FLEURS Urdu) ===")
    print(summ.pivot_table(index=["metric", "generator"], columns="setting", values="val", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
