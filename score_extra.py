"""Score EXISTING checkpoints on the extra review splits (eval_allmp3, eval_voices) without retraining,
so the paper's main numbers stay exactly as they are.

Run (after make_extra_splits.py, prep.py and extract_ssl.py --clean-only):
      python score_extra.py                -> results/eer_extra_<run>.csv + results/extra_summary.csv
      python score_extra.py --only ssl     (skip the slower LCNN / AASIST scoring)
"""
import argparse
import glob
import os
import re

import pandas as pd
import torch

from common import eer_table, read_split

EXTRA = [n for n in ["eval_allmp3", "eval_voices"] if os.path.exists(os.path.join("meta", f"split_{n}.csv"))]


def score_df(run, ckpt, df, device):
    if run.startswith("ssl-"):
        from train_ssl import LayerMLP, load_feats, pool_from_run, scores
        pool, layer = pool_from_run(run)
        tag = run.split("_")[0][len("ssl-"):]
        X = load_feats(df, tag)
        m = LayerMLP(X.shape[1], X.shape[2], pool=pool, layer=layer).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        return scores(m, X, device)
    if run.startswith("lcnn"):
        from train_lcnn import LCNN, score
        m = LCNN().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        return score(m, df, device, 32)
    if run.startswith("aasist"):
        from train_aasist import build_model, score
        m = build_model().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        return score(m, df, device)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "ssl"], default="all")
    args = ap.parse_args()
    if not EXTRA:
        raise SystemExit("No extra splits found. Run make_extra_splits.py first.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tabs = []
    for ckpt in sorted(glob.glob(os.path.join("checkpoints", "*.pt"))):
        run = os.path.splitext(os.path.basename(ckpt))[0]
        if args.only == "ssl" and not run.startswith("ssl-"):
            continue
        if "_tr" in run or "_uniform" in run or "_layer" in run:
            continue
        parts = []
        for name in EXTRA:
            df = read_split(name)
            try:
                s = score_df(run, ckpt, df, device)
            except FileNotFoundError as e:
                print(f"Skip {run}: missing features/prep ({os.path.basename(str(e.filename))})")
                s = None
            if s is None:
                break
            parts.append(df.assign(score=s))
        if len(parts) < len(EXTRA):
            continue
        print("Scored", run)
        allp = pd.concat(parts)
        allp.to_csv(os.path.join("results", f"scores_extra_{run}.csv"), index=False)
        t = eer_table(allp, run)
        t.to_csv(os.path.join("results", f"eer_extra_{run}.csv"), index=False)
        tabs.append(t)
    df = pd.concat(tabs)
    df["setting"] = df.run.str.replace(r"_s\d+$", "", regex=True)
    out = (df[df.bonafide_ref == "all_real"].groupby(["condition", "generator", "setting"]).eer
           .agg(["mean", "std", "count"]).round(2).reset_index())
    out.to_csv("results/extra_summary.csv", index=False)
    out["val"] = out.apply(lambda r: f"{r['mean']:.2f} ± {0 if pd.isna(r['std']) else r['std']:.2f}", axis=1)
    print(out.pivot_table(index=["condition", "generator"], columns="setting", values="val", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
