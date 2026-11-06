"""Combine all results/eer_*.csv into one table: mean ± std over seeds per model setting.
Run:  python summarize.py   -> results/summary.csv
"""
import glob
import os
import re

import pandas as pd

files = [f for f in glob.glob("results/eer_*.csv") if not os.path.basename(f).startswith("eer_extra_")]
if not files:
    raise SystemExit("No results yet.")
df = pd.concat(pd.read_csv(f) for f in files)
df["setting"] = df.run.str.replace(r"_s\d+$", "", regex=True)
g = df.groupby(["setting", "condition", "generator", "bonafide_ref"]).eer
out = g.agg(["mean", "std", "count"]).reset_index().round(2)
out.to_csv("results/summary.csv", index=False)

view = out[out.bonafide_ref == "all_real"].copy()
view["EER"] = view.apply(lambda r: f"{r['mean']:.2f} ± {0 if pd.isna(r['std']) else r['std']:.2f} (n={int(r['count'])})", axis=1)
print(view.pivot_table(index=["condition", "generator"], columns="setting", values="EER", aggfunc="first").to_string())
print("\nCodec-matched (bonafide = MP3 round-trip real only):")
cm = out[out.bonafide_ref == "real_mp3_only"]
if len(cm):
    cm = cm.assign(EER=cm["mean"].map("{:.2f}".format))
    print(cm.pivot_table(index="generator", columns="setting", values="EER", aggfunc="first").to_string())
