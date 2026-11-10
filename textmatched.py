"""Text-matched out-of-domain test: genuine FLEURS speech vs. TTS of the SAME FLEURS sentences
(new-text fakes from round3.py gen). Text is matched across classes; speakers, recording conditions,
text and generators are all unseen in training.

Run:  python textmatched.py     -> results/round3_textmatched.csv (per run) + printed means
"""
import glob
import re

import numpy as np
import pandas as pd

from common import eer

rows = []
for f in sorted(glob.glob("results/round3_scores_*.csv")):
    run = f[len("results/round3_scores_"):-4]
    try:
        ood = pd.read_csv(f"results/scores_ood_{run}.csv")
    except FileNotFoundError:
        continue
    fl = ood[ood.y == 1]
    nt = pd.read_csv(f)
    nt = nt[(nt.condition == "newtext") & (nt.y == 0)]
    r = dict(run=run, setting=re.sub(r"_s\d+$", "", run),
             pooled=eer(np.r_[fl.score, nt.score], np.r_[fl.y, nt.y]))
    for g, s in nt.groupby("generator"):
        r[g] = eer(np.r_[fl.score, s.score], np.r_[fl.y, s.y])
    rows.append(r)
df = pd.DataFrame(rows)
df.round(3).to_csv("results/round3_textmatched.csv", index=False)
print(f"{len(fl)} FLEURS real vs {len(nt)} text-matched fakes")
print(df.drop(columns="run").groupby("setting").agg(["mean", "std"]).round(2).to_string())
