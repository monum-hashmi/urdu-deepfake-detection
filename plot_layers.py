"""Figure: which layers of the frozen front end the classifier relies on (learned softmax layer weights).
Run:  python plot_layers.py      -> results/fig_layers.pdf and .png
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

groups = {}
for f in sorted(glob.glob(os.path.join("results", "layers_*.csv"))):
    run = os.path.basename(f)[len("layers_"):-4]
    if "_tr" in run:                       # skip leave-one-generator-out runs
        continue
    groups.setdefault(re.sub(r"_s\d+$", "", run), []).append(pd.read_csv(f).weight.values)
if not groups:
    raise SystemExit("No results/layers_*.csv found.")

names = [g for g in sorted(groups) if g.endswith("_noaug")] or sorted(groups)
fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 2.2), squeeze=False)
for ax, g in zip(axes[0], names):
    df = pd.DataFrame(groups[g])
    m, s = df.mean(axis=0), df.std(axis=0).fillna(0)
    ax.bar(range(len(m)), m, yerr=s, color="#2a6f97", ecolor="#555555", capsize=1.5, width=0.8)
    ax.axhline(1 / len(m), color="#c44536", lw=0.8, ls="--", label="uniform")
    label = g.replace("ssl-", "").replace("_noaug", "").replace("xlsr", "XLS-R").replace("whisper", "Whisper-small")
    ax.set_title(f"{label} (n={len(df)} seeds)", fontsize=8)
    ax.set_xlabel("Layer (0 = CNN/embedding output)", fontsize=7)
    ax.set_ylabel("Learned weight", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, frameon=False)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/fig_layers.pdf")
fig.savefig("results/fig_layers.png", dpi=200)
print("Saved results/fig_layers.pdf and .png")
for g in names:
    m = pd.DataFrame(groups[g]).mean(axis=0)
    top = m.sort_values(ascending=False).head(3)
    print(f"{g}: top layers {list(top.index)} with weights {[round(float(v), 3) for v in top.values]}")
