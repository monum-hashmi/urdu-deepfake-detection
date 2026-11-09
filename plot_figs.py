import glob, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = sys.argv[1]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
INK, MUTED, BAR = "#222222", "#777777", "#4a6fa5"

# Fig 1: learned XLS-R layer weights
w = np.stack([pd.read_csv(f).weight.values for f in sorted(glob.glob(f"{R}/results/layers_ssl-xlsr_noaug_s[0-9].csv"))])
m, s = w.mean(0), w.std(0)
L = np.arange(len(m))
fig, ax = plt.subplots(figsize=(3.4, 1.55))
ax.bar(L, m, width=0.72, color=BAR, edgecolor="white", linewidth=0.4, zorder=2)
ax.errorbar(L, m, yerr=s, fmt="none", ecolor=INK, elinewidth=0.9, capsize=1.8, capthick=0.9, zorder=3)
ax.axhline(1 / len(m), color=INK, ls=(0, (4, 2)), lw=0.9, zorder=1)
ax.text(len(m) - 0.4, 0.0605, "- - uniform (1/25)", ha="right", va="top", fontsize=7, color=INK)
ax.set_xlabel("XLS-R hidden state (0 = transformer input)")
ax.set_ylabel("Layer weight")
ax.set_xticks(L[::2]); ax.set_xlim(-0.7, len(m) - 0.3)
ax.set_ylim(0, (m + s).max() * 1.12)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig("fig_layers.pdf"); fig.savefig("fig_layers.png", dpi=300)

# Fig 2: operating points at the dev-set threshold
d = pd.read_csv(f"{R}/results/review2_dev_threshold.csv")
d["setting"] = d.run.str.replace(r"_s\d+$", "", regex=True)
g = d.groupby("setting")[["fa_fleurs", "miss_unseen_fake"]].mean()
show = [("aasistL-pretrained", "AASIST-L (ASVspoof)", "v"), ("aasistL_noaug", "AASIST-L", "^"),
        ("lcnn_noaug", "LFCC-LCNN", "s"), ("ssl-whisper_noaug", "Whisper", "D"),
        ("ssl-xlsr_noaug", "XLS-R (learned)", "o"), ("ssl-xlsr_noaug_layer5", "XLS-R layer 5", "P"),
        ("ssl-xlsr_noaug_uniform", "XLS-R uniform", "X")]
off = {"AASIST-L (ASVspoof)": (4, -3, "left"), "AASIST-L": (4, 3, "left"), "LFCC-LCNN": (4, -8, "left"),
       "Whisper": (5, 0, "left"), "XLS-R (learned)": (0, -11, "center"), "XLS-R layer 5": (6, 5, "left"),
       "XLS-R uniform": (6, 6, "left")}
fig, ax = plt.subplots(figsize=(3.4, 2.0))
for key, lab, mk in show:
    x, y = g.loc[key]
    filled = key.startswith("ssl-xlsr")
    ax.scatter(x, y, s=34, marker=mk, facecolor=BAR if filled else "white", edgecolor=INK, linewidth=0.9, zorder=3)
    dx, dy, ha = off[lab]
    ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=7, color=INK)
ax.set_xlabel("Genuine FLEURS speech rejected (%)")
ax.set_ylabel("Unseen fakes accepted (%)")
ax.set_xlim(-1, 22); ax.set_ylim(-14, 100)
ax.grid(color="#dddddd", lw=0.5, zorder=0)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig("fig_operating.pdf"); fig.savefig("fig_operating.png", dpi=300)
print(g.round(2))
