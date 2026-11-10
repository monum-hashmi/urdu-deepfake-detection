"""Review round 3 analyses from the released score files (CPU only, no retraining).

  A. Interval for the reported mean-over-seeds EER: cluster bootstrap in which each draw recomputes the
     EER of every seed on the same resampled clusters and averages them                      [note 21]
  B. Same, with clusters keyed by (speaker, corpus part, utterance number)                      [20]
  C. Per test speaker at the POOLED threshold: bona fide rejected and seen fakes accepted        [24]
  D. AASIST-L (ASVspoof weights): score distributions and an orientation check                   [23]
  E. No-padding vs default: clustered bootstrap interval of the EER difference                   [30]
  F. Balance of the 22 prompts lost to the filename collision (part, speaker, length)           [9]
  G. Per-seed EER of the key ablations                                                           [17, 31]

Run:  python review3_analysis.py   -> results/review3_*.csv, results/review3_report.txt, results/fig_aasist_scores.pdf
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from review2_analysis import clean_unseen, eer, eer_thr, seen, setting_of

N = 2000
TEST = ["Speaker_03", "Speaker_07", "Speaker_11", "Speaker_15"]


def part_map():
    t = pd.read_csv("meta/texts.csv", encoding="utf-8-sig")
    t["part"] = t.src_path.str.extract(r"Part (\d)")[0]
    t = t[~(t.speaker + "|" + t.utt.astype(str)).duplicated(keep="first")]
    return {(s, str(u)): p for s, u, p in zip(t.speaker, t.utt, t.part)}


def keys(df, with_part, pm):
    out = []
    for p, spk in zip(df.path, df.speaker):
        n = str(p).replace("\\", "/")
        stem = os.path.splitext(os.path.basename(n))[0]
        if "/new/" in n:                                         # generated: Speaker_XX_<utt>
            s, u = stem.rsplit("_", 1)
            part = pm.get((s, u), "?")
        else:                                                   # corpus: .../Speaker_XX/Part k/<utt>.wav
            s, u = spk, stem
            m = re.search(r"Part (\d)", n)
            part = m.group(1) if m else "?"
        out.append(f"{s}|{part}|{u}" if with_part else f"{s}|{u}")
    return np.array(out)


def load_setting(st):
    fs = sorted(glob.glob(f"results/scores_{st}_s[0-9].csv"))
    return [pd.read_csv(f) for f in fs]


def mean_eer_ci(dfs, sel, with_part, pm, rng):
    ds = [sel(d) for d in dfs]
    base = ds[0]
    k = keys(base, with_part, pm)
    # all seeds share the same rows in the same order (same split files)
    for d in ds[1:]:
        assert (d.path.values == base.path.values).all()
    groups = list(pd.Series(range(len(k))).groupby(k).indices.values())
    S = [d.score.values for d in ds]
    y = base.y.values
    point = np.mean([eer(s, y) for s in S])
    vals = []
    for _ in range(N):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(np.mean([eer(s[idx], y[idx]) for s in S]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    avg_score_eer = eer(np.mean(S, axis=0), y)
    return point, lo, hi, len(groups), avg_score_eer


def main():
    rng = np.random.default_rng(0)
    pm = part_map()
    rep = []

    # A + B
    rows = []
    for st in ["ssl-xlsr_noaug", "ssl-whisper_noaug", "lcnn_noaug", "aasistL_noaug", "ssl-xlsr_noaug_layer5",
               "ssl-xlsr_noaug_uniform", "ssl-xlsrnopad_noaug", "ssl-xlsr_aug", "lcnn_aug", "aasistL_aug",
               "ssl-whisper_aug"]:
        dfs = load_setting(st)
        if not dfs:
            continue
        for tname, sel in [("seen", seen), ("unseen", clean_unseen)]:
            for wp in [False, True]:
                p, lo, hi, ng, ae = mean_eer_ci(dfs, sel, wp, pm, rng)
                rows.append(dict(setting=st, test=tname, cluster="speaker|part|utt" if wp else "speaker|utt",
                                 n_clusters=ng, mean_eer=p, ci_low=lo, ci_high=hi, eer_of_seed_avg_score=ae))
    a = pd.DataFrame(rows).round(2)
    a.to_csv("results/review3_mean_eer_ci.csv", index=False)
    rep.append("=== A/B. 95% interval for the MEAN-over-seeds EER (each draw: same clusters, EER per seed, "
               "averaged). Also the EER of the seed-averaged score. ===\n" + a.to_string(index=False))

    # C. per speaker at pooled threshold
    rows = []
    for st in ["ssl-xlsr_noaug", "ssl-whisper_noaug", "lcnn_noaug", "aasistL_noaug", "aasistL-pretrained"]:
        for d0 in load_setting(st):
            for tname, sel in [("seen", seen), ("unseen", clean_unseen)]:
                d = sel(d0)
                _, t = eer_thr(d.score.values, d.y.values)
                for spk in TEST:
                    real = d[(d.y == 1) & (d.speaker == spk)]
                    row = dict(setting=st, test=tname, speaker=spk,
                               bona_rejected=(real.score < t).mean() * 100)
                    if tname == "seen":
                        fk = d[(d.y == 0) & (d.speaker == spk)]
                        row["seen_fakes_accepted"] = (fk.score >= t).mean() * 100
                    rows.append(row)
    c = pd.DataFrame(rows).groupby(["setting", "test", "speaker"]).mean(numeric_only=True).round(2)
    c.to_csv("results/review3_per_speaker_common_threshold.csv")
    rep.append("\n=== C. Per test speaker at the pooled EER threshold (mean over seeds) ===\n" + c.to_string())

    # D. AASIST pretrained distributions
    d0 = load_setting("aasistL-pretrained")[0]
    rows = []
    for tname, sel in [("seen", seen), ("unseen", clean_unseen)]:
        d = sel(d0)
        auc = roc_auc_score(d.y, d.score) * 100
        rows.append(dict(test=tname, eer=eer(d.score, d.y), eer_flipped=eer(-d.score, d.y), auc=auc,
                         median_real=d[d.y == 1].score.median(), median_fake=d[d.y == 0].score.median()))
        for g, s in d[d.y == 0].groupby("generator"):
            rows.append(dict(test=tname, generator=g, median_fake=s.score.median(),
                             frac_fake_above_median_real=(s.score > d[d.y == 1].score.median()).mean() * 100))
    dd = pd.DataFrame(rows).round(2)
    dd.to_csv("results/review3_aasist_pretrained_orientation.csv", index=False)
    rep.append("\n=== D. AASIST-L (ASVspoof weights): orientation check ===\n" + dd.to_string(index=False))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 1.9), sharey=False)
        for ax, (tname, sel) in zip(axes, [("Seen test set", seen), ("Unseen test set", clean_unseen)]):
            d = sel(d0)
            bins = np.linspace(d.score.min(), d.score.max(), 50)
            ax.hist(d[d.y == 1].score, bins=bins, density=True, histtype="step", lw=1.2, color="#222222",
                    label="bona fide")
            ax.hist(d[d.y == 0].score, bins=bins, density=True, histtype="stepfilled", alpha=0.45,
                    color="#4a6fa5", label="spoof")
            ax.set_title(tname)
            ax.set_xlabel("score (higher = more bona fide)")
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
        axes[0].set_ylabel("density")
        axes[0].legend(frameon=False)
        fig.tight_layout(pad=0.3)
        fig.savefig("results/fig_aasist_scores.pdf")
        fig.savefig("results/fig_aasist_scores.png", dpi=200)
    except ImportError:
        pass

    # E. no-padding vs default, clustered CI of the mean-EER difference
    rows = []
    A_, B_ = load_setting("ssl-xlsrnopad_noaug"), load_setting("ssl-xlsr_noaug")
    for tname, sel in [("seen", seen), ("unseen", clean_unseen)]:
        da, db = [sel(d) for d in A_], [sel(d) for d in B_]
        idx_b = {p: i for i, p in enumerate(db[0].path)}
        order = np.array([idx_b[p] for p in da[0].path])
        y = da[0].y.values
        Sa, Sb = [d.score.values for d in da], [d.score.values[order] for d in db]
        k = keys(da[0], True, pm)
        groups = list(pd.Series(range(len(k))).groupby(k).indices.values())
        diffs = []
        for _ in range(N):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            diffs.append(np.mean([eer(s[idx], y[idx]) for s in Sa]) - np.mean([eer(s[idx], y[idx]) for s in Sb]))
        point = np.mean([eer(s, y) for s in Sa]) - np.mean([eer(s, y) for s in Sb])
        rows.append(dict(test=tname, diff_nopad_minus_default=point, ci_low=np.percentile(diffs, 2.5),
                         ci_high=np.percentile(diffs, 97.5)))
    e = pd.DataFrame(rows).round(2)
    e.to_csv("results/review3_padding_difference.csv", index=False)
    rep.append("\n=== E. No padding minus default (XLS-R), mean-over-seeds EER difference, 95% interval ===\n"
               + e.to_string(index=False))

    # F. balance of the 22 lost prompts
    t = pd.read_csv("meta/texts.csv", encoding="utf-8-sig")
    t["part"] = t.src_path.str.extract(r"(Part \d)")[0]
    t["chars"] = t.text.fillna("").str.len()
    lost = (t.speaker + "|" + t.utt.astype(str)).duplicated(keep="first")
    f1 = pd.DataFrame({"kept (488)": t[~lost].part.value_counts(), "lost (22)": t[lost].part.value_counts(),
                       "all sampled (510)": t.part.value_counts()})
    f2 = pd.DataFrame({"kept": t[~lost].chars.describe(), "lost": t[lost].chars.describe()}).round(1)
    f3 = t[lost].speaker.value_counts().sort_index()
    rep.append("\n=== F. 22 lost prompts: corpus part ===\n" + f1.to_string()
               + "\n\nprompt length (characters):\n" + f2.to_string()
               + "\n\nlost per speaker:\n" + f3.to_string())

    # G. per-seed values for key ablations
    ps = pd.read_csv("results/review2_per_seed.csv") if os.path.exists("results/review2_per_seed.csv") else None
    if ps is not None:
        key = ps[ps.setting.isin(["ssl-xlsr_noaug", "ssl-xlsr_noaug_layer5", "ssl-xlsr_noaug_uniform",
                                  "ssl-xlsrnopad_noaug", "ssl-xlsr_noaug_trvits", "ssl-xlsr_noaug_trtacotron",
                                  "ssl-xlsr_aug"])]
        rep.append("\n=== G. Per-seed EER of the key ablations (3 runs each) ===\n" + key.to_string(index=False))

    open("results/review3_report.txt", "w", encoding="utf-8").write("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    main()
