"""Second-round review analyses, computed from the released score files (no GPU, no retraining).

  A. Per-seed EER for every setting (seen, clean unseen)                                  [48, 53, 78]
  B. Error counts at the EER threshold on the clean unseen set (incl. layer-5 audit)      [78, 83]
  C. Prompt-level clustered bootstrap: clusters = utterance NUMBER across all speakers    [59, 63]
     (more conservative than review_analysis.py, whose clusters are speaker + utterance)
  D. Prompt-source subgroup difference with a clustered CI                                [71, 72]
  E. FLEURS control: per-generator and macro-averaged EER_OOD                             [40]
  F. Per-speaker miss / false-alarm rates at the POOLED threshold                         [70]
  G. Prompt de-duplication: the discarded keys and which corpus part they came from       [30, 31]
  H. Clip counts behind the feature-extraction time                                       [49]

Run:  python review2_analysis.py     -> results/review2_*.csv and results/review2_report.txt
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

TEST = ["Speaker_03", "Speaker_07", "Speaker_11", "Speaker_15"]
N_BOOT = 2000


def eer_thr(s, y):
    fpr, tpr, thr = roc_curve(y, s, pos_label=1)       # y=1 bona fide, higher score = more bona fide
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2 * 100, thr[i]


def eer(s, y):
    return eer_thr(s, y)[0]


def setting_of(run):
    return re.sub(r"_s\d+$", "", run)


def utt_of(path):
    return os.path.splitext(os.path.basename(str(path).replace("\\", "/")))[0].rsplit("_", 1)[-1]


def spk_utt(path, speaker):
    norm = str(path).replace("\\", "/")
    stem = os.path.splitext(os.path.basename(norm))[0]
    if "/new/" in norm or "/new_extra/" in norm:
        spk, utt = stem.rsplit("_", 1)
        return spk, utt
    return speaker, stem


def load(prefix=""):
    runs = {}
    for f in sorted(glob.glob(os.path.join("results", f"scores_{prefix}*.csv"))):
        name = os.path.basename(f)[len("scores_"):-4]
        if not prefix and name.startswith(("ood_", "extra_")):
            continue
        df = pd.read_csv(f)
        su = [spk_utt(p, s) for p, s in zip(df.path, df.speaker)]
        df["src_spk"] = [a for a, _ in su]
        df["utt"] = [b for _, b in su]
        runs[name[len(prefix):]] = df
    return runs


def clean_unseen(df):
    d = df[df.condition == "unseen_generators"]
    return d[d.generator != "real_mp3"].reset_index(drop=True)


def seen(df):
    return df[df.condition == "seen_generators"].reset_index(drop=True)


def cluster_ci(d, key, rng, n=N_BOOT):
    groups = list(d.groupby(key).indices.values())
    s, y = d.score.values, d.y.values
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(eer(s[idx], y[idx]))
    return np.percentile(vals, [2.5, 97.5]), len(groups)


def main():
    rng = np.random.default_rng(0)
    runs = load()
    rep = []

    # A. per-seed EER
    rows = []
    for run, df in runs.items():
        rows.append(dict(run=run, setting=setting_of(run), seen=eer(seen(df).score, seen(df).y),
                         unseen_clean=eer(clean_unseen(df).score, clean_unseen(df).y)))
    a = pd.DataFrame(rows).round(2).sort_values("run")
    a.to_csv("results/review2_per_seed.csv", index=False)
    rep.append("=== A. Per-seed EER (%) ===\n" + a.to_string(index=False))

    # B. error counts at the EER threshold (clean unseen)
    rows = []
    for run, df in runs.items():
        d = clean_unseen(df)
        e, t = eer_thr(d.score.values, d.y.values)
        real, fake = d[d.y == 1], d[d.y == 0]
        rows.append(dict(run=run, eer=round(e, 2), n_real=len(real), n_fake=len(fake),
                         real_rejected=int((real.score < t).sum()), fakes_accepted=int((fake.score >= t).sum()),
                         fakes_above_lowest_real=int((fake.score >= real.score.min()).sum())))
    b = pd.DataFrame(rows).sort_values("run")
    b.to_csv("results/review2_error_counts.csv", index=False)
    rep.append("\n=== B. Clean unseen set: errors at the EER threshold ===\n" + b.to_string(index=False))

    # C. prompt-level clusters (utterance number across speakers) vs speaker+utterance clusters
    rows = []
    for st in ["ssl-xlsr_noaug", "ssl-whisper_noaug", "ssl-xlsr_noaug_layer5"]:
        for r in sorted(k for k in runs if setting_of(k) == st):
            for tname, f in [("seen", seen), ("unseen_clean", clean_unseen)]:
                d = f(runs[r]).copy()
                d["c_spk_utt"] = d.src_spk + "|" + d.utt
                (lo1, hi1), n1 = cluster_ci(d, "c_spk_utt", rng)
                (lo2, hi2), n2 = cluster_ci(d, "utt", rng)
                rows.append(dict(run=r, test=tname, eer=eer(d.score, d.y), ci_spk_utt=f"{lo1:.2f}-{hi1:.2f}",
                                 n_clusters_spk_utt=n1, ci_prompt=f"{lo2:.2f}-{hi2:.2f}", n_clusters_prompt=n2))
    c = pd.DataFrame(rows).round(2)
    c.to_csv("results/review2_prompt_clusters.csv", index=False)
    rep.append("\n=== C. 95% CI per seed: speaker+utterance clusters vs prompt (utterance-number) clusters ===\n"
               + c.to_string(index=False))

    # D. prompt-source subgroup difference with clustered CI (XLS-R)
    rows = []
    for r in sorted(k for k in runs if setting_of(k) == "ssl-xlsr_noaug"):
        d = clean_unseen(runs[r])
        real = d[d.y == 1]
        fake = d[d.y == 0]
        sub = {"test": fake[fake.src_spk.isin(TEST)], "traindev": fake[~fake.src_spk.isin(TEST)]}
        rg, gt = list(real.groupby("utt").indices.values()), {k: list(v.groupby("utt").indices.values())
                                                              for k, v in sub.items()}
        diffs = []
        for _ in range(N_BOOT):
            ri = np.concatenate([rg[i] for i in rng.integers(0, len(rg), len(rg))])
            rr = real.iloc[ri]
            es = {}
            for k, v in sub.items():
                gi = np.concatenate([gt[k][i] for i in rng.integers(0, len(gt[k]), len(gt[k]))])
                ff = v.iloc[gi]
                es[k] = eer(np.r_[rr.score, ff.score], np.r_[rr.y, ff.y])
            diffs.append(es["test"] - es["traindev"])
        rows.append(dict(run=r, n_fake_test_prompts=len(sub["test"]), n_fake_traindev_prompts=len(sub["traindev"]),
                         eer_test_prompts=eer(np.r_[real.score, sub["test"].score], np.r_[real.y, sub["test"].y]),
                         eer_traindev_prompts=eer(np.r_[real.score, sub["traindev"].score],
                                                  np.r_[real.y, sub["traindev"].y]),
                         diff_ci_low=np.percentile(diffs, 2.5), diff_ci_high=np.percentile(diffs, 97.5)))
    dd = pd.DataFrame(rows).round(2)
    dd.to_csv("results/review2_prompt_subgroup.csv", index=False)
    rep.append("\n=== D. XLS-R clean unseen EER by prompt source, difference (test - train/dev) with 95% CI ===\n"
               + dd.to_string(index=False))

    # E. FLEURS: per-generator and macro EER_OOD
    ood = load("ood_")
    rows = []
    for run, df in ood.items():
        bona = df[df.y == 1]
        row = dict(run=run, setting=setting_of(run), pooled=eer(df.score, df.y))
        for g, s in df[df.y == 0].groupby("generator"):
            row[g] = eer(np.r_[bona.score, s.score], np.r_[bona.y, s.y])
        rows.append(row)
    e = pd.DataFrame(rows)
    gens = [c for c in e.columns if c not in ("run", "setting", "pooled")]
    e["macro"] = e[gens].mean(axis=1)
    e.round(2).to_csv("results/review2_ood_per_generator.csv", index=False)
    es = e.groupby("setting")[["pooled", "macro"] + gens].mean().round(2)
    rep.append(f"\n=== E. FLEURS EER_OOD (%): pooled vs macro over {len(gens)} generators (mean over seeds) ===\n"
               + es.to_string())

    # F. per-speaker miss / false alarm at the pooled threshold (XLS-R, clean unseen)
    rows = []
    for r in sorted(k for k in runs if setting_of(k) == "ssl-xlsr_noaug"):
        for tname, f in [("seen", seen), ("unseen_clean", clean_unseen)]:
            d = f(runs[r])
            _, t = eer_thr(d.score.values, d.y.values)
            for spk in TEST:
                real = d[(d.y == 1) & (d.speaker == spk)]
                own = d[(d.speaker == spk) | (d.y == 0)] if tname == "unseen_clean" else d[d.speaker == spk]
                rows.append(dict(run=r, test=tname, speaker=spk, real_rejected_pct=(real.score < t).mean() * 100,
                                 own_threshold_eer=eer(own.score, own.y)))
    fdf = pd.DataFrame(rows)
    fs = fdf.groupby(["test", "speaker"])[["real_rejected_pct", "own_threshold_eer"]].mean().round(2)
    fs.to_csv("results/review2_per_speaker_pooled_threshold.csv")
    rep.append("\n=== F. XLS-R per speaker: bona fide rejected at the POOLED threshold vs EER with own threshold ===\n"
               + fs.to_string())

    # G. prompt de-duplication
    if os.path.exists("meta/texts.csv"):
        t = pd.read_csv("meta/texts.csv", encoding="utf-8-sig")
        t["part"] = t.src_path.str.extract(r"(Part \d)")
        t["key"] = t.speaker + "|" + t.utt.astype(str)
        dup = t[t.key.duplicated(keep="first")]
        kept_of_dup = t[t.key.isin(dup.key) & ~t.key.duplicated(keep="first")]
        dup.to_csv("results/review2_discarded_prompts.csv", index=False, encoding="utf-8-sig")
        rep.append(f"\n=== G. Prompt de-duplication ===\nRows in meta/texts.csv: {len(t)} "
                   f"(sampled with groupby('speaker').sample(n=30, random_state=42))\n"
                   f"Duplicate speaker|utterance keys discarded (later row kept out): {len(dup)}\n"
                   f"Discarded rows by corpus part: {dup.part.value_counts().to_dict()}\n"
                   f"Kept rows of those keys by part: {kept_of_dup.part.value_counts().to_dict()}\n"
                   f"Same transcript text within a duplicated key: "
                   f"{int(sum(t[t.key == k].text.nunique() == 1 for k in dup.key))} of {len(dup)}")

    # H. clip counts
    sp = {os.path.basename(f)[6:-4]: pd.read_csv(f) for f in glob.glob("meta/split_*.csv")}
    main_splits = ["train", "dev", "eval_seen", "eval_unseen", "eval_ood"]
    uniq = pd.concat([sp[k] for k in main_splits if k in sp]).path.nunique()
    aug = pd.concat([sp["train"], sp["dev"]]).path.nunique()
    rep.append(f"\n=== H. Clips behind the main XLS-R extraction ===\nunique clean clips (train, dev, seen, unseen, "
               f"FLEURS): {uniq}\nMP3-augmented copies (train + dev): {aug}\ntotal: {uniq + aug}\n"
               + "\n".join(f"  {k}: {sp[k].path.nunique()} unique" for k in sorted(sp)))

    open("results/review2_report.txt", "w", encoding="utf-8").write("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    main()
