"""Analyses requested in review, computed from EXISTING scores (no GPU, no retraining).

  1. Clean main unseen test: bona fide = original recordings only (MP3 copies excluded)      [note 14]
  2. Clustered bootstrap CIs: resample source utterances (a cluster holds a recording, its
     MP3 copy and every fake generated from the same prompt)                                  [notes 3, 24, 32]
  3. Paired clustered bootstrap test vs. every setting, one-sided, Holm-corrected            [notes 3, 25]
  4. Per-test-speaker sensitivity (leave-one-speaker-out)                                     [notes 9, 32]
  5. Prompt-overlap audit + prompt-source sensitivity                                         [notes 7, 12, 13]
  6. Leave-one-generator-out training counts                                                  [note 30]
  7. Repeat-padding audit: share of test clips shorter than the 4 s window, per class        [note 10]

Run:  python review_analysis.py            -> results/review_*.csv and a printed report
      python review_analysis.py --n 2000   (bootstrap draws; default 2000)
"""
import argparse
import difflib
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

REF = "ssl-xlsr_noaug"
TEST = ["Speaker_03", "Speaker_07", "Speaker_11", "Speaker_15"]
DEV = ["Speaker_05", "Speaker_13"]


def eer(s, y):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2 * 100


def setting_of(run):
    return re.sub(r"_s\d+$", "", run)


def cluster_key(path, speaker):
    """Group a recording, its MP3 copy and all fakes made from the same source prompt."""
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    norm = str(path).replace("\\", "/")
    if "/new/" in norm or "/new_extra/" in norm:            # Speaker_03_10.wav -> Speaker_03|10
        spk, utt = stem.rsplit("_", 1)
        return f"{spk}|{utt}"
    if "/fleurs" in norm:
        return f"fleurs|{stem}"
    return f"{speaker}|{stem}"                                # corpus file: speaker + utterance number


def load_scores():
    runs = {}
    for f in sorted(glob.glob(os.path.join("results", "scores_*.csv"))):
        name = os.path.basename(f)[len("scores_"):-4]
        if name.startswith(("ood_", "extra_")):
            continue
        df = pd.read_csv(f)
        df["cluster"] = [cluster_key(p, s) for p, s in zip(df.path, df.speaker)]
        runs[name] = df
    return runs


def clean_unseen(df):
    d = df[df.condition == "unseen_generators"]
    return d[d.generator != "real_mp3"]


class ClusterSampler:
    """Clustered bootstrap: resample whole clusters with replacement."""

    def __init__(self, clusters):
        clusters = pd.Series(np.asarray(clusters))
        self.groups = list(clusters.groupby(clusters).indices.values())

    def draw(self, rng):
        pick = rng.integers(0, len(self.groups), len(self.groups))
        return np.concatenate([self.groups[i] for i in pick])


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--ref", default=REF)
    args = ap.parse_args()
    ref = args.ref
    rng = np.random.default_rng(0)
    runs = load_scores()
    if not runs:
        raise SystemExit("No results/scores_*.csv found (run git pull first).")
    os.makedirs("results", exist_ok=True)
    report = []

    # ------------------------------------------------ 1 + 2: clean unseen EER and clustered CIs
    tests = {
        "seen": lambda d: d[d.condition == "seen_generators"],
        "unseen_clean": clean_unseen,
        "unseen_with_mp3": lambda d: d[d.condition == "unseen_generators"],
    }
    rows = []
    for name, df in runs.items():
        for tname, f in tests.items():
            d = f(df).reset_index(drop=True)
            if d.empty or d.y.nunique() < 2:
                continue
            base = eer(d.score.values, d.y.values)
            boots = []
            sampler = ClusterSampler(d.cluster)
            for _ in range(args.n):
                idx = sampler.draw(rng)
                yy = d.y.values[idx]
                if len(np.unique(yy)) < 2:
                    continue
                boots.append(eer(d.score.values[idx], yy))
            rows.append(dict(run=name, setting=setting_of(name), test=tname, eer=base,
                             ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                             n_clusters=d.cluster.nunique(), n_bona=int(d.y.sum()), n_spoof=int((d.y == 0).sum())))
            # per-generator, clean unseen
            if tname == "unseen_clean":
                for g, s in d[d.y == 0].groupby("generator"):
                    sub = pd.concat([d[d.y == 1], s])
                    rows.append(dict(run=name, setting=setting_of(name), test=f"unseen_clean:{g}",
                                     eer=eer(sub.score.values, sub.y.values)))
    ci = pd.DataFrame(rows)
    ci.to_csv("results/review_clustered_ci.csv", index=False)
    summ = ci.groupby(["test", "setting"]).agg(eer_mean=("eer", "mean"), eer_std=("eer", "std"),
                                                 ci_low=("ci_low", "mean"), ci_high=("ci_high", "mean"),
                                                 n_seeds=("eer", "size")).round(2).reset_index()
    summ.to_csv("results/review_clustered_ci_summary.csv", index=False)
    report.append("=== 1-2. EER (%) with 95% CLUSTERED bootstrap CI (clusters = source utterances); "
                  "CI averaged over seeds ===")
    report.append(summ.to_string(index=False))

    # ------------------------------------------------ 3: paired clustered test vs reference, Holm
    ref_runs = sorted(k for k in runs if setting_of(k) == ref)
    prow = []
    if ref_runs:
        for st in sorted({setting_of(k) for k in runs} - {ref}):
            other = sorted(k for k in runs if setting_of(k) == st)
            pairs = list(zip(ref_runs, other if len(other) > 1 else other * len(ref_runs)))
            for tname in ["seen", "unseen_clean"]:
                f = tests[tname]
                cache, ok = [], True
                for r, o in pairs:
                    a = f(runs[r]).sort_values("path").reset_index(drop=True)
                    b = f(runs[o]).sort_values("path").reset_index(drop=True)
                    if len(a) != len(b) or not (a.path.values == b.path.values).all():
                        ok = False
                        break
                    cache.append((a.y.values, a.score.values, b.score.values))
                if not ok or not cache:
                    continue
                clusters = f(runs[ref_runs[0]]).sort_values("path").reset_index(drop=True).cluster
                sampler = ClusterSampler(clusters)
                prng = np.random.default_rng(10_000)
                diffs = []
                for i in range(args.n):
                    idx = sampler.draw(prng)
                    diffs.append(np.mean([eer(sb[idx], y[idx]) - eer(sa[idx], y[idx]) for y, sa, sb in cache]))
                diffs = np.array(diffs)
                p = (np.sum(diffs <= 0) + 1) / (len(diffs) + 1)          # one-sided, add-one estimate
                prow.append(dict(reference=ref, other=st, test=tname, mean_gain=diffs.mean(),
                                 ci_low=np.percentile(diffs, 2.5), ci_high=np.percentile(diffs, 97.5),
                                 p_one_sided=p))
    paired = pd.DataFrame(prow)
    if len(paired):
        paired["p_holm"] = holm(paired.p_one_sided.values)
        paired = paired.round(4)
        paired.to_csv("results/review_paired_clustered.csv", index=False)
        report.append(f"\n=== 3. Paired clustered bootstrap: EER(other) - EER({ref}); one-sided; "
                      f"{args.n} draws; smallest resolvable p = {1/(args.n+1):.4f}; Holm over {len(paired)} tests ===")
        report.append(paired.to_string(index=False))

    # ------------------------------------------------ 4: per-test-speaker sensitivity
    srows = []
    for name, df in runs.items():
        seen = tests["seen"](df)
        uns = clean_unseen(df)
        for spk in TEST:
            d = seen[seen.speaker == spk]
            if d.y.nunique() == 2:
                srows.append(dict(run=name, setting=setting_of(name), test="seen", speaker=spk,
                                  eer=eer(d.score.values, d.y.values)))
            d = pd.concat([uns[(uns.y == 1) & (uns.speaker == spk)], uns[uns.y == 0]])
            if d.y.nunique() == 2:
                srows.append(dict(run=name, setting=setting_of(name), test="unseen_clean", speaker=spk,
                                  eer=eer(d.score.values, d.y.values)))
    sp = pd.DataFrame(srows)
    sp.to_csv("results/review_per_speaker.csv", index=False)
    sps = sp.groupby(["test", "setting", "speaker"]).eer.mean().unstack("speaker").round(2)
    sps["min"], sps["max"] = sps.min(axis=1), sps.max(axis=1)
    report.append("\n=== 4. Per-test-speaker EER (%) (mean over seeds) ===")
    report.append(sps.to_string())

    # ------------------------------------------------ 5: prompt-overlap audit
    texts = pd.read_csv("meta/texts.csv", encoding="utf-8-sig")
    texts["cluster"] = texts.speaker + "|" + texts.utt.astype(str)
    texts = texts.drop_duplicates("cluster")
    role = lambda s: "test" if s in TEST else ("dev" if s in DEV else "train")
    texts["source_split"] = texts.speaker.map(role)
    report.append("\n=== 5. Prompt audit ===")
    report.append("Unique prompts used for the modern-TTS fakes, by the split of their source speaker:")
    report.append(texts.source_split.value_counts().to_string())
    # do utterance numbers index the SAME sentence across speakers?
    same, diff = [], []
    t = texts.reset_index(drop=True)
    r2 = np.random.default_rng(1)
    for u, g in t.groupby(t.utt.astype(str)):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                same.append(difflib.SequenceMatcher(None, str(g.text[i]), str(g.text[j])).ratio())
    for _ in range(max(200, len(same))):
        i, j = r2.integers(len(t), size=2)
        if t.utt[i] != t.utt[j]:
            diff.append(difflib.SequenceMatcher(None, str(t.text[i]), str(t.text[j])).ratio())
    if same:
        report.append(f"Transcript similarity, same utterance number / different speakers: "
                      f"mean {np.mean(same):.2f} over {len(same)} pairs; random pairs: {np.mean(diff):.2f}")
        report.append("-> similarity near 1 means utterance numbers index shared prompts read by every speaker.")
    train = pd.read_csv("meta/split_train.csv")
    train_utts = {os.path.splitext(os.path.basename(p))[0] for p in train.path}
    texts["utt_in_train"] = texts.utt.astype(str).isin(train_utts)
    report.append(f"Prompts whose utterance number also occurs in the training split: "
                  f"{texts.utt_in_train.sum()} / {len(texts)}")
    texts[["cluster", "speaker", "utt", "source_split", "utt_in_train"]].to_csv(
        "results/review_prompt_audit.csv", index=False)
    # prompt-source sensitivity: unseen-generator EER using only fakes whose prompt came from test speakers
    psens = []
    for name, df in runs.items():
        d = clean_unseen(df)
        fake = d[d.y == 0].copy()
        fake["src_spk"] = fake.cluster.str.split("|").str[0]
        for subset, keep in [("prompts_from_test_speakers", fake.src_spk.isin(TEST)),
                             ("prompts_from_train_dev_speakers", ~fake.src_spk.isin(TEST))]:
            sub = pd.concat([d[d.y == 1], fake[keep]])
            if sub.y.nunique() == 2 and keep.sum() > 0:
                psens.append(dict(run=name, setting=setting_of(name), subset=subset,
                                  n_fake=int(keep.sum()), eer=eer(sub.score.values, sub.y.values)))
    ps = pd.DataFrame(psens)
    ps.to_csv("results/review_prompt_sensitivity.csv", index=False)
    report.append("Unseen-generator EER (%) by prompt source (mean over seeds):")
    report.append(ps.groupby(["setting", "subset"]).agg(eer=("eer", "mean"), n_fake=("n_fake", "first"))
                  .round(2).unstack("subset").to_string())

    # ------------------------------------------------ 6: LOGO training counts
    report.append("\n=== 6. Training utterances by generator (full train split) ===")
    report.append(train.groupby(["generator", "label"]).size().to_string())

    # ------------------------------------------------ 7: repeat-padding audit
    try:
        from common import SEG, load_prep
        rows = []
        for split in ["eval_seen", "eval_unseen", "train"]:
            df = pd.read_csv(f"meta/split_{split}.csv")
            for r in df.itertuples():
                try:
                    n = len(load_prep(r.path))
                except Exception:
                    continue
                rows.append(dict(split=split, generator=r.generator, seconds=n / 16000, short=n < SEG))
        pad = pd.DataFrame(rows)
        pt = pad.groupby(["split", "generator"]).agg(n=("short", "size"), share_padded=("short", "mean"),
                                                     median_s=("seconds", "median")).round(3)
        pt.to_csv("results/review_padding.csv")
        report.append("\n=== 7. Share of clips shorter than 4 s after trimming (these are repeat-padded) ===")
        report.append(pt.to_string())
    except Exception as e:
        report.append(f"\n=== 7. Padding audit skipped ({e}); needs data/prep ===")

    text = "\n".join(report)
    open("results/review_report.txt", "w", encoding="utf-8").write(text)
    print(text)
    print("\nSaved results/review_report.txt and results/review_*.csv")


if __name__ == "__main__":
    main()
