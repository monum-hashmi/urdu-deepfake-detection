"""Review round 3 experiments. Run the steps in this order (each is safe to re-run):

  python round3.py gen          generate (a) NEW-TEXT fakes from FLEURS sentences and (b) the 22 prompts
                                that were lost to the Part 1/Part 2 filename collision          [notes 7, 9, 10]
  python round3.py splits       write meta/split_eval_newtext.csv and meta/split_eval_fix510.csv
  python prep.py                (preprocesses only the new clips)
  python extract_ssl.py --clean-only
  python extract_whisper.py --clean-only
  python round3.py score        score every existing checkpoint on the two new test sets          [7, 9, 10]
  python round3.py oppoints     operating points fixed on dev: EER, 1% bona fide rejection,
                                1% spoof acceptance -> FA / miss on test and FLEURS              [27]
  python round3.py partitions   leave-speaker-group-out: XLS-R (learned, layer 5) and Whisper retrained
                                on 4 further speaker-disjoint partitions, 3 seeds each          [6, 17, 32]
  python round3.py manifest     SHA-256 + duration + generation date of every generated clip    [35]

Outputs: results/round3_*.csv
"""
import glob
import hashlib
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

FLEURS_TSV = os.path.join("data", "fleurs_raw", "data", "ur_pk", "dev.tsv")
GENS = ["edge_uzma", "edge_asad", "gtts", "mms"]
TEST = ["Speaker_03", "Speaker_07", "Speaker_11", "Speaker_15"]


# --------------------------------------------------------------------------------------------- gen
def _generate(rows, out_root, gens):
    """rows: list of (name, text). Uses the same synthesis functions as generate.py."""
    from generate import EDGE_VOICES, TMP, gen_edge, gen_gtts, gen_mms
    os.makedirs(TMP, exist_ok=True)
    fails = []
    for gen in gens:
        os.makedirs(os.path.join(out_root, gen), exist_ok=True)
        ok = 0
        for i, (name, text) in enumerate(rows):
            dst = os.path.join(out_root, gen, f"{name}.wav")
            if os.path.exists(dst):
                ok += 1
                continue
            err = ""
            for attempt in range(3):
                try:
                    if gen in EDGE_VOICES:
                        gen_edge(text, EDGE_VOICES[gen], dst)
                    elif gen == "gtts":
                        gen_gtts(text, dst)
                    else:
                        gen_mms(text, dst, 100000 + i)
                    ok += 1
                    break
                except Exception as e:
                    err = str(e)[:200]
                    time.sleep(3 * (attempt + 1))
            else:
                fails.append(dict(gen=gen, name=name, error=err))
        print(f"{out_root} / {gen}: {ok}/{len(rows)} ok")
    return fails


def cmd_gen():
    import difflib
    # (a) new text: FLEURS Urdu dev sentences, one per sentence id
    f = pd.read_csv(FLEURS_TSV, sep="\t", header=None, quoting=3)
    f = f.drop_duplicates(subset=0)
    rows = [(f"fleurs_{int(r[0])}", str(r[2])) for _, r in f.iterrows() if len(str(r[2])) > 3]
    corpus = pd.read_csv(os.path.join("meta", "texts.csv"), encoding="utf-8-sig").text.fillna("").tolist()
    sims = [max(difflib.SequenceMatcher(None, t, c).ratio() for c in corpus) for _, t in rows]
    pd.DataFrame(dict(name=[n for n, _ in rows], text=[t for _, t in rows], max_sim_to_corpus=sims)) \
        .to_csv(os.path.join("results", "round3_newtext_prompts.csv"), index=False, encoding="utf-8-sig")
    print(f"New-text prompts: {len(rows)}; max similarity to any corpus transcript: "
          f"mean {np.mean(sims):.2f}, max {np.max(sims):.2f}")
    fails = _generate(rows, os.path.join("data", "newtext"), GENS)

    # (b) the 22 prompts lost to the filename collision, now named with the corpus part
    t = pd.read_csv(os.path.join("meta", "texts.csv"), encoding="utf-8-sig")
    t["part"] = t.src_path.str.extract(r"Part (\d)")[0]
    lost = t[(t.speaker + "|" + t.utt.astype(str)).duplicated(keep="first")]
    rows = [(f"{r.speaker}_P{r.part}_{r.utt}", r.text) for r in lost.itertuples()]
    fails += _generate(rows, os.path.join("data", "new_fix"), GENS)
    pd.DataFrame(fails).to_csv(os.path.join("results", "round3_gen_failures.csv"), index=False)
    print(f"Done. Failures: {len(fails)} (results/round3_gen_failures.csv)")


# ------------------------------------------------------------------------------------------ splits
def cmd_splits():
    real = pd.read_csv(os.path.join("meta", "split_eval_unseen.csv"))
    real = real[real.generator == "real"].copy()
    rows = []
    for gen in GENS:
        for p in sorted(glob.glob(os.path.join("data", "newtext", gen, "*.wav"))):
            rows.append(dict(path=p, label="spoof", generator=gen, speaker="newtext", split="test",
                             condition="newtext"))
    nt = pd.concat([real.assign(condition="newtext"), pd.DataFrame(rows)])
    nt.to_csv(os.path.join("meta", "split_eval_newtext.csv"), index=False)

    uns = pd.read_csv(os.path.join("meta", "split_eval_unseen.csv"))
    uns = uns[uns.generator != "real_mp3"]
    rows = []
    for gen in GENS:
        for p in sorted(glob.glob(os.path.join("data", "new_fix", gen, "*.wav"))):
            spk = os.path.basename(p).split("_P")[0]
            rows.append(dict(path=p, label="spoof", generator=gen, speaker=spk, split="test", condition="fix510"))
    fx = pd.concat([uns.assign(condition="fix510"), pd.DataFrame(rows)])
    fx.to_csv(os.path.join("meta", "split_eval_fix510.csv"), index=False)
    print(f"split_eval_newtext: {len(nt)} rows ({(nt.label == 'spoof').sum()} fakes); "
          f"split_eval_fix510: {len(fx)} rows ({(fx.label == 'spoof').sum()} fakes)")


# ------------------------------------------------------------------------------------------- score
def cmd_score():
    import torch
    from common import eer_table, read_split
    from score_extra import score_df
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = ["eval_newtext", "eval_fix510"]
    tabs = []
    for ckpt in sorted(glob.glob(os.path.join("checkpoints", "*.pt"))):
        run = os.path.splitext(os.path.basename(ckpt))[0]
        if "_tr" in run or "_uniform" in run:
            continue
        parts = []
        for n in names:
            df = read_split(n)
            try:
                s = score_df(run, ckpt, df, device)
            except FileNotFoundError:
                print("skip", run, "(no features for the new clips)")
                s = None
            if s is None:
                break
            parts.append(df.assign(score=s))
        if len(parts) < len(names):
            continue
        allp = pd.concat(parts)
        allp.to_csv(os.path.join("results", f"round3_scores_{run}.csv"), index=False)
        tabs.append(eer_table(allp, run))
        print("scored", run)
    df = pd.concat(tabs)
    df.to_csv(os.path.join("results", "round3_eer_newtext_fix510.csv"), index=False)
    df["setting"] = df.run.str.replace(r"_s\d+$", "", regex=True)
    out = df[df.bonafide_ref == "all_real"].groupby(["condition", "generator", "setting"]).eer \
        .agg(["mean", "std"]).round(2).reset_index()
    out.to_csv(os.path.join("results", "round3_summary_newtext_fix510.csv"), index=False)
    out["val"] = out.apply(lambda r: f"{r['mean']:.2f} ± {0 if pd.isna(r['std']) else r['std']:.2f}", axis=1)
    print(out.pivot_table(index=["condition", "generator"], columns="setting", values="val",
                          aggfunc="first").to_string())


# ---------------------------------------------------------------------------------------- oppoints
def _thr_at(scores, y, kind):
    """kind: 'eer' | 'frr1' (1% bona fide rejected on dev) | 'far1' (1% spoof accepted on dev)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, scores, pos_label=1)     # fpr = spoof accepted, 1-tpr = bona rejected
    fnr = 1 - tpr
    if kind == "eer":
        return thr[np.nanargmin(np.abs(fnr - fpr))]
    if kind == "frr1":
        return thr[np.where(fnr <= 0.01)[0][0]]
    return thr[np.where(fpr <= 0.01)[0][-1]]


def cmd_oppoints():
    import torch
    from common import read_split
    from score_extra import score_df
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = read_split("dev")
    rows = []
    for ckpt in sorted(glob.glob(os.path.join("checkpoints", "*.pt"))):
        run = os.path.splitext(os.path.basename(ckpt))[0]
        f_test, f_ood = f"results/scores_{run}.csv", f"results/scores_ood_{run}.csv"
        if not (os.path.exists(f_test) and os.path.exists(f_ood)):
            continue
        s = score_df(run, ckpt, dev, device)
        if s is None:
            continue
        test, ood = pd.read_csv(f_test), pd.read_csv(f_ood)
        seen = test[test.condition == "seen_generators"]
        uns = test[(test.condition == "unseen_generators") & (test.generator != "real_mp3")]
        fl = ood[ood.y == 1]
        for kind in ["eer", "frr1", "far1"]:
            t = _thr_at(np.asarray(s), dev.y.values, kind)
            rows.append(dict(run=run, op=kind,
                             fa_seen_real=(seen[seen.y == 1].score < t).mean() * 100,
                             fa_fleurs=(fl.score < t).mean() * 100,
                             miss_seen=(seen[seen.y == 0].score >= t).mean() * 100,
                             miss_unseen=(uns[uns.y == 0].score >= t).mean() * 100))
        print("done", run)
    df = pd.DataFrame(rows)
    df.round(2).to_csv(os.path.join("results", "round3_operating_points.csv"), index=False)
    df["setting"] = df.run.str.replace(r"_s\d+$", "", regex=True)
    print(df.groupby(["op", "setting"])[["fa_seen_real", "fa_fleurs", "miss_seen", "miss_unseen"]]
          .mean().round(2).to_string())


# -------------------------------------------------------------------------------------- partitions
def _train_eval(tag, pool, layer, seed, tr, dv, tests, device):
    import torch
    import torch.nn as nn
    from common import eer
    from train_ssl import LayerMLP, load_feats, scores
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xtr, Xdv = load_feats(tr, tag), load_feats(dv, tag)
    ytr = torch.tensor(tr.y.values)
    m = LayerMLP(Xtr.shape[1], Xtr.shape[2], pool=pool, layer=layer).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    w = torch.tensor([1.0, (tr.y == 0).sum() / max(1, (tr.y == 1).sum())], dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w.to(device))
    best, bad, state = 1e9, 0, None
    for ep in range(60):
        m.train()
        perm = rng.permutation(len(Xtr))
        for b in range(0, len(perm), 64):
            idx = perm[b:b + 64]
            loss = lossf(m(torch.from_numpy(Xtr[idx]).to(device)), ytr[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        e = eer(scores(m, Xdv, device), dv.y)
        if e < best - 1e-6:
            best, bad = e, 0
            state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    m.load_state_dict(state)
    out = {"dev_eer": best}
    sdv = scores(m, Xdv, device)
    t = _thr_at(sdv, dv.y.values, "eer")
    for name, df in tests.items():
        s = scores(m, load_feats(df, tag), device)
        if name == "fleurs":
            out["eer_ood"] = eer(s, df.y)
            out["fa_fleurs"] = (s[df.y.values == 1] < t).mean() * 100
        else:
            out[f"eer_{name}"] = eer(s, df.y)
            out[f"fa_{name}"] = (s[df.y.values == 1] < t).mean() * 100
            out[f"miss_{name}"] = (s[df.y.values == 0] >= t).mean() * 100
    return out


def cmd_partitions():
    import torch
    from common import read_split
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = pd.concat([read_split(n) for n in ["train", "dev", "eval_seen"]])
    corpus = corpus[corpus.condition != "unseen_generators"] if "condition" in corpus else corpus
    unseen = read_split("eval_unseen")
    unseen_fakes = unseen[(unseen.y == 0)]
    ood = read_split("eval_ood")
    speakers = sorted(corpus.speaker.unique())
    others = [s for s in speakers if s not in TEST + ["Speaker_05", "Speaker_13"]]
    rng = np.random.default_rng(2026)
    pool = list(rng.permutation(others))                   # 11 speakers never tested in the paper
    folds = {"paper": (TEST, ["Speaker_05", "Speaker_13"])}
    tests = [pool[0:4], pool[4:8], pool[8:11] + [TEST[0]], ["Speaker_05", "Speaker_13", "Speaker_07", "Speaker_15"]]
    # with the paper split, every one of the 17 speakers is a test speaker at least once
    for k, test in enumerate(tests):                       # 4 further test groups of 4 speakers
        rest = [s for s in speakers if s not in test]
        dv = list(rng.choice(rest, 2, replace=False))
        folds[f"fold{k + 1}"] = (sorted(map(str, test)), sorted(map(str, dv)))
    rows = []
    for fold, (test, dv_spk) in folds.items():
        tr = corpus[~corpus.speaker.isin(test + dv_spk)].reset_index(drop=True)
        dv = corpus[corpus.speaker.isin(dv_spk)].reset_index(drop=True)
        seen = corpus[corpus.speaker.isin(test)].reset_index(drop=True)
        real_test = seen[seen.y == 1]
        tests = {"seen": seen,
                 "unseen": pd.concat([real_test, unseen_fakes]).reset_index(drop=True),
                 "fleurs": ood}
        for tag, pool_, layer in [("xlsr", "learned", 0), ("xlsr", "layer", 5), ("whisper", "learned", 0)]:
            for seed in range(3):
                r = _train_eval(tag, pool_, layer, seed, tr, dv, tests, device)
                rows.append(dict(fold=fold, test_speakers=" ".join(test), dev_speakers=" ".join(dv_spk),
                                 model=f"{tag}_{pool_}{layer if pool_ == 'layer' else ''}", seed=seed, **r))
                print(fold, rows[-1]["model"], seed, {k: round(v, 2) for k, v in r.items()})
    df = pd.DataFrame(rows)
    df.round(3).to_csv(os.path.join("results", "round3_partitions.csv"), index=False)
    cols = ["eer_seen", "eer_unseen", "fa_seen", "fa_fleurs", "miss_unseen", "eer_ood"]
    print("\n=== per fold (mean over seeds) ===")
    print(df.groupby(["model", "fold"])[cols].mean().round(2).to_string())
    print("\n=== across folds: mean ± std of fold means ===")
    fm = df.groupby(["model", "fold"])[cols].mean()
    print(fm.groupby("model").agg(["mean", "std"]).round(2).to_string())


# ---------------------------------------------------------------------------------------- manifest
def cmd_manifest():
    import soundfile as sf
    rows = []
    for root in ["new", "new_extra", "newtext", "new_fix"]:
        for p in sorted(glob.glob(os.path.join("data", root, "*", "*.wav"))):
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            info = sf.info(p)
            rows.append(dict(path=p.replace("\\", "/"), set=root, generator=os.path.basename(os.path.dirname(p)),
                             sha256=h, seconds=round(info.frames / info.samplerate, 3), sr=info.samplerate,
                             created=datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d")))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join("results", "round3_audio_manifest.csv"), index=False)
    print(df.groupby(["set", "generator"]).agg(n=("path", "size"), first=("created", "min"),
                                               last=("created", "max")).to_string())


if __name__ == "__main__":
    cmds = dict(gen=cmd_gen, splits=cmd_splits, score=cmd_score, oppoints=cmd_oppoints,
                partitions=cmd_partitions, manifest=cmd_manifest)
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    os.makedirs("results", exist_ok=True)
    cmds[sys.argv[1]]()
