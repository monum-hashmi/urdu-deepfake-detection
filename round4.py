"""Review round 3, extra controls (main .venv):

  python round4.py fleurs      download a balanced male/female sample of the FLEURS Urdu TRAIN split and write
                               meta/split_eval_ood2.csv (FLEURS test + the unseen fakes)        [note 14]
  python round4.py parler      write meta/split_eval_parler.csv (test-speaker real + Parler fakes) [11, 12]
                               (run parler_gen.py in .venv_parler first)
  python prep.py ; python extract_ssl.py --clean-only ; python extract_whisper.py --clean-only
  python round4.py score       dev-threshold FA / miss and EER on both sets, by gender and by voice

Outputs: results/round4_*.csv
"""
import glob
import os
import sys
import tarfile

import numpy as np
import pandas as pd

from round3 import _thr_at


def cmd_fleurs():
    """FLEURS Urdu dev and test are female-only; the TRAIN split has both genders. We take a balanced
    random sample of 300 male + 300 female train clips (FLEURS train is never used for training here)."""
    from huggingface_hub import hf_hub_download
    root = os.path.join("data", "fleurs_raw")
    tsv = hf_hub_download("google/fleurs", "data/ur_pk/train.tsv", repo_type="dataset", local_dir=root)
    f = pd.read_csv(tsv, sep="\t", header=None, quoting=3)
    f = pd.concat([g.sample(min(300, len(g)), random_state=0) for _, g in f.groupby(6)])
    out = os.path.join("data", "fleurs_ur")
    want = set(f[1])
    have = {os.path.basename(p) for p in glob.glob(os.path.join(out, "train", "*.wav"))}
    if not want <= have:
        tgz = hf_hub_download("google/fleurs", "data/ur_pk/audio/train.tar.gz", repo_type="dataset",
                              local_dir=root)
        with tarfile.open(tgz) as tf:
            members = [m for m in tf.getmembers() if os.path.basename(m.name) in want]
            tf.extractall(out, members=members)
    f["path"] = [os.path.join(out, "train", n) for n in f[1]]
    f = f[[os.path.exists(p) for p in f.path]]
    print("FLEURS train sample:", len(f), "| gender:", f[6].value_counts().to_dict(),
          "| sentences:", f[0].nunique())
    real = pd.DataFrame(dict(path=f.path, label="bonafide", generator="real_fleurs_" + f[6].str.lower(),
                             speaker="fleurs_train", split="test", condition="ood2"))
    uns = pd.read_csv(os.path.join("meta", "split_eval_unseen.csv"))
    fakes = uns[uns.label == "spoof"].assign(condition="ood2")
    pd.concat([real, fakes]).to_csv(os.path.join("meta", "split_eval_ood2.csv"), index=False)
    print("meta/split_eval_ood2.csv written")


def cmd_parler():
    real = pd.read_csv(os.path.join("meta", "split_eval_unseen.csv"))
    real = real[real.generator == "real"]
    rows = []
    for p in sorted(glob.glob(os.path.join("data", "parler", "*", "*.wav"))):
        spk = os.path.basename(p).rsplit("_", 1)[0]
        rows.append(dict(path=p, label="spoof", generator=os.path.basename(os.path.dirname(p)), speaker=spk,
                         split="test", condition="parler"))
    df = pd.concat([real.assign(condition="parler"), pd.DataFrame(rows)])
    df.to_csv(os.path.join("meta", "split_eval_parler.csv"), index=False)
    print(f"meta/split_eval_parler.csv: {len(rows)} Parler fakes, "
          f"{pd.DataFrame(rows).generator.nunique() if rows else 0} voices")


def cmd_score():
    import torch
    from common import eer, read_split
    from score_extra import score_df
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = read_split("dev")
    sets = [n for n in ["eval_parler", "eval_ood2"] if os.path.exists(os.path.join("meta", f"split_{n}.csv"))]
    rows = []
    for ckpt in sorted(glob.glob(os.path.join("checkpoints", "*.pt"))):
        run = os.path.splitext(os.path.basename(ckpt))[0]
        if "_tr" in run or "_uniform" in run or "nopad" in run:
            continue
        sd = score_df(run, ckpt, dev, device)
        if sd is None:
            continue
        t = _thr_at(np.asarray(sd), dev.y.values, "eer")
        for n in sets:
            df = read_split(n)
            df["score"] = score_df(run, ckpt, df, device)
            df.to_csv(os.path.join("results", f"round4_scores_{n}_{run}.csv"), index=False)
            real, fake = df[df.y == 1], df[df.y == 0]
            r = dict(run=run, set=n, eer=eer(df.score, df.y), fa=(real.score < t).mean() * 100,
                     miss=(fake.score >= t).mean() * 100)
            groups = real.groupby("generator") if n == "eval_ood2" else fake.groupby("generator")
            for g, s in groups:
                sub = pd.concat([s, fake]) if n == "eval_ood2" else pd.concat([real, s])
                r[f"eer_{g}"] = eer(sub.score, sub.y)
                if n == "eval_ood2":
                    r[f"fa_{g}"] = (s.score < t).mean() * 100
                else:
                    r[f"miss_{g}"] = (s.score >= t).mean() * 100
            rows.append(r)
        print("scored", run)
    out = pd.DataFrame(rows)
    out.round(2).to_csv(os.path.join("results", "round4_controls.csv"), index=False)
    out["setting"] = out.run.str.replace(r"_s\d+$", "", regex=True)
    for n in sets:
        d = out[out.set == n].dropna(axis=1, how="all")
        print(f"\n=== {n}: mean over seeds (FA / miss at the dev-set EER threshold) ===")
        print(d.drop(columns=["run", "set"]).groupby("setting").mean().round(2).T.to_string())


if __name__ == "__main__":
    cmds = dict(fleurs=cmd_fleurs, parler=cmd_parler, score=cmd_score)
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    os.makedirs("results", exist_ok=True)
    cmds[sys.argv[1]]()
