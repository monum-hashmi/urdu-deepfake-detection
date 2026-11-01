"""Out-of-domain REAL speech control (negative control).
Bonafide = genuine Urdu speech from Google FLEURS (different speakers, microphones, rooms).
Spoof    = the same test fakes as before (Tacotron/VITS for test speakers + all unseen generators).
If a detector flags FLEURS as fake, it learned recording conditions, not synthesis artefacts.

Run (after extracting FLEURS into data\\fleurs_ur):  python make_ood.py
"""
import glob
import os

import pandas as pd

TEST = ["Speaker_03", "Speaker_07", "Speaker_11", "Speaker_15"]

wavs = sorted(glob.glob(os.path.join("data", "fleurs_ur", "**", "*.wav"), recursive=True))
if not wavs:
    raise SystemExit("No FLEURS wavs found under data\\fleurs_ur")

orig = pd.read_csv("meta/manifest_original.csv")
new = pd.read_csv("meta/manifest_new.csv")
cols = ["path", "label", "generator", "speaker"]

bona = pd.DataFrame(dict(path=wavs, label="bonafide", generator="real_fleurs", speaker="fleurs"))
seen_fakes = orig[orig.speaker.isin(TEST) & (orig.generator != "real")][cols]
unseen_fakes = new[new.generator != "real_mp3"][cols]

df = pd.concat([bona, seen_fakes, unseen_fakes], ignore_index=True)
df["split"], df["condition"] = "test", "ood_real"
df.to_csv("meta/split_eval_ood.csv", index=False)
print(df.groupby(["generator", "label"]).size().to_string())
print(f"\nSaved meta/split_eval_ood.csv ({len(df)} clips, {len(bona)} FLEURS real)")
