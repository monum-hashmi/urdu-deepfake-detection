import os, csv
import soundfile as sf
import pandas as pd

root = r"data\urdu_deepfake"
gen_map = {"Bonafide": "real", "Spoofed_TTS": "vits", "Spoofed_Tacotron": "tacotron"}
rows = []
for cls, gen in gen_map.items():
    for d, _, files in os.walk(os.path.join(root, cls)):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            p = os.path.join(d, f)
            rel = os.path.relpath(p, root).split(os.sep)
            info = sf.info(p)
            rows.append(dict(
                path=p,
                label="bonafide" if gen == "real" else "spoof",
                generator=gen,
                speaker=rel[1],
                subfolder="/".join(rel[2:-1]),
                utt=os.path.splitext(f)[0],
                sr=info.samplerate,
                dur=round(info.duration, 2),
            ))

df = pd.DataFrame(rows)
df.to_csv("meta/manifest_original.csv", index=False, encoding="utf-8")
print("Total:", len(df))
print("\nClips per generator/subfolder:\n", df.groupby(["generator", "subfolder"]).size())
print("\nDuration (s) per generator:\n", df.groupby("generator").dur.describe().round(2))
print("\nSample rates:\n", df.groupby("generator").sr.value_counts())
s1 = df[df.speaker == "Speaker_01"]
for (g, sub), grp in s1.groupby(["generator", "subfolder"]):
    ids = sorted(grp.utt.tolist(), key=lambda x: int(x) if x.isdigit() else x)[:10]
    print(f"\nSpeaker_01 | {g} | {sub} | first IDs: {ids}")