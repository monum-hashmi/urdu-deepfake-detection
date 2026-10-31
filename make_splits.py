import pandas as pd

orig = pd.read_csv("meta/manifest_original.csv")
new = pd.read_csv("meta/manifest_new.csv")

# Speaker-disjoint split: no speaker appears in more than one set
TEST = ["Speaker_03", "Speaker_07", "Speaker_11", "Speaker_15"]
DEV = ["Speaker_05", "Speaker_13"]
TRAIN = [s for s in sorted(orig.speaker.unique()) if s not in TEST + DEV]


def tag(df, split, cond):
    df = df.copy()
    df["split"], df["condition"] = split, cond
    return df[["path", "label", "generator", "speaker", "split", "condition"]]


train = tag(orig[orig.speaker.isin(TRAIN)], "train", "original")
dev = tag(orig[orig.speaker.isin(DEV)], "dev", "original")

# Eval A: same generators as training (Tacotron, VITS), unseen speakers
eval_orig = tag(orig[orig.speaker.isin(TEST)], "test", "seen_generators")

# Eval B: unseen generators. Real side = test-speaker real clips, both raw WAV
# and MP3 round-trip, so MP3 artefacts alone cannot separate the classes
real_test = orig[(orig.speaker.isin(TEST)) & (orig.generator == "real")]
real_mp3 = new[(new.generator == "real_mp3") & (new.speaker.isin(TEST))]
fakes_new = new[new.generator != "real_mp3"]
eval_new = pd.concat([
    tag(real_test, "test", "unseen_generators"),
    tag(real_mp3, "test", "unseen_generators"),
    tag(fakes_new, "test", "unseen_generators"),
])

for name, df in [("train", train), ("dev", dev),
                 ("eval_seen", eval_orig), ("eval_unseen", eval_new)]:
    df.to_csv(f"meta/split_{name}.csv", index=False)
    print(f"\n== {name}: {len(df)} clips")
    print(df.groupby(["generator", "label"]).size().to_string())

print("\nTrain speakers:", TRAIN)
print("Dev speakers:", DEV)
print("Test speakers:", TEST)
