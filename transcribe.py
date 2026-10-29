import pandas as pd
from faster_whisper import WhisperModel

df = pd.read_csv("meta/manifest_original.csv")
real = df[df.generator == "real"]
sample = real.groupby("speaker").sample(n=30, random_state=42)

model = WhisperModel("small", device="cpu", compute_type="int8")
out = []
for i, r in enumerate(sample.itertuples()):
    segs, _ = model.transcribe(r.path, language="ur", beam_size=1)
    text = " ".join(s.text.strip() for s in segs).strip()
    out.append(dict(src_path=r.path, speaker=r.speaker, utt=r.utt, text=text))
    if i % 25 == 0:
        print(i, "/", len(sample), "|", text)
        pd.DataFrame(out).to_csv("meta/texts.csv", index=False, encoding="utf-8-sig")

pd.DataFrame(out).to_csv("meta/texts.csv", index=False, encoding="utf-8-sig")
print("Done:", len(out), "sentences -> meta/texts.csv")