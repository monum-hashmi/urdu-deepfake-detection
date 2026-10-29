import os, collections
import soundfile as sf

root = r"data\urdu_deepfake"
ext_count = collections.Counter()
dirs = collections.Counter()
samples = []

for d, _, files in os.walk(root):
    if ".cache" in d:
        continue
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        ext_count[ext] += 1
        rel = os.path.relpath(d, root).split(os.sep)
        dirs["/".join(rel[:2])] += 1
        if ext in (".wav", ".flac", ".mp3") and len(samples) < 5:
            samples.append(os.path.join(d, f))

print("File types:", dict(ext_count))
print("\nTop folders (first 2 levels) -> file count:")
for k, v in sorted(dirs.items())[:40]:
    print(f"  {k}: {v}")

print("\nSample audio files:")
for p in samples:
    info = sf.info(p)
    print(f"  {p} | {info.samplerate} Hz | {info.duration:.1f}s | {info.channels}ch")

for d, _, files in os.walk(root):
    for f in files:
        if f.endswith((".csv", ".jsonl", ".json", ".parquet")) and ".cache" not in d:
            p = os.path.join(d, f)
            print(f"\n--- {p} ---")
            if p.endswith(".parquet"):
                import pandas as pd
                df = pd.read_parquet(p)
                print(df.columns.tolist()); print(df.head())
            else:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    for _ in range(5):
                        print(fh.readline().rstrip())