"""Extra test splits requested in review (no change to the main splits).

eval_allmp3 : every seen- and unseen-test clip (real and fake) passed through the SAME MP3 round trip
              (24 kHz, 48 kbps), so codec artefacts are identical across classes (review note 15).
eval_voices : test-speaker real speech vs. two additional Microsoft Urdu voices (ur-IN Gul, Salman)
              generated with generate.py --out data\\new_extra (review notes 17/33).

Run:  python make_extra_splits.py
"""
import hashlib
import os
import subprocess
import tempfile
from multiprocessing import Pool

import pandas as pd

OUT_MP3 = os.path.join("data", "allmp3")


def mp3_copy(args):
    src, dst = args
    if os.path.exists(dst):
        return "skip"
    fd, tmp = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1",
                        "-ar", "24000", "-b:a", "48k", tmp], check=True)
        part = dst + ".part.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1",
                        "-ar", "16000", part], check=True)
        os.replace(part, dst)
        return "ok"
    except Exception as e:
        return f"fail: {src} | {str(e)[:120]}"
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    cols = ["path", "label", "generator", "speaker"]
    seen = pd.read_csv("meta/split_eval_seen.csv")
    unseen = pd.read_csv("meta/split_eval_unseen.csv")

    # ---- eval_allmp3
    rows, jobs = [], []
    for df, cond in [(seen, "allmp3_seen_generators"),
                     (unseen[unseen.generator != "real_mp3"], "allmp3_unseen_generators")]:
        for r in df.itertuples():
            h = hashlib.md5(r.path.encode("utf-8")).hexdigest()[:12]
            dst = os.path.join(OUT_MP3, r.generator, f"{r.speaker}_{h}.wav")
            jobs.append((r.path, dst))
            rows.append(dict(path=dst, label=r.label, generator=r.generator, speaker=r.speaker,
                             split="test", condition=cond, src_path=r.path))
    jobs = list(dict.fromkeys(jobs))
    workers = max(1, (os.cpu_count() or 2) - 1)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(mp3_copy, jobs, chunksize=8), 1):
            k = res if res in counts else "fail"
            counts[k] += 1
            if k == "fail":
                print(res)
            if i % 500 == 0 or i == len(jobs):
                print(f"MP3 copies {i}/{len(jobs)} {counts}")
    pd.DataFrame(rows).to_csv("meta/split_eval_allmp3.csv", index=False)
    print("Saved meta/split_eval_allmp3.csv:", len(rows), "rows")

    # ---- eval_voices
    man = os.path.join("meta", "manifest_extra.csv")
    if os.path.exists(man):
        extra = pd.read_csv(man)
        real = unseen[unseen.generator == "real"][cols]
        v = pd.concat([real, extra[extra.label == "spoof"][cols]], ignore_index=True)
        v["split"], v["condition"] = "test", "extra_voices"
        v.to_csv("meta/split_eval_voices.csv", index=False)
        print("Saved meta/split_eval_voices.csv:")
        print(v.groupby(["generator", "label"]).size().to_string())
    else:
        print("No meta/manifest_extra.csv yet -> run generate.py for the extra voices first.")


if __name__ == "__main__":
    main()
