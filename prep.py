"""Preprocess every clip once (16 kHz, trimmed, RMS-normalised) into data/prep/*.npy.
Also builds codec-augmented copies of the TRAIN and DEV clips (MP3 round-trip at a
random bitrate) used by the --aug training runs.

Run:  python prep.py            (safe to stop and re-run; skips finished files)
"""
import os
import subprocess
import tempfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf

from common import load_raw, preprocess, prep_path, read_split, split_names

BITRATES = ["24k", "32k", "48k", "64k"]


def mp3_roundtrip(src, seed):
    rng = np.random.default_rng(seed)
    br = BITRATES[int(rng.integers(len(BITRATES)))]
    fd, tmp_mp3 = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1",
                        "-ar", "16000", "-b:a", br, tmp_mp3], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_mp3,
                        "-ac", "1", "-ar", "16000", tmp_wav], check=True)
        wav, _ = sf.read(tmp_wav, dtype="float32")
    finally:
        for p in (tmp_mp3, tmp_wav):
            if os.path.exists(p):
                os.remove(p)
    return wav


def job(args):
    path, aug, seed = args
    out = prep_path(path, aug)
    if os.path.exists(out):
        return "skip"
    try:
        wav = mp3_roundtrip(path, seed) if aug else load_raw(path)
        wav = preprocess(wav)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out[:-4] + ".tmp.npy"
        np.save(tmp, wav.astype(np.float16))
        os.replace(tmp, out)          # atomic: never leaves a half-written .npy
        return "ok"
    except Exception as e:
        return f"fail: {path} | {str(e)[:150]}"


if __name__ == "__main__":
    splits = {n: read_split(n) for n in split_names()}
    all_paths = pd.concat(splits.values()).path.unique().tolist()
    aug_paths = pd.concat([splits["train"], splits["dev"]]).path.unique().tolist()
    jobs = [(p, False, 0) for p in all_paths] + \
           [(p, True, i) for i, p in enumerate(aug_paths)]
    print(f"Clean clips: {len(all_paths)} | augmented copies: {len(aug_paths)}")

    workers = max(1, (os.cpu_count() or 2) - 1)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(job, jobs, chunksize=8), 1):
            key = res if res in counts else "fail"
            counts[key] += 1
            if key == "fail":
                print(res)
            if i % 500 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}  {counts}")
    print("Done:", counts)
