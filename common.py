"""Shared preprocessing and evaluation used by every model, so all models see identical input.

Shortcut controls (see paper Sec. Method):
  * resample everything to 16 kHz        (Tacotron files are 20 kHz)
  * trim leading/trailing silence         (TTS silence differs from real speech)
  * RMS loudness normalisation            (level differences between sources)
  * fixed 4 s window, repeat-padded       (VITS clips are ~2 s longer than real)
"""
import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.metrics import roc_curve

SR = 16000
SEG = 4 * SR          # 4-second window
PREP_DIR = os.path.join("data", "prep")


def load_raw(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


def preprocess(wav):
    trimmed, _ = librosa.effects.trim(wav, top_db=30)
    if len(trimmed) >= SR // 4:       # keep original if trimming leaves < 0.25 s
        wav = trimmed
    rms = np.sqrt(np.mean(wav ** 2)) + 1e-8
    wav = np.clip(wav / rms * 0.05, -1.0, 1.0)
    return wav.astype(np.float32)


def prep_path(path, aug=False):
    """data\\urdu_deepfake\\Bonafide\\Speaker_01\\Part 1\\10.wav -> data\\prep\\...\\10.npy"""
    rel = os.path.normpath(path)
    if rel.startswith("data" + os.sep):
        rel = rel[len("data" + os.sep):]
    stem = os.path.splitext(rel)[0] + ("__aug" if aug else "")
    return os.path.join(PREP_DIR, stem + ".npy")


def load_prep(path, aug=False):
    return np.load(prep_path(path, aug)).astype(np.float32)


def fix_len(wav, train=False, rng=None):
    n = len(wav)
    if n >= SEG:
        start = int(rng.integers(0, n - SEG + 1)) if train else (n - SEG) // 2
        return wav[start:start + SEG]
    reps = int(np.ceil(SEG / n))
    return np.tile(wav, reps)[:SEG]


def split_names(include_ood=True):
    """All split files present in meta/: train, dev and every eval_* split."""
    import glob
    names = sorted(os.path.basename(f)[6:-4] for f in glob.glob(os.path.join("meta", "split_*.csv")))
    order = {"train": 0, "dev": 1}
    names.sort(key=lambda n: (order.get(n, 2), n))
    return [n for n in names if include_ood or n != "eval_ood"]


def read_split(name):
    df = pd.read_csv(os.path.join("meta", f"split_{name}.csv"))
    df["y"] = (df.label == "bonafide").astype(int)     # 1 = bonafide (real), 0 = spoof
    return df


# ---------------------------------------------------------------- evaluation
def eer(scores, labels):
    """EER in %, higher score = more likely bonafide, labels 1 = bonafide."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2 * 100)


def eer_table(df, run_name):
    """df needs columns: generator, y, score, condition. Returns tidy rows."""
    rows = []
    for cond, d in df.groupby("condition"):
        bona = d[d.y == 1]
        spoof = d[d.y == 0]
        sub = pd.concat([bona, spoof])
        rows.append(dict(run=run_name, condition=cond, generator="ALL",
                         bonafide_ref="all_real", eer=eer(sub.score, sub.y),
                         n_bona=len(bona), n_spoof=len(spoof)))
        refs = {"all_real": bona}
        if (bona.generator == "real_mp3").any():          # codec-matched reference
            refs["real_mp3_only"] = bona[bona.generator == "real_mp3"]
        for g, s in spoof.groupby("generator"):
            for ref_name, ref in refs.items():
                sub = pd.concat([ref, s])
                rows.append(dict(run=run_name, condition=cond, generator=g,
                                 bonafide_ref=ref_name, eer=eer(sub.score, sub.y),
                                 n_bona=len(ref), n_spoof=len(s)))
    return pd.DataFrame(rows)


def save_results(scores_df, run_name):
    os.makedirs("results", exist_ok=True)
    scores_df.to_csv(os.path.join("results", f"scores_{run_name}.csv"), index=False)
    tab = eer_table(scores_df, run_name)
    tab.to_csv(os.path.join("results", f"eer_{run_name}.csv"), index=False)
    print(f"\n===== EER (%) — {run_name} =====")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return tab
