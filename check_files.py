"""Delete broken/half-written .npy files (e.g. after a 'disk full' crash) so they get redone.
Run:  python check_files.py
"""
import glob
import os

import numpy as np

bad = 0
files = glob.glob(os.path.join("data", "prep", "**", "*.npy"), recursive=True) + \
        glob.glob(os.path.join("data", "feat_*", "**", "*.npy"), recursive=True)
for i, f in enumerate(files, 1):
    try:
        if f.endswith(".tmp.npy"):
            raise ValueError("temp file")
        a = np.load(f)
        if a.size == 0 or not np.isfinite(a.astype(np.float32)).all():
            raise ValueError("empty or NaN")
    except Exception:
        os.remove(f)
        bad += 1
    if i % 2000 == 0:
        print(f"checked {i}/{len(files)}  deleted {bad}")
print(f"Checked {len(files)} files, deleted {bad} broken ones.")
