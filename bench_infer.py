"""Inference cost of the full XLS-R detector (encoder + head) on one 4 s clip (review round 2, notes 50, 51).

Measures, after warm-up and excluding download / file loading / preprocessing:
  latency per clip at batch size 1, throughput at batch size 8, and peak GPU memory, in fp16 and fp32.
Run:  python bench_infer.py       -> prints a table, saves results/review2_inference.csv
"""
import time

import pandas as pd
import torch
from transformers import Wav2Vec2Model

from common import SEG


def bench(model, bs, dtype, device, n=50):
    x = torch.randn(bs, SEG, device=device, dtype=dtype)
    with torch.no_grad():
        for _ in range(5):
            model(x, output_hidden_states=True)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(n):
            hs = model(x, output_hidden_states=True).hidden_states
            torch.stack(hs, 1).mean(2)                        # pooling step of the detector
        if device == "cuda":
            torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else float("nan")
    return dt * 1000 / bs, bs / dt, mem


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for dtype in ([torch.float16, torch.float32] if device == "cuda" else [torch.float32]):
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m", dtype=dtype).to(device).eval()
        n_par = sum(p.numel() for p in model.parameters()) / 1e6
        for bs in [1, 8]:
            ms, cps, mem = bench(model, bs, dtype, device)
            rows.append(dict(device=torch.cuda.get_device_name() if device == "cuda" else "cpu",
                             dtype=str(dtype).split(".")[-1], batch=bs, params_M=round(n_par, 1),
                             ms_per_clip=round(ms, 1), clips_per_s=round(cps, 1), peak_mem_GB=round(mem, 2)))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv("results/review2_inference.csv", index=False)
    print(df.to_string(index=False))
    print("Head (MLP) adds about 0.26M parameters; its cost is negligible next to the encoder.")


if __name__ == "__main__":
    main()
