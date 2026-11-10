"""Independent-provider, multi-voice test set (review round 3, notes 11 and 12).

AI4Bharat Indic Parler-TTS (ai4bharat/indic-parler-tts, Apache-2.0) supports Urdu; the voice is set by a text
description. We synthesise the same prompts with six described voices (3 female, 3 male).

Runs in its OWN virtual environment, because parler-tts needs an older transformers version:
    python -m venv .venv_parler
    .\.venv_parler\Scripts\Activate.ps1
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install git+https://github.com/huggingface/parler-tts.git soundfile pandas
    hf auth login            (paste a Hugging Face read token; accept the model's terms on its page first)
    python parler_gen.py     -> data/parler/<voice>/<speaker>_<utt>.wav (16 kHz), results/round4_parler_voices.csv
Safe to stop and re-run; finished clips are skipped.
"""
import os
import subprocess

import pandas as pd
import soundfile as sf
import torch

N_PROMPTS = 60
VOICES = {
    "parler_f_high": "A female speaker with a high-pitched voice speaks at a moderate pace. The recording is very clear, with no background noise.",
    "parler_f_low": "A female speaker with a low-pitched, calm voice speaks slowly. The recording is very clear, with no background noise.",
    "parler_f_fast": "A young female speaker speaks quickly in an expressive tone. The recording is very clear, with no background noise.",
    "parler_m_deep": "A male speaker with a deep voice speaks at a moderate pace. The recording is very clear, with no background noise.",
    "parler_m_high": "A male speaker with a high-pitched voice speaks quickly. The recording is very clear, with no background noise.",
    "parler_m_slow": "An older male speaker speaks slowly in a monotone voice. The recording is very clear, with no background noise.",
}


def main():
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "ai4bharat/indic-parler-tts"
    model = ParlerTTSForConditionalGeneration.from_pretrained(name).to(device).eval()
    tok = AutoTokenizer.from_pretrained(name)
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate

    t = pd.read_csv(os.path.join("meta", "texts.csv"), encoding="utf-8-sig")
    t = t[~(t.speaker + "|" + t.utt.astype(str)).duplicated(keep="first")]
    t = t[t.text.fillna("").str.len() > 3].sample(N_PROMPTS, random_state=3)
    os.makedirs(os.path.join("data", "tmp"), exist_ok=True)
    rows = []
    for v, desc in VOICES.items():
        os.makedirs(os.path.join("data", "parler", v), exist_ok=True)
        d = dtok(desc, return_tensors="pt").to(device)
        for i, r in enumerate(t.itertuples()):
            dst = os.path.join("data", "parler", v, f"{r.speaker}_{r.utt}.wav")
            rows.append(dict(voice=v, description=desc, speaker=r.speaker, utt=r.utt, path=dst))
            if os.path.exists(dst):
                continue
            torch.manual_seed(1000 + i)
            p = tok(r.text, return_tensors="pt").to(device)
            with torch.no_grad():
                wav = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                     prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
            tmp = os.path.join("data", "tmp", "parler.wav")
            sf.write(tmp, wav.cpu().float().numpy().squeeze(), sr)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-ar", "16000", dst],
                           check=True)
        print(f"{v}: done")
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join("results", "round4_parler_voices.csv"), index=False, encoding="utf-8-sig")
    print("Saved results/round4_parler_voices.csv")


if __name__ == "__main__":
    main()
