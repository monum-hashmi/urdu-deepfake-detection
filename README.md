# Frozen Self-Supervised Features Transfer to Unseen Urdu TTS Under Cross-Generator Testing

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)](https://pytorch.org/)

Code, splits and synthesis prompts for the paper

> **Frozen Self-Supervised Features Transfer to Unseen Urdu TTS Under Cross-Generator Testing**
> Monum Hashmi and Daniyal Adeeb, Department of Artificial Intelligence, University of Management and Technology, Lahore, Pakistan

Public anti-spoofing data for Urdu is scarce. The main Urdu-specific deepfake corpus (Munir et al., 2024) contains only Tacotron and VITS fakes, and detectors are usually tested on the generators they were trained on. We ask: **does a detector trained on this benchmark still work on selected, currently available TTS services?**

We build a speaker-disjoint protocol that mitigates known dataset shortcuts and add **1,952 fakes from four unseen synthesis configurations** from three providers (two Microsoft neural voices, Google TTS, Meta MMS-TTS). Every result is checked with codec, voice and out-of-domain controls.

## Key findings

- **AASIST-L with its released ASVspoof weights does not transfer:** 65.3% EER on the unseen TTS. LFCC-LCNN reaches 52.7% and frozen Whisper features 16.4%.
- **Frozen XLS-R features transfer best.** A small head on frozen XLS-R 300M reaches **2.94% EER** on unseen generators. The 95% cluster-bootstrap CI is 2.2–3.8%, conditional on the four test speakers. It beats every baseline (p = 0.0005, Holm-adjusted p ≤ 0.013).
- **The result holds under the controls:**
  - 4.09% when both classes pass through an identical MP3 pipeline
  - 3.69% on two extra synthetic voices
- **Out-of-domain genuine Urdu speech (FLEURS, 267 clips, all female):**
  - XLS-R's threshold-free EER is 4.29%.
  - At a threshold fixed on the development set, its false alarms rise from 8.5% (seen genuine speech) to 13.7% (FLEURS).
  - AASIST-L trained on Urdu rises from 1.75% to 15.5% false alarms and misses 89.6% of unseen fakes, consistent with corpus-specific cues.
- **The training generator matters:** XLS-R trained on VITS alone reaches 4.5% EER on unseen TTS; trained on Tacotron alone, 22.6%.
- **Layer 5 alone** is better than the learned layer weighting on unseen TTS (0.02%) and FLEURS, but weaker on VITS. The learned weighting is a reasonable default, not a proven necessity.

## Main results

EER (%), mean ± std over 3 seeds. Lower is better. *Unseen* uses the original recordings only as the bona fide class.

| Detector | Seen (Tacotron, VITS) | Unseen TTS | 95% CI | All-MP3 unseen | Extra voices |
|---|---|---|---|---|---|
| AASIST-L (ASVspoof weights, no Urdu training) | 40.80 | 65.25 | 63.6–66.8 | 65.32 | 70.31 |
| AASIST-L (trained on Urdu) | 20.59 ± 3.36 | 37.20 ± 2.84 | 35.4–39.3 | 36.92 | 50.43 |
| LFCC-LCNN | 13.31 ± 2.20 | 52.65 ± 3.58 | 51.0–54.2 | 28.51 | 54.96 |
| Whisper-small (frozen) + MLP | 8.68 ± 0.70 | 16.40 ± 1.22 | 15.2–17.7 | 16.90 | 7.87 |
| **XLS-R 300M (frozen) + MLP** | **5.36 ± 0.85** | **2.94 ± 1.51** | **2.2–3.8** | **4.09** | **3.69** |

### Out-of-domain control (FLEURS)

FA = genuine speech rejected; Miss = unseen fakes accepted. Both are taken at the EER threshold of the development set, set per seed. EER<sub>OOD</sub> uses FLEURS as the only bona fide class.

| Detector | FA seen real | FA FLEURS | Miss unseen fakes | EER<sub>OOD</sub> pooled | EER<sub>OOD</sub> macro |
|---|---|---|---|---|---|
| AASIST-L (ASVspoof) | 13.39 | 2.25 | 89.86 | 45.03 | 42.08 |
| AASIST-L | 1.75 | 15.48 | 89.60 | 48.44 | 44.63 |
| LFCC-LCNN | 3.00 | 0.25 | 80.50 | 45.21 | 39.20 |
| Whisper + MLP | 2.88 | 3.62 | 36.39 | 16.01 | 12.71 |
| **XLS-R + MLP** | 8.51 | 13.73 | **0.53** | **4.29** | **3.52** |

### XLS-R ablations

| Variant | Seen | Unseen | FA FLEURS | EER<sub>OOD</sub> pooled |
|---|---|---|---|---|
| Learned layer weights (default) | 5.36 | 2.94 | 13.73 | 4.29 |
| Uniform layer average | 7.24 | 6.41 | 14.11 | 6.97 |
| Layer 5 only | 3.42 | 0.02 | 3.37 | 5.16 |
| No repeat padding | 5.38 | 3.08 | 14.11 | 4.10 |
| Trained on VITS only | 8.76 | 4.52 | 20.85 | 5.42 |
| Trained on Tacotron only | 29.93 | 22.57 | 3.87 | 21.59 |
| + MP3/noise augmentation | 8.20 | 7.49 | 18.60 | 9.58 |

All numbers are in [`results/`](results/): per-generator and per-seed EERs, augmentation runs, clustered CIs, paired tests, per-speaker results, dev-threshold rates, the prompt audit and the inference benchmark.

## Protocol

| Part | Details |
|---|---|
| Speaker-disjoint split | 11 train / 2 dev / 4 test speakers (4,397 / 799 / 1,598 utterances). Dev: Speaker_05, 13. Test: Speaker_03, 07, 11, 15. Fixed before any training. |
| Shortcut mitigation | Resample to 16 kHz, trim silence (top_db = 30), RMS-normalise, 4 s window (tiled if shorter; random crop in training for the waveform models only) |
| Unseen generators | `ur-PK-UzmaNeural`, `ur-PK-AsadNeural` (Microsoft Edge TTS), gTTS (`ur`), `facebook/mms-tts-urd-script_arabic`; 488 prompts each |
| Voice control | `ur-IN-GulNeural`, `ur-IN-SalmanNeural` (same provider); 976 extra fakes |
| Codec control | Every test clip of both classes through the same MP3 round trip (24 kHz, 48 kbps, the Microsoft output format) |
| Out-of-domain control | 267 genuine Urdu utterances from FLEURS (`ur_pk`, dev; all female speakers) |
| Statistics | 3 seeds; cluster bootstrap (recording + its MP3 copy + fakes from its transcript), 2,000 draws; paired one-sided tests with Holm correction over 26 comparisons; operating point fixed on the dev set |

## Setup

Tested on Windows 11, Python 3.11.9, RTX 5060 Laptop GPU (8 GB), PyTorch 2.11.0 + CUDA 12.8, Transformers 5.22.1 and FFmpeg 9.0.2. Everything except AASIST-L training also runs on a CPU.

```bash
git clone https://github.com/monum-hashmi/urdu-deepfake-detection.git
cd urdu-deepfake-detection
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    Linux/macOS: source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128   # or /whl/cpu
pip install -r requirements.txt
```

`ffmpeg` must be on your PATH (Windows: `winget install Gyan.FFmpeg`).

## Data

Datasets are **not** included. Download them into `data/`:

```bash
# Urdu deepfake corpus (Munir et al., 2024), CC BY-NC 4.0
hf download CSALT/deepfake_detection_dataset_urdu --repo-type dataset --local-dir data/urdu_deepfake

# FLEURS Urdu dev split (out-of-domain control), CC BY 4.0
hf download google/fleurs data/ur_pk/audio/dev.tar.gz data/ur_pk/dev.tsv --repo-type dataset --local-dir data/fleurs_raw
mkdir -p data/fleurs_ur && tar -xzf data/fleurs_raw/data/ur_pk/audio/dev.tar.gz -C data/fleurs_ur
```

The generated TTS audio is not redistributed, because the TTS services' terms may restrict redistribution of their output. Rebuild it from the prompts in `meta/texts.csv` with `generate.py`. This needs internet access but no API keys or accounts. The online services may change, so regenerated audio may not be byte-identical to ours (generated October–November 2026 with edge-tts 7.3.1 and gTTS 2.5.4).

## Reproducing the paper

The split files in `meta/` are included. Steps 1–3 are only needed to rebuild everything from scratch.

| Step | Command |
|---|---|
| 1. Index the corpus | `python make_manifest.py` |
| 2. Transcribe prompts (Whisper-small) | `python transcribe.py` |
| 3. Generate unseen TTS and MP3 control | `python generate.py` |
| 4. Speaker-disjoint splits | `python make_splits.py` |
| 5. FLEURS control split | `python make_ood.py` |
| 6. Preprocess (shortcut mitigation + MP3 copies) | `python prep.py` |
| 7. Cache frozen features | `python extract_ssl.py` and `python extract_whisper.py` |
| 8. XLS-R / Whisper classifiers | `python train_ssl.py --seed 0 [--aug] [--tag whisper]` |
| 9. LFCC-LCNN | `python train_lcnn.py --seed 0 [--aug]` |
| 10. AASIST-L | `git clone https://github.com/clovaai/aasist.git`, then `python train_aasist.py --mode pretrained` and `python train_aasist.py --mode train --seed 0 --bs 8 [--aug]` |
| 11. Summaries and statistics | `python summarize.py`, `python score_ood.py`, `python bootstrap.py` (clustered bootstrap, same as `review_analysis.py`) |

Run every training command with `--seed 0`, `1` and `2`.

### Controls, ablations and analyses

| Experiment | Command |
|---|---|
| Extra voices | `python generate.py --gens edge_in_gul edge_in_salman --out data/new_extra --manifest meta/manifest_extra.csv` |
| All-MP3 and extra-voice splits | `python make_extra_splits.py`, `python prep.py`, `python extract_ssl.py --clean-only` |
| Score existing models on them | `python score_extra.py` |
| Leave-one-generator-out | `python train_ssl.py --seed 0 --train_gen vits` (or `tacotron`) |
| Uniform / single-layer pooling | `python train_ssl.py --seed 0 --pool uniform` / `--pool layer --layer 5` |
| No repeat padding | `python extract_ssl.py --pad none --tag xlsrnopad --clean-only`, then `python train_ssl.py --seed 0 --tag xlsrnopad` |
| Operating point fixed on the dev set | `python score_dev.py` |
| Clustered CIs, paired tests, per-speaker, prompt and padding audits | `python review_analysis.py` and `python review2_analysis.py` |
| Inference speed and memory | `python bench_infer.py` |
| Codec of each TTS service | `python probe_codecs.py` |
| Figures | `python plot_figs.py .` |

XLS-R feature extraction for 14,326 clips takes 4.4 minutes on the RTX 5060 Laptop GPU (fp16, batch 8). One classifier run takes about 12 seconds. Inference takes 12.2 ms per 4 s clip (batch 1, fp16, 0.66 GB peak memory). `check_files.py` removes half-written files after an interrupted run.

## Repository layout

```
common.py              preprocessing, EER, split loading
generate.py            modern-TTS synthesis + MP3 control
make_*.py              manifests and splits
prep.py                shortcut mitigation, augmentation copies
extract_ssl.py         frozen XLS-R features
extract_whisper.py     frozen Whisper encoder features
train_ssl.py           layer-weighted MLP (XLS-R / Whisper)
train_lcnn.py          LFCC-LCNN baseline
train_aasist.py        AASIST-L (pretrained / trained)
score_ood.py           FLEURS out-of-domain control
score_extra.py         all-MP3 and extra-voice evaluation
score_dev.py           false-alarm and miss rates at the dev-set threshold
review_analysis.py     clustered bootstrap, paired tests, audits
review2_analysis.py    per-seed results, error counts, prompt clusters, per-speaker rates
bench_infer.py         inference speed and memory
plot_figs.py           paper figures
meta/                  splits (with speaker IDs) and prompts
results/               all result tables
```

## Licences and credits

- **Code:** MIT (see `LICENSE`).
- **Urdu deepfake corpus:** CC BY-NC 4.0, from [CSALT-LUMS](https://huggingface.co/datasets/CSALT/deepfake_detection_dataset_urdu) (Munir et al., 2024). Non-commercial research use only.
- **FLEURS:** CC BY 4.0, Google.
- **AASIST:** official implementation from [clovaai/aasist](https://github.com/clovaai/aasist) (MIT), cloned separately.
- **Models:** `facebook/wav2vec2-xls-r-300m`, `openai/whisper-small` and `facebook/mms-tts-urd-script_arabic` are downloaded from Hugging Face under their own licences.
- `meta/texts.csv` contains automatic transcripts of corpus utterances and inherits the corpus licence.

Claude (Anthropic) was used to help write the code in this repository. All experiments were run and checked by the authors.

## Citation

```bibtex
@misc{hashmi2026urdudeepfake,
  author = {Hashmi, Monum and Adeeb, Daniyal},
  title  = {Frozen Self-Supervised Features Transfer to Unseen {U}rdu {TTS} Under Cross-Generator Testing},
  year   = {2026},
  note   = {Preprint}
}
```

## Contact

Monum Hashmi, monumhashmi1@gmail.com
