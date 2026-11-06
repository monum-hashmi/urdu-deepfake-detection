# Detecting Unseen Urdu Speech Deepfakes

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)](https://pytorch.org/)

Code, splits and synthesis prompts for the paper

> **Detecting Unseen Urdu Speech Deepfakes: Frozen Self-Supervised Features Generalise Where Standard Detectors Fail**
> Monum Hashmi and Daniyal Adeeb, Department of Artificial Intelligence, University of Management and Technology, Lahore, Pakistan

Urdu has more than 200 million speakers, but Urdu deepfake detectors have so far been tested only on the two older TTS systems (Tacotron, VITS) used to build the public Urdu corpus. We ask a practical question: **does a detector trained on this benchmark still catch the TTS systems people use today?**

We build a speaker-disjoint, shortcut-controlled protocol, add **1,952 fakes from four unseen synthesis configurations** (Microsoft neural TTS ×2, Google TTS, Meta MMS-TTS), and check every result with codec, voice and out-of-domain controls.

## Key findings

- **Standard detectors fail on modern Urdu TTS.** AASIST-L with ASVspoof weights reaches 65% EER (its scores are inverted), and LFCC-LCNN about 50%.
- **Frozen XLS-R features generalise.** A small classifier on frozen XLS-R 300M features reaches **2.9% EER** on unseen generators (95% cluster-bootstrap CI 2.2–3.8%). It beats every other configuration (p = 0.0005, Holm-adjusted p ≤ 0.013).
- **The result survives the controls.**
  - Both classes passed through an identical MP3 pipeline: 4.1% EER.
  - Two extra synthetic voices: 3.7% EER.
  - Genuine out-of-domain Urdu speech (FLEURS): the false-alarm rate rises only from 5.3% to 7.4%.
- **AASIST-L trained on Urdu learns the corpus, not the fakes.** It rejects 51% of genuine FLEURS speech.
- **Unseen-generator EER alone can mislead.** A single XLS-R layer scores 0.02% EER on unseen TTS but rejects 28% of genuine FLEURS speech.
- **The training generator matters.** Training on VITS alone transfers to modern TTS (4.5% EER); training on Tacotron alone does not (22.6%).

## Main results

EER (%), mean ± std over 3 seeds. Lower is better. *Unseen* uses original recordings only as the bona fide class.

| Detector | Seen (Tacotron, VITS) | Unseen modern TTS | All-MP3 unseen | Extra voices | FLEURS EER<sub>OOD</sub> |
|---|---|---|---|---|---|
| AASIST-L (ASVspoof weights, no Urdu training) | 40.80 | 65.25 | 65.32 | 70.31 | 45.03 |
| AASIST-L (trained on Urdu) | 20.59 ± 3.36 | 37.20 ± 2.84 | 36.92 | 50.43 | 48.44 |
| LFCC-LCNN | 13.31 ± 2.20 | 52.65 ± 3.58 | 28.51 | 54.96 | 45.21 |
| Whisper-small (frozen) + MLP | 8.68 ± 0.70 | 16.40 ± 1.22 | 16.90 | 7.87 | 16.01 |
| **XLS-R 300M (frozen) + MLP** | **5.36 ± 0.85** | **2.94 ± 1.51** | **4.09** | **3.69** | **4.29** |

### XLS-R ablations

| Variant | Seen | Unseen | False alarms on FLEURS |
|---|---|---|---|
| Learned layer weights (default) | 5.36 | 2.94 | 7.37 |
| Uniform layer average | 7.24 | 6.41 | 7.87 |
| Layer 5 only | 3.42 | 0.02 | **28.21** |
| No repeat padding | 5.38 | 3.08 | 6.62 |
| Trained on VITS only | 8.76 | 4.52 | 10.86 |
| Trained on Tacotron only | 29.93 | 22.57 | 23.47 |
| + MP3/noise augmentation | 8.20 | 7.49 | 12.86 |

All numbers are in [`results/`](results/): per-generator EERs, augmentation runs, clustered confidence intervals, paired tests, per-speaker EERs and the prompt audit.

## Protocol

| Part | Details |
|---|---|
| Speaker-disjoint split | 11 train / 2 dev / 4 test speakers (4,397 / 799 / 1,598 utterances). Dev: Speaker_05, 13. Test: Speaker_03, 07, 11, 15. Fixed before any training. |
| Shortcut removal | Resample to 16 kHz (removes the 20 kHz Tacotron cue), trim silence at 30 dB, RMS-normalise, fixed 4 s window (removes the VITS duration cue) |
| Unseen generators | `ur-PK-UzmaNeural`, `ur-PK-AsadNeural` (Microsoft Edge TTS), gTTS (`ur`), `facebook/mms-tts-urd-script_arabic`; 488 prompts each |
| Voice control | `ur-IN-GulNeural`, `ur-IN-SalmanNeural`; 976 extra fakes |
| Codec control | Every test clip of both classes through the same MP3 round trip (24 kHz, 48 kbps, the same format as Microsoft's output) |
| Out-of-domain control | 267 genuine Urdu utterances from FLEURS (`ur_pk`, dev) |
| Statistics | 3 seeds; cluster bootstrap (recording + its MP3 copy + fakes from its transcript), 2,000 draws; paired one-sided tests with Holm correction |

## Setup

Tested on Windows 11, Python 3.11.9, RTX 5060 Laptop GPU (8 GB), PyTorch 2.11.0 + CUDA 12.8, Transformers 5.22.1, FFmpeg 9.0.2. Everything except AASIST-L training also runs on a CPU.

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

The generated TTS audio is not redistributed. Rebuild it from the prompts in `meta/texts.csv` with `generate.py`. No API keys or accounts are needed. The online services are unofficial endpoints whose voices and rate limits may change, so regenerated audio may differ slightly from ours (generated October–November 2026 with edge-tts 7.3.1 and gTTS 2.5.4).

## Reproducing the paper

The split files in `meta/` are included. Steps 1–3 are only needed to rebuild everything from scratch.

| Step | Command |
|---|---|
| 1. Index the corpus | `python make_manifest.py` |
| 2. Transcribe prompts (Whisper-small) | `python transcribe.py` |
| 3. Generate unseen TTS and MP3 control | `python generate.py` |
| 4. Speaker-disjoint splits | `python make_splits.py` |
| 5. FLEURS control split | `python make_ood.py` |
| 6. Preprocess (shortcut removal + MP3 copies) | `python prep.py` |
| 7. Cache frozen features | `python extract_ssl.py` and `python extract_whisper.py` |
| 8. XLS-R / Whisper classifiers | `python train_ssl.py --seed 0 [--aug] [--tag whisper]` |
| 9. LFCC-LCNN | `python train_lcnn.py --seed 0 [--aug]` |
| 10. AASIST-L | `git clone https://github.com/clovaai/aasist.git`, then `python train_aasist.py --mode pretrained` and `python train_aasist.py --mode train --seed 0 --bs 8 [--aug]` |
| 11. Summaries | `python summarize.py`, `python score_ood.py`, `python bootstrap.py`, `python plot_layers.py` |

Run every training command with `--seed 0`, `1` and `2`.

### Robustness and ablation experiments

| Experiment | Command |
|---|---|
| Extra voices | `python generate.py --gens edge_in_gul edge_in_salman --out data/new_extra --manifest meta/manifest_extra.csv` |
| All-MP3 and extra-voice splits | `python make_extra_splits.py`, `python prep.py`, `python extract_ssl.py --clean-only` |
| Score existing models on them | `python score_extra.py` |
| Leave-one-generator-out | `python train_ssl.py --seed 0 --train_gen vits` (or `tacotron`) |
| Uniform / single-layer pooling | `python train_ssl.py --seed 0 --pool uniform` / `--pool layer --layer 5` |
| No repeat padding | `python extract_ssl.py --pad none --tag xlsrnopad --clean-only`, then `python train_ssl.py --seed 0 --tag xlsrnopad` |
| Clustered stats, per-speaker, prompt and padding audits | `python review_analysis.py` |
| Codec of each TTS service | `python probe_codecs.py` |

On an RTX 5060 Laptop GPU, XLS-R feature extraction for about 14k clips takes about 4.5 minutes and one classifier run about 12 seconds. `check_files.py` removes half-written files after an interrupted run.

## Repository layout

```
common.py              preprocessing, EER, split loading
generate.py            modern-TTS synthesis + MP3 control
make_*.py              manifests and splits
prep.py                shortcut removal, augmentation copies
extract_ssl.py         frozen XLS-R features
extract_whisper.py     frozen Whisper encoder features
train_ssl.py           layer-weighted MLP (XLS-R / Whisper)
train_lcnn.py          LFCC-LCNN baseline
train_aasist.py        AASIST-L (pretrained / trained)
score_ood.py           FLEURS out-of-domain control
score_extra.py         all-MP3 and extra-voice evaluation
review_analysis.py     clustered bootstrap, paired tests, audits
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
- Generated audio is not shared, because it derives from non-commercial data and third-party services.

Claude (Anthropic) was used to help write the code in this repository. All experiments were run and checked by the authors.

## Citation

```bibtex
@misc{hashmi2026urdudeepfake,
  author = {Hashmi, Monum and Adeeb, Daniyal},
  title  = {Detecting Unseen {U}rdu Speech Deepfakes: Frozen Self-Supervised Features Generalise Where Standard Detectors Fail},
  year   = {2026},
  note   = {Preprint}
}
```

## Contact

Monum Hashmi, monumhashmi1@gmail.com
