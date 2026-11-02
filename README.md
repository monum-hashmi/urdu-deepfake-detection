# Detecting Unseen Urdu Speech Deepfakes

Code for the paper **"Detecting Unseen Urdu Speech Deepfakes: Frozen Self-Supervised Features Generalise Where Standard Detectors Fail"** (Monum Hashmi, Department of Artificial Intelligence, University of Management and Technology, Lahore).

We test whether Urdu speech deepfake detectors trained on the public Urdu corpus (Tacotron and VITS fakes) still work on modern text-to-speech systems they have never seen. The protocol is speaker-disjoint, removes dataset shortcuts, adds four unseen TTS systems, and uses two validity controls (codec matching, and genuine out-of-domain Urdu speech from FLEURS).

## Main results

Equal error rate (EER, %), mean ± std over 3 seeds. Lower is better.

| Detector | Seen generators | Unseen modern TTS | FLEURS real vs. all fakes |
|---|---|---|---|
| AASIST-L (ASVspoof weights, no Urdu training) | 40.80 | 65.28 | 45.03 |
| AASIST-L (trained on Urdu) | 20.59 ± 3.36 | 37.48 ± 2.37 | 48.44 ± 7.26 |
| LFCC-LCNN | 13.31 ± 2.20 | 50.57 ± 3.02 | 45.21 ± 3.03 |
| Whisper-small (frozen) + MLP | 8.68 ± 0.70 | 16.46 ± 1.31 | 16.01 ± 2.76 |
| **XLS-R 300M (frozen) + MLP** | **5.36 ± 0.85** | **3.27 ± 1.51** | **4.29 ± 1.82** |

All results, including augmentation runs, leave-one-generator-out runs, per-generator EERs and bootstrap tests, are in [`results/`](results/).

## Protocol at a glance

| Part | What it does |
|---|---|
| Speaker-disjoint split | 11 train / 2 dev / 4 test speakers (4,397 / 799 / 1,598 utterances) |
| Shortcut removal | resample to 16 kHz, trim silence, RMS-normalise, fixed 4 s window |
| Unseen generators | Microsoft Edge neural TTS (`ur-PK-UzmaNeural`, `ur-PK-AsadNeural`), Google TTS (gTTS), Meta MMS-TTS (`facebook/mms-tts-urd-script_arabic`); 488 utterances each |
| Codec control | real test speech passed through the same MP3 round trip (24 kHz, 48 kbps) |
| Out-of-domain control | 267 genuine Urdu utterances from FLEURS (`ur_pk`, dev split) |

## Setup

Tested on Windows 11 with Python 3.11 and an RTX 5060 (8 GB). Everything except AASIST-L training also runs on CPU.

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    Linux/macOS: source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128   # or /whl/cpu
pip install -r requirements.txt
```

`ffmpeg` must be on your PATH (Windows: `winget install Gyan.FFmpeg`).

## Data

Datasets are **not** included in this repository. Download them into `data/`:

```bash
# Urdu deepfake corpus (Munir et al., ACL Findings 2024), CC BY-NC 4.0
hf download CSALT/deepfake_detection_dataset_urdu --repo-type dataset --local-dir data/urdu_deepfake

# FLEURS Urdu dev split (out-of-domain control)
hf download google/fleurs data/ur_pk/audio/dev.tar.gz data/ur_pk/dev.tsv --repo-type dataset --local-dir data/fleurs_raw
mkdir -p data/fleurs_ur && tar -xzf data/fleurs_raw/data/ur_pk/audio/dev.tar.gz -C data/fleurs_ur
```

The generated modern-TTS audio is not redistributed. Rebuild it with `generate.py` from the prompts in `meta/texts.csv`. Online TTS services change over time, so regenerated audio may differ slightly from ours (generated in October 2026).

## Reproducing the experiments

The split files in `meta/` are included, so steps 1–3 are only needed to rebuild everything from scratch.

| Step | Command | Output |
|---|---|---|
| 1. Index the corpus | `python make_manifest.py` | `meta/manifest_original.csv` |
| 2. Transcribe prompts (Whisper-small) | `python transcribe.py` | `meta/texts.csv` |
| 3. Generate unseen TTS + MP3 control | `python generate.py` | `data/new/`, `meta/manifest_new.csv` |
| 4. Speaker-disjoint splits | `python make_splits.py` | `meta/split_*.csv` |
| 5. FLEURS control split | `python make_ood.py` | `meta/split_eval_ood.csv` |
| 6. Preprocess (shortcut removal + MP3 copies) | `python prep.py` | `data/prep/` |
| 7. Cache frozen features | `python extract_ssl.py` and `python extract_whisper.py` | `data/feat_xlsr/`, `data/feat_whisper/` |
| 8. Train XLS-R / Whisper classifiers | `python train_ssl.py --seed 0 [--aug] [--tag whisper] [--train_gen vits\|tacotron]` | `results/eer_*.csv` |
| 9. Train LFCC-LCNN | `python train_lcnn.py --seed 0 [--aug]` | `results/eer_lcnn_*.csv` |
| 10. AASIST-L | `git clone https://github.com/clovaai/aasist.git` then `python train_aasist.py --mode pretrained` and `python train_aasist.py --mode train --seed 0 --bs 8 [--aug]` | `results/eer_aasistL*.csv` |
| 11. Summaries and checks | `python summarize.py`, `python score_ood.py`, `python bootstrap.py`, `python plot_layers.py` | `results/summary.csv`, `results/ood_summary.csv`, `results/bootstrap_*.csv`, `results/fig_layers.*` |

All reported numbers use seeds 0, 1 and 2. `check_files.py` removes half-written feature files after an interrupted run, and `inspect_data.py` prints the corpus structure.

## Licences and credits

- **Code** in this repository: MIT (see `LICENSE`).
- **Urdu deepfake corpus:** CC BY-NC 4.0, from [CSALT-LUMS](https://huggingface.co/datasets/CSALT/deepfake_detection_dataset_urdu) (Munir et al., 2024). Research use only.
- **FLEURS:** CC BY 4.0, Google.
- **AASIST:** official implementation from [clovaai/aasist](https://github.com/clovaai/aasist) (MIT), cloned separately.
- **Models:** XLS-R (`facebook/wav2vec2-xls-r-300m`), Whisper-small (`openai/whisper-small`) and MMS-TTS (`facebook/mms-tts-urd-script_arabic`) are downloaded from Hugging Face under their own licences.
- `meta/texts.csv` contains automatic transcripts of corpus utterances and inherits the corpus licence.

Claude (Anthropic) was used to help write the code in this repository.

## Citation

```bibtex
@misc{hashmi2026urdudeepfake,
  author = {Hashmi, Monum},
  title  = {Detecting Unseen {U}rdu Speech Deepfakes: Frozen Self-Supervised Features Generalise Where Standard Detectors Fail},
  year   = {2026},
  note   = {Preprint}
}
```
