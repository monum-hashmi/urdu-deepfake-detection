import os, time, asyncio, subprocess, argparse
import pandas as pd

OUT = r"data\new"
TMP = r"data\tmp"

def to_wav16k(src, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-ac", "1", "-ar", "16000", dst], check=True)

def mp3_roundtrip(src, dst):
    tmp = os.path.join(TMP, "rt.mp3")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-ac", "1", "-ar", "24000", "-b:a", "48k", tmp], check=True)
    to_wav16k(tmp, dst)

async def _edge(text, voice, mp3):
    import edge_tts
    await edge_tts.Communicate(text, voice).save(mp3)

def gen_edge(text, voice, dst):
    mp3 = os.path.join(TMP, "edge.mp3")
    asyncio.run(_edge(text, voice, mp3))
    to_wav16k(mp3, dst)
    time.sleep(0.3)

def gen_gtts(text, dst):
    from gtts import gTTS
    mp3 = os.path.join(TMP, "gtts.mp3")
    gTTS(text, lang="ur").save(mp3)
    to_wav16k(mp3, dst)
    time.sleep(1.0)

_mms = None
def gen_mms(text, dst, idx):
    global _mms
    import torch, soundfile as sf
    if _mms is None:
        from transformers import VitsModel, AutoTokenizer
        name = "facebook/mms-tts-urd-script_arabic"
        _mms = (AutoTokenizer.from_pretrained(name), VitsModel.from_pretrained(name).eval())
    tok, model = _mms
    torch.manual_seed(idx)
    with torch.no_grad():
        wav = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
    tmpw = os.path.join(TMP, "mms.wav")
    sf.write(tmpw, wav, model.config.sampling_rate)
    to_wav16k(tmpw, dst)

ap = argparse.ArgumentParser()
ap.add_argument("--gens", nargs="+",
                default=["real_mp3", "edge_uzma", "edge_asad", "gtts", "mms"])
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

df = pd.read_csv("meta/texts.csv", encoding="utf-8-sig")
df = df[df.text.fillna("").str.len() > 3].reset_index(drop=True)
if args.limit:
    df = df.head(args.limit)
os.makedirs(TMP, exist_ok=True)
failures = []

for gen in args.gens:
    os.makedirs(os.path.join(OUT, gen), exist_ok=True)
    ok = fail = 0
    for i, r in df.iterrows():
        dst = os.path.join(OUT, gen, f"{r.speaker}_{r.utt}.wav")
        if os.path.exists(dst):
            ok += 1
            continue
        err = ""
        for attempt in range(3):
            try:
                if gen == "edge_uzma":
                    gen_edge(r.text, "ur-PK-UzmaNeural", dst)
                elif gen == "edge_asad":
                    gen_edge(r.text, "ur-PK-AsadNeural", dst)
                elif gen == "gtts":
                    gen_gtts(r.text, dst)
                elif gen == "mms":
                    gen_mms(r.text, dst, i)
                elif gen == "real_mp3":
                    mp3_roundtrip(r.src_path, dst)
                ok += 1
                break
            except Exception as e:
                err = str(e)[:200]
                time.sleep(3 * (attempt + 1))
        else:
            fail += 1
            failures.append(dict(gen=gen, speaker=r.speaker, utt=r.utt, error=err))
        if (i + 1) % 50 == 0:
            print(f"{gen}: {i+1}/{len(df)}  ok={ok}  fail={fail}")
    print(f"== {gen} done: ok={ok} fail={fail}")

rows = []
for gen in os.listdir(OUT):
    for f in os.listdir(os.path.join(OUT, gen)):
        spk, utt = f[:-4].rsplit("_", 1)
        rows.append(dict(path=os.path.join(OUT, gen, f),
                         label="bonafide" if gen == "real_mp3" else "spoof",
                         generator=gen, speaker=spk, utt=utt))
pd.DataFrame(rows).to_csv("meta/manifest_new.csv", index=False)
pd.DataFrame(failures).to_csv("meta/gen_failures.csv", index=False)
print("Saved meta/manifest_new.csv  | failures:", len(failures))