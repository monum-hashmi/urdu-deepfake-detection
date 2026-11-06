"""Record exactly what each online TTS service returns (codec, sample rate, bitrate), to check that
the MP3 control matches the providers' delivery format (review note 15).

Run:  python probe_codecs.py      -> prints a table and saves results/review_codecs.csv
"""
import asyncio
import json
import os
import subprocess
import tempfile

import pandas as pd

TEXT = "یہ ایک آزمائشی جملہ ہے"


async def edge(voice, path):
    import edge_tts
    await edge_tts.Communicate(TEXT, voice).save(path)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=codec_name,sample_rate,channels,bit_rate", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"][0]


def main():
    import edge_tts
    import gtts
    tmp = tempfile.mkdtemp()
    rows = []
    for voice in ["ur-PK-UzmaNeural", "ur-PK-AsadNeural", "ur-IN-GulNeural", "ur-IN-SalmanNeural"]:
        p = os.path.join(tmp, voice + ".mp3")
        try:
            asyncio.run(edge(voice, p))
            rows.append(dict(source=f"Microsoft Edge TTS {voice}", tool=f"edge-tts {edge_tts.__version__}", **probe(p)))
        except Exception as e:
            rows.append(dict(source=f"Microsoft Edge TTS {voice}", error=str(e)[:100]))
    p = os.path.join(tmp, "gtts.mp3")
    try:
        gtts.gTTS(TEXT, lang="ur").save(p)
        rows.append(dict(source="Google TTS (gTTS, lang=ur)", tool=f"gTTS {gtts.__version__}", **probe(p)))
    except Exception as e:
        rows.append(dict(source="Google TTS (gTTS)", error=str(e)[:100]))
    rows.append(dict(source="MMS-TTS (facebook/mms-tts-urd-script_arabic)", tool="transformers VitsModel",
                     codec_name="pcm (no compression)", sample_rate="16000", channels=1, bit_rate="-"))
    rows.append(dict(source="Our MP3 control (real_mp3 / allmp3)", tool="ffmpeg libmp3lame",
                     codec_name="mp3", sample_rate="24000", channels=1, bit_rate="48000"))
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/review_codecs.csv", index=False)
    print(df.to_string(index=False))
    try:
        print("\nffmpeg:", subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0])
    except Exception:
        pass


if __name__ == "__main__":
    main()
