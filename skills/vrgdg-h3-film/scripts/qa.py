"""QA for rendered scenes (run with the qa_python from settings.json, usually ComfyUI's own Python).

  python qa.py <project> <scene> [<scene> ...] [--cpu]   scenes: Whisper vs script, spoken tags, voice sex (pitch),
                                                          line running into the clip end, 8-frame review sheet
  python qa.py <project> --music [--cpu]                  score files in audio/score: flags accidental vocals
  python qa.py <project> --fetch-whisper small            download one Whisper model (only when the user agrees)

Whisper uses the largest model already downloaded (large-v3, medium, small, base); none -> transcripts are skipped.
Output: printed report, logs/qa/qa_report.json, logs/qa/review_<n>.jpg (view these images!).
"""
import difflib
import json
import os
import re
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ffprobe_duration, load_settings, project_dir  # noqa: E402

TAG_WORDS = {"pause", "softer", "soffer", "catches", "pants", "whisper", "breath", "inhale", "exhale", "sighs", "chuckle",
             "laughs", "gasp", "stutter", "humming", "coughs", "clears", "throat"}
WHISPER_ORDER = ["large-v3", "large-v2", "large", "medium", "medium.en", "small", "small.en", "base", "base.en"]


def whisper_model(device):
    try:
        import whisper
    except ImportError:
        print("warning: whisper is not installed in this Python; transcripts are skipped")
        return None, None
    cache = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    have = [n for n in WHISPER_ORDER if os.path.isfile(os.path.join(cache, f"{n}.pt"))]
    if not have:
        print("warning: no Whisper model downloaded; ask the user, then run qa.py <project> --fetch-whisper small")
        return None, None
    return whisper.load_model(have[0], device=device), have[0]


def words(text):
    text = re.sub(r"<[^>]+>", " ", text).lower()
    return re.sub(r"[^a-z0-9' ]", " ", text).split()


def read_wav(path):
    with wave.open(path) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return data, w.getframerate()


def cepstral_f0(seg, sr):
    """Pitch from harmonic spacing: still works on thin phone/radio/tape audio where the fundamental is filtered out."""
    n = 1024
    if len(seg) < n * 2:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(seg, n)[::256] * np.hanning(n)
    rms = np.sqrt((frames ** 2).mean(1))
    f0 = []
    for f in frames[rms > np.percentile(rms, 60)]:
        c = np.fft.irfft(np.log(np.abs(np.fft.rfft(f)) + 1e-9))
        k = int(sr / 320) + int(np.argmax(c[int(sr / 320):int(sr / 65)]))
        f0.append(sr / k)
    return float(np.median(f0)) if f0 else None


def db(seg):
    return 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else -120.0


def review_sheet(ffmpeg, clip, title, out, n=8, w=240):
    dur = ffprobe_duration(ffmpeg, clip)
    tiles = []
    for k in range(n):
        t = dur * (k + 0.5) / n
        png = out + f".{k}.png"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", clip, "-frames:v", "1", "-vf", f"scale={w}:-1", png], check=True)
        im = Image.open(png).convert("RGB")
        ImageDraw.Draw(im).text((3, 2), f"{t:.1f}s", fill=(255, 255, 0))
        tiles.append(im)
        os.remove(png)
    sheet = Image.new("RGB", (w * n, tiles[0].height + 26), (20, 20, 20))
    for k, im in enumerate(tiles):
        sheet.paste(im, (k * w, 26))
    ImageDraw.Draw(sheet).text((4, 6), title[:260], fill=(255, 255, 255))
    sheet.save(out, quality=90)


def scene_qa(project, settings, sp, scene, model):
    ffmpeg, num = settings["ffmpeg"], scene["scene_number"]
    clip = os.path.join(project, "video_clips", f"scene_{num:03d}.mp4")
    if not os.path.isfile(clip):
        return None
    wav = os.path.join(project, "logs", "qa", f"scene_{num:03d}.wav")
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", clip, "-ac", "1", "-ar", "16000", wav], check=True)
    y, sr = read_wav(wav)
    dur = len(y) / sr
    order = {v: k for k, v in scene["speakers"].items()}
    expected = [(order[sid], line) for sid, line in re.findall(r"\((S\d+)\)[^<]*?<d>\[English\] (.*?)</d>", scene["prompt"])]
    rep = {"scene": num, "duration": round(dur, 2), "lines": [], "flags": []}
    heard, hw = [], []
    if model:
        res = model.transcribe(wav, language="en", word_timestamps=True)
        rep["heard"] = res["text"].strip()
        heard = [(w["word"], w["start"], w["end"]) for s in res["segments"] for w in s.get("words", [])
                 if s.get("no_speech_prob", 0) < 0.6]
        hw = [(words(w) or [""])[0] for w, _, _ in heard]
        if not expected and heard:
            rep["flags"].append(f"speech heard in a silent scene: '{rep['heard']}' (check by ear / re-trim)")
    for key, line in expected:
        entry = {"speaker": key, "line": line}
        want = words(line)
        if hw and want:
            sm = difflib.SequenceMatcher(None, hw, want)
            blocks = [b for b in sm.get_matching_blocks() if b.size]
            entry["match"] = round(sum(b.size for b in blocks) / len(want), 2)
            if blocks:
                entry["start"], entry["end"] = round(heard[blocks[0].a][1], 2), round(heard[blocks[-1].a + blocks[-1].size - 1][2], 2)
                f0 = cepstral_f0(y[int(entry["start"] * sr):int((entry["end"] + 0.1) * sr)], sr)
                entry["pitch_hz"] = round(f0) if f0 else None
                want_sex = sp["sex"].get(key)
                if f0 and want_sex == "M" and f0 > 165:
                    rep["flags"].append(f"{key} should be male but pitches at {f0:.0f} Hz (female range)")
                if f0 and want_sex == "F" and f0 < 140:
                    rep["flags"].append(f"{key} should be female but pitches at {f0:.0f} Hz (male range)")
                if dur - entry["end"] < 0.25 and db(y[-int(0.15 * sr):]) > -38:
                    rep["flags"].append(f"{key}'s line runs into the last frames: re-trim from the raw take (retrim.py)")
            if entry["match"] < 0.85:
                rep["flags"].append(f"{key}: heard '{rep.get('heard', '')}' vs '{line}' (numbers may just be written as digits)")
        rep["lines"].append(entry)
    if hw:
        exp_words = {w for _, line in expected for w in words(line)}
        spoken = sorted({w for w in hw if w in TAG_WORDS and w not in exp_words})
        if spoken:
            rep["flags"].append(f"tag words spoken aloud: {spoken}")
    sheet = os.path.join(project, "logs", "qa", f"review_{num:03d}.jpg")
    lines = " / ".join(f"{k}: {l}" for k, l in expected) or "(no dialogue)"
    review_sheet(ffmpeg, clip, f"SCENE {num}  {scene['cast']}  |  {lines}", sheet)
    rep["review_sheet"] = sheet
    return rep


def main():
    project, args = project_dir()
    settings = load_settings(project)
    device = "cpu" if "--cpu" in args else "cuda"
    if "--fetch-whisper" in args:
        import whisper
        name = args[args.index("--fetch-whisper") + 1]
        whisper.load_model(name, device="cpu")
        print(f"downloaded Whisper {name}")
        return
    model, name = whisper_model(device)
    if name:
        print(f"Whisper model: {name} on {device}")
    report_path = os.path.join(project, "logs", "qa", "qa_report.json")
    report = json.load(open(report_path, encoding="utf-8")) if os.path.isfile(report_path) else {}
    if "--music" in args:
        folder = os.path.join(project, "audio", "score")
        for f in sorted(os.listdir(folder)):
            if model and f.lower().endswith((".mp3", ".wav", ".flac")):
                res = model.transcribe(os.path.join(folder, f), language="en", no_speech_threshold=0.5)
                sung = [s["text"].strip() for s in res["segments"] if s["no_speech_prob"] < 0.5 and len(s["text"].strip()) > 3]
                print(f"{f}: {'VOCALS? ' + ' | '.join(sung[:4]) if sung else 'instrumental'}")
        return
    sp = json.load(open(os.path.join(project, "screenplay.json"), encoding="utf-8"))
    board = {s["scene_number"]: s for s in json.load(open(os.path.join(project, "storyboard", "scenes.json"), encoding="utf-8"))["scenes"]}
    for num in [int(a) for a in args if a.isdigit()]:
        rep = scene_qa(project, settings, sp, board[num], model)
        if rep is None:
            print(f"scene {num}: not rendered")
            continue
        report[str(num)] = rep
        print(f"scene {num:03d} ({rep['duration']} s) heard: {rep.get('heard', '(no transcript)')}")
        for line in rep["lines"]:
            print(f"    {line['speaker']}: match {line.get('match')} pitch {line.get('pitch_hz')} Hz "
                  f"{line.get('start')}-{line.get('end')} s | {line['line']}")
        for flag in rep["flags"]:
            print(f"    CHECK: {flag}")
        print(f"    review sheet: {rep['review_sheet']}")
    json.dump(report, open(report_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
