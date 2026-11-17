"""Word timing WITHOUT ElevenLabs: the Video Builder's own stems + transcribe (Stable-ts / Whisper large-v3 in
ComfyUI), saved in the same JSON shape as `elevenlabs.py stt`, so align_lyrics.py and everything after it work
unchanged.

usage: builder_transcribe.py --work W [--vocals STEM.wav] [--lyrics lyrics.txt] [--out audio/stt/vocals_builder.json]
                             [--stem-model htdemucs|htdemucs_ft|htdemucs_6s] [--model large-v3]

Steps (each a GPU job through the Agent API; run only when ComfyUI is idle):
  1. No --vocals and no lead_stem_path in config: POST /projects/{pid}/audio/stems splits the project song with Demucs
     (the Audio Mask splitter). Writes audio/builder_stems.json with vocals_path, no_vocals_path and every stem, and
     fills config song_no_vocals / lead_stem_path when they are empty. With the user's own stems, pass the lead vocal
     stem as --vocals instead (it is cleaner than a Demucs split).
  2. PUT /projects/{pid}/lyrics with the official lyrics (the reference text the aligner follows).
  3. POST /projects/{pid}/lyrics/align on the vocal stem, segment_mode reference_lines: timed lines with word times,
     saved by the Builder in project_context/timestamped_lyrics.json (the same data the Builder's Transcribe uses).
  4. Every word -> {"text","start","end","type":"word"} in --out.
Then: align_lyrics.py --work W --words <out>. Check the last lines and held notes against the vocal stem
(vocal_activity.py): aligners squeeze long held final words, as both reference videos needed a hand fix at the end.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import Api, load_config, parser, read_json, save_config, wpath, write_json  # noqa: E402


def result_of(job):
    data = job.get("result") if isinstance(job, dict) else None
    if isinstance(data, dict) and isinstance(data.get("result"), dict):
        data = data["result"]
    return data or {}


def main():
    p = parser(__doc__)
    p.add_argument("--vocals", default=None, help="lead vocal stem (default: config lead_stem_path, else a Demucs split)")
    p.add_argument("--lyrics", default=None, help="official lyrics (default: config lyrics)")
    p.add_argument("--out", default="audio/stt/vocals_builder.json")
    p.add_argument("--stem-model", default="htdemucs")
    p.add_argument("--model", default="large-v3")
    a = p.parse_args()
    cfg = load_config(a.work)
    api = Api(cfg, a.project)

    vocals = a.vocals or (wpath(cfg, cfg["lead_stem_path"]) if cfg.get("lead_stem_path") else None)
    if not vocals or not os.path.isfile(vocals):
        print("[VRGDG MV] splitting the song into stems with the Builder (POST audio/stems)...", flush=True)
        api.wait_idle()
        job = api.run_job("POST", "/projects/{pid}/audio/stems", {"model_name": a.stem_model}, poll=5.0)
        res = result_of(job)
        if job.get("status") != "succeeded" or not res.get("vocals_path"):
            sys.exit(f"[VRGDG MV] stem split failed: {json.dumps(job)[:800]}")
        write_json(wpath(cfg, "audio/builder_stems.json"), res)
        vocals = res["vocals_path"]
        if not cfg.get("lead_stem_path"):
            cfg["lead_stem_path"] = vocals
        if not cfg.get("song_no_vocals") and res.get("no_vocals_path"):
            cfg["song_no_vocals"] = res["no_vocals_path"]
        save_config(cfg)
        print(f"  vocals: {vocals}\n  no vocals: {res.get('no_vocals_path')}\n  stems: {list((res.get('stems') or {}))}")

    lyrics_path = wpath(cfg, a.lyrics or cfg.get("lyrics") or "source/lyrics.txt")
    lyrics = open(lyrics_path, encoding="utf-8").read().strip()
    status, env = api.mutate("PUT", "/projects/{pid}/lyrics", {"lyrics_text": lyrics})
    if not env.get("ok"):
        sys.exit(f"[VRGDG MV] PUT lyrics failed: {status} {json.dumps(env)[:500]}")

    print(f"[VRGDG MV] timing the lyrics on {vocals} (POST lyrics/align, Whisper {a.model})...", flush=True)
    api.wait_idle()
    body = {"audio_path": vocals, "reference_lyrics": lyrics, "segment_mode": "reference_lines", "language": "english",
            "model_name": a.model, "include_instrumental_gaps": True}
    job = api.run_job("POST", "/projects/{pid}/lyrics/align", body, poll=5.0)
    res = result_of(job)
    payload_path = res.get("payload_path") or os.path.join(cfg["project_folder"], "project_context", "timestamped_lyrics.json")
    payload = read_json(payload_path)
    if job.get("status") != "succeeded" or not payload:
        sys.exit(f"[VRGDG MV] lyrics/align failed: {json.dumps(job)[:800]}")

    words, no_words = [], 0
    for seg in payload.get("segments") or []:
        if seg.get("type") == "instrumental":
            continue
        seg_words = seg.get("words") or []
        if not seg_words:
            no_words += 1
        for w in seg_words:
            text = str(w.get("text") or "").strip()
            if text:
                words.append({"text": text, "start": round(float(w["start"]), 3), "end": round(float(w["end"]), 3),
                              "type": "word"})
    words.sort(key=lambda w: w["start"])
    out = wpath(cfg, a.out)
    write_json(out, {"text": " ".join(w["text"] for w in words), "words": words,
                     "source": f"vrgdg lyrics/align ({a.model}) on {os.path.basename(vocals)}"})
    print(f"[VRGDG MV] {len(words)} words -> {out}" + (f"; {no_words} sung lines came back without word times" if no_words else ""))
    print(f"next: align_lyrics.py --work {cfg['_work']} --words {a.out}")


if __name__ == "__main__":
    main()
