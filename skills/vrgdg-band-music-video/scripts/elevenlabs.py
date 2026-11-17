"""ElevenLabs helper (standard library only): key test, Voice Library search, Scribe word timing, Eleven v4 speech,
sound effects, and a speech-to-text check of generated lines.

usage:
  elevenlabs.py --work W test
  elevenlabs.py --work W voices [--accent british] [--gender female] [--age young] [--search cockney] [--mine]
  elevenlabs.py --work W stt AUDIO OUT.json [--model scribe_v2]
  elevenlabs.py --work W tts LINES.json OUTDIR [--only key1,key2] [--format mp3_44100_128]
  elevenlabs.py --work W sfx SFX.json OUTDIR [--only key1,key2]
  elevenlabs.py --work W check FILE [FILE ...]

The key is line `elevenlabs_key_index` (0-based, blank lines skipped) of `elevenlabs_key_file` in config.json.
It is never printed. TLS uses the Windows certificate store via ssl.create_default_context(): requests/certifi in
ComfyUI's Python fail verification on some machines.
"""
import json
import os
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, parser, wpath  # noqa: E402

BASE = "https://api.elevenlabs.io"
CTX = ssl.create_default_context()
TTS_MODEL = "eleven_v4"  # the user wants the newest model; check https://elevenlabs.io/docs/models for newer ones


def api_key(cfg):
    path = wpath(cfg, cfg["elevenlabs_key_file"])
    keys = [l.strip() for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
    return keys[int(cfg.get("elevenlabs_key_index", 0))]


def request(cfg, method, path, body=None, headers=None, timeout=600):
    hdrs = {"xi-api-key": api_key(cfg)}
    hdrs.update(headers or {})
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers or {})


def post_json(cfg, path, payload, accept="application/json"):
    return request(cfg, "POST", path, json.dumps(payload).encode("utf-8"),
                   {"Content-Type": "application/json", "Accept": accept})


def post_multipart(cfg, path, fields, files):
    boundary = "----vrgdg" + uuid.uuid4().hex
    parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8")
             for k, v in fields.items()]
    for k, (fname, data, ctype) in files.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fname}\"\r\n"
                     f"Content-Type: {ctype}\r\n\r\n".encode("utf-8") + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return request(cfg, "POST", path, b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"})


def as_mp3(path):
    """Scribe uploads are faster as MP3; convert WAV/FLAC to a temporary 160 kbps MP3."""
    if path.lower().endswith(".mp3"):
        return path, False
    out = os.path.join(tempfile.mkdtemp(), "upload.mp3")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-b:a", "160k", out], check=True)
    return out, True


def scribe(cfg, path, model="scribe_v2", events=False):
    upload, temp = as_mp3(path)
    data = open(upload, "rb").read()
    status, body, _ = post_multipart(cfg, "/v1/speech-to-text",
                                     {"model_id": model, "timestamps_granularity": "word",
                                      "tag_audio_events": "true" if events else "false", "diarize": "false"},
                                     {"file": (os.path.basename(upload), data, "audio/mpeg")})
    if temp:
        os.remove(upload)
    if status != 200:
        raise RuntimeError(f"Scribe HTTP {status}: {body[:300]!r}")
    return json.loads(body)


def cmd_test(cfg, _args):
    status, body, _ = request(cfg, "GET", "/v1/user/subscription")
    if status != 200:
        sys.exit(f"key rejected: HTTP {status} {body[:200]!r}")
    sub = json.loads(body)
    print(f"tier {sub.get('tier')}: {sub.get('character_count')}/{sub.get('character_limit')} characters used, "
          f"voice slots {sub.get('voice_slots_used')}/{sub.get('voice_limit')}")
    status, body, _ = request(cfg, "GET", "/v1/models")
    if status == 200:
        print("models:", ", ".join(m["model_id"] for m in json.loads(body)))


def cmd_voices(cfg, args):
    if args.mine:
        status, body, _ = request(cfg, "GET", "/v2/voices?page_size=100")
        for v in json.loads(body).get("voices", []):
            lab = v.get("labels") or {}
            print(v["voice_id"], "|", v["name"][:45], "|", v.get("category"), "|", lab.get("gender"), lab.get("age"), lab.get("accent"))
        return
    query = {"page_size": str(args.limit)}
    for key in ("accent", "gender", "age", "search"):
        if getattr(args, key):
            query[key] = getattr(args, key)
    status, body, _ = request(cfg, "GET", "/v1/shared-voices?" + urllib.parse.urlencode(query))
    if status != 200:
        sys.exit(f"HTTP {status}: {body[:300]!r}")
    for v in json.loads(body).get("voices", []):
        print(v.get("voice_id"), "|", (v.get("name") or "")[:40], "|", v.get("accent"), v.get("age"), v.get("gender"),
              "|", v.get("descriptive"), "|", (v.get("description") or "")[:90].replace("\n", " "))


def cmd_stt(cfg, args):
    data = scribe(cfg, args.audio, args.model)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
    words = [w for w in data.get("words", []) if w.get("type") == "word"]
    print(f"{len(words)} words -> {args.out}")
    print(data.get("text", "")[:3000])


def cmd_tts(cfg, args):
    spec = json.load(open(args.lines, encoding="utf-8"))
    only = set(args.only.split(",")) if args.only else None
    os.makedirs(args.outdir, exist_ok=True)
    for line in spec["lines"]:
        if only and line["key"] not in only:
            continue
        voice = spec["voices"][line["who"]]["id"]
        payload = {"text": line["text"], "model_id": TTS_MODEL,
                   "voice_settings": line.get("voice_settings") or {"stability": 0.5, "similarity_boost": 0.8}}
        status, body, _ = post_json(cfg, f"/v1/text-to-speech/{voice}?output_format={args.format}", payload, "audio/mpeg")
        if status != 200:
            print(line["key"], "ERROR", status, body[:300])
            continue
        path = os.path.join(args.outdir, line["key"] + ".mp3")
        open(path, "wb").write(body)
        print(line["key"], "ok", path)


def cmd_sfx(cfg, args):
    items = json.load(open(args.spec, encoding="utf-8"))
    only = set(args.only.split(",")) if args.only else None
    os.makedirs(args.outdir, exist_ok=True)
    for item in items:
        if only and item["key"] not in only:
            continue
        payload = {"text": item["text"], "duration_seconds": item["seconds"], "prompt_influence": item.get("influence", 0.45)}
        status, body, _ = post_json(cfg, f"/v1/sound-generation?output_format={args.format}", payload, "audio/mpeg")
        if status != 200:
            print(item["key"], "ERROR", status, body[:300])
            continue
        path = os.path.join(args.outdir, item["key"] + ".mp3")
        open(path, "wb").write(body)
        print(item["key"], "ok", path)


def cmd_check(cfg, args):
    """Transcribe generated lines/effects: misread words, tags spoken aloud, or speech inside an effect show up here."""
    for path in args.files:
        data = scribe(cfg, path, events=True)
        words = [w for w in data.get("words", []) if w.get("type") in ("word", "audio_event")]
        span = f"{words[0]['start']:.2f}->{words[-1]['end']:.2f}s" if words else "silent"
        print(f"{os.path.basename(path):24s} {span:16s} {data.get('text', '')}")


def main():
    p = parser(__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    v = sub.add_parser("voices")
    for key in ("accent", "gender", "age", "search"):
        v.add_argument("--" + key)
    v.add_argument("--limit", type=int, default=25)
    v.add_argument("--mine", action="store_true")
    s = sub.add_parser("stt")
    s.add_argument("audio")
    s.add_argument("out")
    s.add_argument("--model", default="scribe_v2")
    t = sub.add_parser("tts")
    t.add_argument("lines")
    t.add_argument("outdir")
    t.add_argument("--only")
    t.add_argument("--format", default="mp3_44100_128", help="Starter tier refuses mp3_44100_192 (HTTP 403)")
    f = sub.add_parser("sfx")
    f.add_argument("spec")
    f.add_argument("outdir")
    f.add_argument("--only")
    f.add_argument("--format", default="mp3_44100_128")
    c = sub.add_parser("check")
    c.add_argument("files", nargs="+")
    args = p.parse_args()
    cfg = load_config(args.work)
    {"test": cmd_test, "voices": cmd_voices, "stt": cmd_stt, "tts": cmd_tts, "sfx": cmd_sfx, "check": cmd_check}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
