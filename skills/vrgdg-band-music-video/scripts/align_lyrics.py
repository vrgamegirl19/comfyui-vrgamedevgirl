"""Time the official lyrics word by word from an ElevenLabs Scribe transcript of the vocal stem.

usage: align_lyrics.py --work W --words audio/stt/vocals_scribe.json [--lyrics source/lyrics.txt] [--offset 16]
       -> lyrics_timed.json in the work folder, plus a printed table of lines with gaps

The official text wins over Scribe's spelling ("weight" -> "wait", "raw" -> "rude"), and the timing comes from
Scribe. Words Scribe heard that the lyrics file does not contain (a repeated line, an "oh") are listed as
extra_sung. Add real repeats to lyrics.txt and run again, because a singer on screen must have words for every
sung moment. --offset is the intro length: the times in lyrics_timed.json are full-track times.
"""
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, parser, wpath, write_json  # noqa: E402


def norm(word):
    return re.sub(r"[^a-z0-9]", "", word.lower().replace("’", "'").replace("'", ""))


def parse_lyrics(text):
    lines, section = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            continue
        lines.append({"text": line, "section": section})
    return lines


def main():
    p = parser(__doc__)
    p.add_argument("--words", required=True, help="Scribe JSON of the vocal stem (elevenlabs.py stt)")
    p.add_argument("--lyrics", default=None)
    p.add_argument("--offset", type=float, default=None, help="intro seconds (default: config intro_seconds)")
    p.add_argument("--out", default="lyrics_timed.json")
    args = p.parse_args()
    cfg = load_config(args.work)
    offset = float(cfg.get("intro_seconds", 0.0) if args.offset is None else args.offset)
    lyrics = parse_lyrics(open(wpath(cfg, args.lyrics or cfg["lyrics"]), encoding="utf-8").read())
    scribe = [w for w in json.load(open(wpath(cfg, args.words), encoding="utf-8"))["words"] if w.get("type") == "word"]

    official = []  # (line index, original token)
    for li, line in enumerate(lyrics):
        for token in line["text"].split():
            if norm(token):
                official.append((li, token))
    a = [norm(t) for _, t in official]
    b = [norm(w["text"]) for w in scribe]
    times = [None] * len(official)
    extras = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                times[i1 + k] = (scribe[j1 + k]["start"], scribe[j1 + k]["end"])
        elif tag == "replace":
            for k in range(i2 - i1):  # unequal block: spread the official words over the heard ones
                j = j1 + round(k * (j2 - j1 - 1) / max(i2 - i1 - 1, 1))
                times[i1 + k] = (scribe[j]["start"], scribe[j]["end"])
        elif tag == "insert":
            extras.append({"start": scribe[j1]["start"] + offset, "end": scribe[j2 - 1]["end"] + offset,
                           "text": " ".join(w["text"] for w in scribe[j1:j2])})

    # Words Scribe missed: interpolate between their timed neighbours.
    for i, t in enumerate(times):
        if t is None:
            prev_end = next((times[k][1] for k in range(i - 1, -1, -1) if times[k]), 0.0)
            next_start = next((times[k][0] for k in range(i + 1, len(times)) if times[k]), prev_end + 0.4)
            times[i] = (prev_end, max(prev_end, min(next_start, prev_end + 0.4)))

    words, out_lines = [], []
    for li, line in enumerate(lyrics):
        idx = [i for i, (l, _) in enumerate(official) if l == li]
        if not idx:
            continue
        for i in idx:
            words.append({"start": round(times[i][0] + offset, 3), "end": round(times[i][1] + offset, 3),
                          "word": official[i][1], "line": li})
        out_lines.append({"start": round(times[idx[0]][0] + offset, 3), "end": round(times[idx[-1]][1] + offset, 3),
                          "text": line["text"], "section": line["section"]})
    write_json(wpath(cfg, args.out), {"offset": offset, "lines": out_lines, "words": words, "extra_sung": extras})

    print(f"{len(out_lines)} lines, {len(words)} words (full-track times, offset {offset:+.2f} s) -> {args.out}")
    prev = offset
    for line in out_lines:
        print(f"  {line['start']:7.2f} -> {line['end']:7.2f}  (gap {line['start'] - prev:5.2f})  [{line['section']}] {line['text']}")
        prev = line["end"]
    if extras:
        print("\nSung but not in the lyrics file (repeats or ad-libs; add real repeats to lyrics.txt and rerun):")
        for e in extras:
            print(f"  {e['start']:7.2f} -> {e['end']:7.2f}  {e['text']}")


if __name__ == "__main__":
    main()
