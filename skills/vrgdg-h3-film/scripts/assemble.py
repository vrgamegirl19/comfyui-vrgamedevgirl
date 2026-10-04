"""Assemble the film (run with qa_python: the title and end cards need Pillow).

  python assemble.py <project>
  1. Builder stitch_scene_videos: scenes in play order, placed by their actual clip lengths, keeping each scene's own
     H3 audio (use_embedded_scene_audio).
  2. Score bed from score.json "placement" (cue file, from_scene/to_scene, offset, fades), sidechain-ducked under the
     dialogue. Scenes outside every placement play on their own sound only.
  3. Title overlay (score.json "title_card": over_scene, at, seconds), end card ("end_card": lines), fades,
     loudness gained to settings loudness_lufs with a -1 dBTP limiter.
  -> final/<TITLE>.mp4 and final/<TITLE>_builder_stitch.mp4 (clean, no score or titles)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Comfy, ffprobe_duration, load_settings, project_dir  # noqa: E402

FONTS = [r"C:\Windows\Fonts\GARA.TTF", r"C:\Windows\Fonts\georgia.ttf", "/System/Library/Fonts/Supplemental/Georgia.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]
CREAM = (246, 238, 222, 255)


def font(size):
    for path in FONTS:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def centered(img, y, text, f, fill):
    d = ImageDraw.Draw(img)
    d.text(((img.width - d.textlength(text, font=f)) / 2, y), text, font=f, fill=fill)


def cards(project, title, end_lines, w, h):
    big, small = font(int(h * 0.085)), font(int(h * 0.032))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    band = ImageDraw.Draw(overlay)
    for y in range(int(h * 0.30), int(h * 0.62)):   # soft dark band so the title reads over any shot
        band.line([(0, y), (w, y)], fill=(0, 0, 0, int(140 * (1 - abs(y - h * 0.46) / (h * 0.16)))))
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    spaced = " ".join(title.upper())
    centered(glow, int(h * 0.40), spaced, big, (255, 220, 170, 160))
    overlay = Image.alpha_composite(overlay, glow.filter(ImageFilter.GaussianBlur(h * 0.012)))
    centered(overlay, int(h * 0.40), spaced, big, CREAM)
    overlay.save(os.path.join(project, "logs", "title_card.png"))
    end = Image.new("RGBA", (w, h), (8, 7, 6, 255))
    y = int(h * 0.36)
    for i, line in enumerate(end_lines):
        f = big if i == 0 else small
        centered(end, y, line if i else " ".join(line.upper()), f, CREAM if i < 2 else (160, 150, 135, 255))
        y += int(h * (0.13 if i == 0 else 0.055))
    end.convert("RGB").save(os.path.join(project, "logs", "end_card.png"))


def ff(ffmpeg, *args):
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", *args], check=True, stdin=subprocess.DEVNULL)


def audio_format(ffmpeg, path):
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace").stderr
    m = re.search(r"Audio: .*?(\d+) Hz, (mono|stereo|[\d.]+)", probe)
    return m.groups() if m else None


def conform_audio(ffmpeg, project, paths):
    """The Builder stitch breaks its audio timeline when clips mix sample rates (a hand-edited clip saved at 48 kHz
    among H3's 32 kHz clips silenced everything after it), so re-encode the odd ones out into copies first."""
    formats = [audio_format(ffmpeg, p) for p in paths]
    common = max(set(f for f in formats if f), key=formats.count)
    out_dir = os.path.join(project, "builder_project", "conformed")
    fixed = []
    for p, f in zip(paths, formats):
        if f == common:
            fixed.append(p)
            continue
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, os.path.basename(p))
        v, rate = f"{ffprobe_duration(ffmpeg, p):.3f}", common[0]
        enc = ["-t", v, "-c:v", "copy", "-ar", rate, "-ac", "2", "-c:a", "aac", "-b:a", "192k", dst]
        if f:
            ff(ffmpeg, "-i", p, "-map", "0:v", "-map", "0:a", "-af", f"aresample={rate},apad", *enc)
        else:   # a clip with no audio track gets silence
            ff(ffmpeg, "-i", p, "-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=stereo", "-map", "0:v", "-map", "1:a", *enc)
        print(f"conformed {os.path.basename(p)} audio {f} -> {common}")
        fixed.append(dst)
    return fixed


def build_dialogue(ffmpeg, project, paths, timing, total, sr=48000):
    """Each clip's own audio, cut or padded to exactly its picture length and butted together sample-accurately.
    The Builder stitch's audio drifted about 5 ms per scene (AAC padding), 0.26 s late by the end of a 46-scene film."""
    tmp = os.path.join(project, "audio", "_clip.wav")
    out = os.path.join(project, "audio", "dialogue.wav")
    with wave.open(out, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        for p, item in zip(paths, timing):
            n = round(item["end"] * sr) - round(item["start"] * sr)
            ff(ffmpeg, "-i", p, "-vn", "-af", f"aresample={sr},apad", "-t", f"{n / sr + 0.05:.4f}", "-ac", "2",
               "-c:a", "pcm_s16le", tmp)
            with wave.open(tmp) as r:
                pcm = np.frombuffer(r.readframes(r.getnframes()), np.int16).reshape(-1, 2)
            pcm = np.pad(pcm[:n], ((0, max(0, n - len(pcm))), (0, 0)))
            w.writeframes(pcm.tobytes())
        w.writeframes(np.zeros((round(total * sr) - round(timing[-1]["end"] * sr), 2), np.int16).tobytes())
    os.remove(tmp)
    return out


def check_sync(ffmpeg, film_path, paths, starts, order, sr=8000):
    """Find each sampled scene's own audio inside the finished film; returns the worst offset in ms."""
    def pcm(path, ss=None, t=None):
        cmd = [ffmpeg, "-v", "error"] + (["-ss", f"{ss:.3f}"] if ss is not None else []) + ["-i", path] +               (["-t", f"{t}"] if t else []) + ["-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32)
    worst = 0.0
    for n, p in list(zip(order, paths))[1::max(1, len(paths) // 8)]:
        clip = pcm(p)[: sr * 4]
        seg = pcm(film_path, ss=max(0.0, starts[n] - 1.0), t=6)
        if clip.std() < 1e-4 or len(seg) < len(clip):
            continue
        c = np.correlate(seg - seg.mean(), clip - clip.mean(), mode="valid")
        worst = max(worst, abs((np.argmax(c) / sr - min(1.0, starts[n])) * 1000))
    return worst


def main():
    project, _ = project_dir()
    settings = load_settings(project)
    ffmpeg, fps, W, H = settings["ffmpeg"], settings["fps"], settings["final_width"], settings["final_height"]
    sp = json.load(open(os.path.join(project, "screenplay.json"), encoding="utf-8"))
    score = json.load(open(os.path.join(project, "score.json"), encoding="utf-8"))
    board = [s["scene_number"] for s in json.load(open(os.path.join(project, "storyboard", "scenes.json"), encoding="utf-8"))["scenes"]]
    order = score.get("play_order") or board
    paths = [os.path.join(project, "video_clips", f"scene_{n:03d}.mp4") for n in order]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"missing scenes: {missing}")
    paths = conform_audio(ffmpeg, project, paths)
    starts, ends, timing, t = {}, {}, [], 0.0
    for n, p in zip(order, paths):
        frames = round(ffprobe_duration(ffmpeg, p) * fps)
        starts[n], ends[n] = t, t + frames / fps
        timing.append({"start": t, "end": t + frames / fps})
        t += frames / fps
    film, end_card = t, float(score.get("end_card_seconds", 8.0))
    total = film + end_card
    comfy = Comfy(settings["comfy_url"], os.path.join(project, "logs", "run.log"))
    stitched = comfy.post("/vrgdg/workflow_runner/stitch_scene_videos", {
        "project_folder": os.path.join(project, "builder_project"), "scene_paths": paths, "use_embedded_scene_audio": True,
        "scene_timing_items": timing, "timeline_fps": fps, "width": W, "height": H, "output_prefix": "FILM_STITCHED",
    }, timeout=3600)
    if not stitched.get("ok"):
        raise SystemExit(f"stitch failed: {stitched.get('error')}")
    safe = re.sub(r"[^A-Za-z0-9]+", "_", sp["title"]).strip("_") or "FILM"
    raw = os.path.join(project, "final", f"{safe}_builder_stitch.mp4")
    shutil.move(stitched["final_video_path"], raw)
    dialogue = build_dialogue(ffmpeg, project, paths, timing, total)
    synced = raw + ".tmp.mp4"
    ff(ffmpeg, "-i", raw, "-i", dialogue, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
       "-t", f"{film:.3f}", synced)
    os.replace(synced, raw)
    comfy.log(f"stitched {raw} ({film:.2f}s), audio rebuilt from the clips")

    # score bed
    inputs, chains = [], []
    for i, cue in enumerate(score.get("placement", [])):
        src = os.path.join(project, cue["file"])
        if not os.path.isfile(src):
            raise SystemExit(f"score file missing: {src}")
        start = starts[cue["from_scene"]] + float(cue.get("offset", 0))
        stop = ends[cue["to_scene"]] if cue.get("to_scene") != "end" else total
        length = min(stop - start, ffprobe_duration(ffmpeg, src) - float(cue.get("trim_start", 0)))
        fi, fo = float(cue.get("fade_in", 2.0)), float(cue.get("fade_out", 3.0))
        inputs += ["-i", src]
        chains.append(f"[{i}:a]aresample=48000,atrim={float(cue.get('trim_start', 0)):.3f}:{float(cue.get('trim_start', 0)) + length:.3f},"
                      f"asetpts=PTS-STARTPTS,afade=t=in:d={fi},afade=t=out:st={max(0, length - fo):.3f}:d={fo},"
                      f"adelay={start * 1000:.0f}:all=1,apad=whole_dur={total:.3f}[c{i}]")
    master = os.path.join(project, "audio", "MASTER.wav")
    premix = os.path.join(project, "audio", "premix.wav")
    if chains:
        bed = os.path.join(project, "audio", "score_bed.wav")
        mix = "".join(f"[c{i}]" for i in range(len(chains)))
        ff(ffmpeg, *inputs, "-filter_complex", ";".join(chains) + f";{mix}amix=inputs={len(chains)}:normalize=0,"
           f"volume={score.get('music_db', -16)}dB,atrim=0:{total:.3f}[out]", "-map", "[out]", "-ac", "2", "-c:a", "pcm_s16le", bed)
        ff(ffmpeg, "-i", dialogue, "-i", bed, "-filter_complex",
           f"[0:a]aresample=48000,apad=whole_dur={total:.3f},asplit=2[dia][key];"
           "[1:a][key]sidechaincompress=threshold=0.02:ratio=4:attack=15:release=450:makeup=1[duck];"
           "[dia][duck]amix=inputs=2:normalize=0[out]", "-map", "[out]", "-t", f"{total:.3f}", "-ac", "2", "-c:a", "pcm_s16le", premix)
    else:
        ff(ffmpeg, "-i", dialogue, "-af", f"aresample=48000,apad=whole_dur={total:.3f}", "-ac", "2", "-c:a", "pcm_s16le", premix)
    stats = subprocess.run([ffmpeg, "-hide_banner", "-i", premix, "-af", "ebur128", "-f", "null", "-"],
                           capture_output=True, text=True, errors="replace").stderr
    loud = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", stats)[-1])
    target = settings.get("loudness_lufs", -14.0)
    ff(ffmpeg, "-i", premix, "-af", f"volume={target - loud:.2f}dB,alimiter=limit=0.891:attack=5:release=80:level=false",
       "-c:a", "pcm_s16le", master)

    # picture
    tc = score.get("title_card", {"over_scene": order[0], "at": 1.0, "seconds": 5.0})
    end_lines = score.get("end_card", {}).get("lines") or [sp["title"], "A short film by Claude",
                                                             "Made with MiniMax H3 and the VRGDG Video Builder for ComfyUI"]
    cards(project, sp["title"], end_lines, W, H)
    t0, secs = starts[tc["over_scene"]] + float(tc.get("at", 1.0)), float(tc.get("seconds", 5.0))
    final = os.path.join(project, "final", f"{safe}.mp4")
    ff(ffmpeg, "-i", raw, "-loop", "1", "-framerate", str(fps), "-t", f"{secs:.3f}", "-i", os.path.join(project, "logs", "title_card.png"),
       "-loop", "1", "-framerate", str(fps), "-t", f"{end_card}", "-i", os.path.join(project, "logs", "end_card.png"), "-i", master,
       "-filter_complex",
       f"[1:v]format=rgba,fade=t=in:st=0:d=1.0:alpha=1,fade=t=out:st={max(0, secs - 1.0):.3f}:d=1.0:alpha=1,setpts=PTS+{t0:.3f}/TB[t];"
       f"[0:v]fps={fps},format=yuv420p[v0];[v0][t]overlay=0:0:eof_action=pass,fade=t=in:st=0:d=1.0,"
       f"fade=t=out:st={film - 0.8:.3f}:d=0.8,setsar=1[main];"
       f"[2:v]fps={fps},format=yuv420p,fade=t=in:st=0:d=1.5,fade=t=out:st={end_card - 1.5:.2f}:d=1.5,setsar=1[end];"
       "[main][end]concat=n=2:v=1:a=0[v]",
       "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-c:a", "aac", "-b:a", "320k",
       "-ar", "48000", "-movflags", "+faststart", final)
    json.dump({"film_seconds": film, "scene_starts": starts}, open(os.path.join(project, "logs", "edit.json"), "w"), indent=1)
    comfy.log(f"FINAL {final} ({ffprobe_duration(ffmpeg, final):.2f}s, premix {loud:.1f} LUFS -> {target} LUFS)")
    drift = check_sync(ffmpeg, final, paths, starts, order)
    comfy.log(f"sync check: worst audio offset {drift:.0f} ms" + ("  <-- OUT OF SYNC, check the clips' audio" if drift > 40 else " (ok)"))


if __name__ == "__main__":
    main()
