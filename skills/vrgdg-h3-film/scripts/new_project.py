"""Create a film project folder: settings.json (from check_env.py's env.json), folders, a silent timing track, and
starter screenplay.json / image_specs.json files to fill in.

  python new_project.py <new project folder>
"""
import json
import os
import shutil
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SKILL_DIR, save_json  # noqa: E402

FOLDERS = ["characters", "locations", "prompts", "storyboard", "video_clips/raw", "video_clips/rejected", "audio/score",
           "logs/qa", "workflows", "final", "builder_project"]
DEFAULTS = {
    "seed": 4242,               # one seed for every scene keeps the built-in voices consistent
    "warmup_frames": 5, "tail_hold_seconds": 1.0,
    "latent_context_frames": 16,
    "fps": 24,
    "film_offset": 2.0,         # scenes sit on the silent timing track from 2 s
    "max_scene_seconds": 8.0,
    "max_cast_per_shot": 3,
    "prompt_limit": 7000,
    "final_width": 1920,
    "final_height": 1080,
    "loudness_lufs": -14.0,
}


def silent_wav(path, seconds=900, rate=48000):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        chunk = b"\x00" * (rate * 4)
        for _ in range(seconds):
            w.writeframes(chunk)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    project = os.path.abspath(sys.argv[1])
    env_path = os.path.join(SKILL_DIR, "env.json")
    if not os.path.isfile(env_path):
        raise SystemExit("Run check_env.py first.")
    env = json.load(open(env_path, encoding="utf-8"))
    if env.get("problems"):
        raise SystemExit("check_env.py reported problems; fix them first:\n  " + "\n  ".join(env["problems"]))
    for f in FOLDERS:
        os.makedirs(os.path.join(project, f), exist_ok=True)
    settings_path = os.path.join(project, "settings.json")
    if not os.path.isfile(settings_path):
        save_json(settings_path, {**DEFAULTS, **{k: env.get(k) for k in ("comfy_url", "models", "music_engine", "music_engines", "yue2_root",
                                                                      "qa_python", "ffmpeg")}})
    track = os.path.join(project, "audio", "silent_timing_track.wav")
    if not os.path.isfile(track):
        silent_wav(track)
    for name in ("screenplay.json", "image_specs.json", "score.json"):
        dst = os.path.join(project, name)
        if not os.path.isfile(dst):
            shutil.copy2(os.path.join(SKILL_DIR, "templates", name), dst)
    print(f"project ready: {project}")
    print("next: write screenplay.json and image_specs.json, then gen_refs.py -> build_prompts.py -> render_scenes.py")


if __name__ == "__main__":
    main()
