"""Stitch the final video through the Builder and run the delivery checks.

usage: stitch.py --work W [--project P] [--prefix FINAL_VIDEO] [--skip-sync]
       stitch.py --work W [--project P] --check-only FINAL.mp4        (re-check an existing final, no stitch)

Checks: duration against the project audio, black or frozen stretches (ffmpeg blackdetect / freezedetect), that
frame 0 is a real picture (never fade the start in from black), an overview sheet (one frame every 4 s ->
qa/final_overview_<name>.jpg; read it), and sync_check.py on every scene.
"""
import os
import subprocess
import sys

from PIL import Image, ImageStat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Api, ffprobe_duration, load_config, parser, wpath  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def checks(cfg, final, name, project, skip_sync):
    duration = ffprobe_duration(final)
    track = ffprobe_duration(wpath(cfg, "audio/full_track.wav"))
    print(f"  duration {duration:.2f} s" + (f" (project audio {track:.2f} s)" if track else ""))
    if track and abs(duration - track) > 0.1:
        print("  WARN duration differs from the project audio")
    scan = subprocess.run(["ffmpeg", "-v", "info", "-i", final, "-vf",
                           "blackdetect=d=0.1:pix_th=0.08,freezedetect=n=0.003:d=1.0", "-an", "-f", "null", "-"],
                          capture_output=True, text=True).stderr
    bad = [l.strip() for l in scan.splitlines() if "black_start" in l or "freeze_start" in l]
    print(f"  black/frozen stretches: {len(bad)}")
    for line in bad[:10]:
        print("   ", line[:160])
    qa = wpath(cfg, "qa")
    os.makedirs(qa, exist_ok=True)
    first = os.path.join(qa, f"final_frame0_{name}.jpg")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", final, "-frames:v", "1", "-vf", "scale=480:-2", first], check=False)
    brightness = sum(ImageStat.Stat(Image.open(first)).mean) / 3
    print(f"  frame 0 brightness {brightness:.0f}" + ("  <-- black: the film must not fade in at the start" if brightness < 12 else ""))
    overview = os.path.join(qa, f"final_overview_{name}.jpg")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", final, "-vf", "fps=1/4,scale=320:-2,tile=8x6",
                    "-frames:v", "1", overview], check=False)
    print(f"  overview sheet: {overview}")
    if not skip_sync:
        cmd = [sys.executable, os.path.join(HERE, "sync_check.py"), "--work", cfg["_work"]]
        if project:
            cmd += ["--project", project]
        subprocess.run(cmd + [final], check=False)


def main():
    p = parser(__doc__)
    p.add_argument("--prefix", default="FINAL_VIDEO")
    p.add_argument("--skip-sync", action="store_true")
    p.add_argument("--check-only", default=None, metavar="FINAL.mp4")
    args = p.parse_args()
    cfg = load_config(args.work)
    if args.check_only:
        print(f"final: {args.check_only}")
        checks(cfg, args.check_only, os.path.splitext(os.path.basename(args.check_only))[0], args.project, args.skip_sync)
        return
    api = Api(cfg, args.project)
    missing = [s["number"] for s in api.scenes() if not s.get("rendered_video")]
    if missing:
        sys.exit(f"scenes without a video: {missing}")
    job = api.run_job("POST", "/projects/{pid}/stitch", {"output_prefix": args.prefix})
    result = job.get("result") or {}
    if job.get("status") != "succeeded":
        sys.exit(f"stitch failed: {job.get('error')}")
    final = result.get("final_video_path")
    print(f"final: {final}")
    print(f"  {result.get('output_width')}x{result.get('output_height')}, {result.get('timeline_frame_count')} frames, "
          f"audio={result.get('audio')}")
    checks(cfg, final, args.prefix, args.project, args.skip_sync)


if __name__ == "__main__":
    main()
