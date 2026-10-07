"""POST /projects/{pid}/stitch is frame-accurate to the timeline, like the Video Builder's preview stitch.

Live report (Cut and Gun, scenes 1-6): the scene clips hold 162+117+141+183+148+120 = 871 frames, the
Builder's preview stitch has 871 and the API stitch had 870. The API sent no scene timing, so the stitcher
skipped timeline frame sync and muxed with ``-shortest``; the joined scene audio was a little shorter than the
video, so ``-shortest`` cut the last frame.
"""

import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
jobs_models = importlib.import_module(f"{pkg_name}.agent_api.jobs.models")
router = importlib.import_module(f"{pkg_name}.agent_api.router")
video_orch = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.video_orchestrator")
video_files = importlib.import_module(f"{pkg_name}.runner.video_files")

ValidationError = errors.ValidationError
Job = jobs_models.Job

API = "/vrgdg/api/v1"
FPS = 24
# Frame counts of Cut and Gun scenes 1-6 (ffprobe -count_packets on video_0001..0006-audio.mp4).
SCENE_FRAMES = [162, 117, 141, 183, 148, 120]
PROJECT = "StitchProject"


def _have_ffmpeg():
    try:
        for tool in ("ffmpeg", "ffprobe"):
            subprocess.run([tool, "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _frame_count(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out)


def _stream_duration(path, stream):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", "stream=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def _mean_volume(path, start, duration):
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-ss", str(start), "-t", str(duration), "-i", path, "-vn", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    for line in out.splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].split("dB")[0])
    return -999.0


def _timeline():
    """Scene start/end in seconds. Off the frame grid a little, like real Builder timings."""
    bounds, total = [0.0], 0
    for frames in SCENE_FRAMES:
        total += frames
        bounds.append(total / FPS + 0.01)
    return [(bounds[i], bounds[i + 1]) for i in range(len(SCENE_FRAMES))]


class _Routes:
    def __init__(self):
        self.handlers = {}

    def __getattr__(self, method):
        def register(path):
            def decorator(handler):
                self.handlers[(method.upper(), path)] = handler
                return handler
            return decorator
        return register


class _Request:
    def __init__(self, project_id, body):
        self.match_info = {"project_id": project_id}
        self.headers = {}
        self.can_read_body = True
        self._body = body

    async def json(self):
        return self._body


class StitchBase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vrgdg_stitch_test_")
        self.project_dir = os.path.join(self.test_dir, PROJECT)
        self.scene_dir = os.path.join(self.project_dir, "rendered_scene_videos")
        os.makedirs(self.scene_dir, exist_ok=True)
        self.song = os.path.join(self.test_dir, "song.wav")
        with wave.open(self.song, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(48000)
            handle.writeframes(b"\x00\x00" * 48000 * 40)
        self.clips = [os.path.join(self.scene_dir, f"video_{i:04d}-audio.mp4") for i in range(1, 7)]
        for clip in self.clips:
            with open(clip, "wb") as handle:
                handle.write(b"\x00\x00\x00\x20ftypisom")
        self.timeline = _timeline()
        self.write_session()
        self.orig_env = os.environ.get("VRGDG_PROJECT_ROOTS")
        os.environ["VRGDG_PROJECT_ROOTS"] = self.test_dir

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["VRGDG_PROJECT_ROOTS"] = self.orig_env
        else:
            os.environ.pop("VRGDG_PROJECT_ROOTS", None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def write_session(self, engine="minimax_h3", audio_mode="built_in_audio", drop_video=None):
        segments = []
        for index, ((start, end), clip) in enumerate(zip(self.timeline, self.clips), start=1):
            segments.append({
                "id": f"seg_{index:02d}",
                "start": start,
                "end": end,
                "video_path": "" if drop_video == index else clip,
            })
        session = {
            "project_name": PROJECT,
            "audio_path": self.song,
            "video_engine": engine,
            "minimax_h3_settings": {"audio_mode": audio_mode},
            "revision": 1,
            "segments": segments,
        }
        with open(os.path.join(self.project_dir, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(session, handle)

    def run_stitch(self, params):
        job = Job(id="job_test", type="video.stitch", project_id=PROJECT, params=params, is_gpu=False)
        return asyncio.run(video_orch.run_video_stitch_job(job, MagicMock()))

    def captured_payload(self, params):
        with patch.object(video_files, "_stitch_scene_videos", return_value={"final_video_path": "x.mp4"}) as stitch:
            self.run_stitch(params)
        self.assertEqual(stitch.call_count, 1)
        return stitch.call_args[0][0]


class StitchPayloadTests(StitchBase):
    def test_preview_passes_timing_items_and_frame_sync_like_the_builder(self):
        payload = self.captured_payload({"scene_ids": ["2", "3", "4"], "output_prefix": "API_PREVIEW_SCENES_002-004"})
        self.assertEqual(payload["scene_paths"], self.clips[1:4])
        self.assertEqual(payload["timeline_fps"], 24)
        items = payload["scene_timing_items"]
        self.assertEqual(len(items), 3)
        frames = [int(i["end"] * FPS + 0.5) - int(i["start"] * FPS + 0.5) for i in items]
        self.assertEqual(frames, SCENE_FRAMES[1:4])
        # Shifted by a whole number of frames, so the per-scene rounding matches the full timeline.
        self.assertLess(items[0]["start"], 1 / FPS)
        self.assertEqual(payload["output_prefix"], "API_PREVIEW_SCENES_002-004")
        # MiniMax built-in audio: the scenes' own audio, as the Builder does.
        self.assertTrue(payload["use_embedded_scene_audio"])
        self.assertEqual(payload["audio_path"], "")

    def test_project_audio_is_trimmed_to_the_selected_scenes(self):
        payload = self.captured_payload({"scene_ids": ["2", "3", "4"], "audio": "project"})
        self.assertFalse(payload["use_embedded_scene_audio"])
        self.assertEqual(payload["audio_path"], self.song)
        start, end = self.timeline[1][0], self.timeline[3][1]
        self.assertAlmostEqual(payload["audio_start"], start, places=6)
        self.assertAlmostEqual(payload["audio_duration"], end - start, places=6)
        self.assertEqual(len(payload["scene_timing_items"]), 3)

    def test_whole_project_uses_the_full_song_like_render_all(self):
        payload = self.captured_payload({"audio": "project"})
        self.assertEqual(payload["scene_paths"], self.clips)
        self.assertEqual(len(payload["scene_timing_items"]), 6)
        self.assertEqual(payload["audio_start"], 0)
        self.assertEqual(payload["audio_duration"], 0)

    def test_input_audio_project_defaults_to_the_song(self):
        self.write_session(audio_mode="input_audio")
        payload = self.captured_payload({"scene_ids": ["1", "2"]})
        self.assertFalse(payload["use_embedded_scene_audio"])
        self.assertEqual(payload["audio_path"], self.song)
        self.assertAlmostEqual(payload["audio_duration"], self.timeline[1][1], places=6)

    def test_ltx_project_has_no_frame_sync_like_the_builder(self):
        self.write_session(engine="ltx")
        payload = self.captured_payload({"scene_ids": ["1", "2"], "audio": "project"})
        self.assertEqual(payload["scene_timing_items"], [])
        self.assertEqual(payload["timeline_fps"], 0)

    def test_result_reports_selection_and_audio_window(self):
        with patch.object(video_files, "_stitch_scene_videos", return_value={"final_video_path": "x.mp4", "timeline_frame_sync": True}):
            result = self.run_stitch({"scene_ids": ["2", "3"], "audio": "project"})
        self.assertEqual(result["scene_ids"], ["seg_02", "seg_03"])
        self.assertEqual(result["audio"], "project")
        self.assertEqual(result["expected_frame_count"], SCENE_FRAMES[1] + SCENE_FRAMES[2])


class StitchSelectionTests(StitchBase):
    def test_scenes_by_number_and_by_id(self):
        by_number = self.captured_payload({"scene_ids": ["1", "2", "3"]})
        by_id = self.captured_payload({"scene_ids": ["seg_01", "seg_02", "seg_03"]})
        mixed = self.captured_payload({"scene_ids": [1, "seg_02", "3"]})
        self.assertEqual(by_number["scene_paths"], self.clips[:3])
        self.assertEqual(by_id["scene_paths"], self.clips[:3])
        self.assertEqual(mixed["scene_paths"], self.clips[:3])

    def test_selection_follows_timeline_order_and_ignores_repeats(self):
        payload = self.captured_payload({"scene_ids": ["4", "2", "3", "seg_02"]})
        self.assertEqual(payload["scene_paths"], self.clips[1:4])

    def test_unknown_scene_is_an_error_naming_it(self):
        with self.assertRaises(ValidationError) as ctx:
            self.run_stitch({"scene_ids": ["2", "9", "seg_nope"]})
        self.assertIn("9", str(ctx.exception))
        self.assertIn("seg_nope", str(ctx.exception))

    def test_selected_scene_without_video_is_an_error(self):
        os.remove(self.clips[2])  # the session loader would find the file again from the folder
        self.write_session(drop_video=3)
        with self.assertRaises(ValidationError) as ctx:
            self.run_stitch({"scene_ids": ["2", "3", "4"]})
        self.assertIn("seg_03", str(ctx.exception))

    def test_gap_in_selection_with_project_audio_is_an_error(self):
        with self.assertRaises(ValidationError) as ctx:
            self.run_stitch({"scene_ids": ["1", "3"], "audio": "project"})
        self.assertIn("contiguous", str(ctx.exception))

    def test_gap_in_selection_with_embedded_audio_is_allowed(self):
        payload = self.captured_payload({"scene_ids": ["1", "3"], "audio": "embedded"})
        self.assertEqual(payload["scene_paths"], [self.clips[0], self.clips[2]])


class StitchRouteTests(StitchBase):
    def handler(self):
        server = MagicMock()
        server.routes = _Routes()
        with patch.object(router, "_VRGDG_AGENT_API_ROUTES_REGISTERED", False):
            router.register_agent_api_routes(server)
        return server.routes.handlers[("POST", f"{API}/projects/{{project_id}}/stitch")]

    def post(self, body):
        manager = MagicMock()
        manager.submit_job.return_value = MagicMock(id="job_x", status="queued", to_dict=lambda: {})
        with patch.object(router, "get_job_manager", return_value=manager), \
             patch.object(router, "verify_auth", lambda request: None):
            response = asyncio.run(self.handler()(_Request(PROJECT, body)))
        return response, manager

    def test_unknown_body_key_is_rejected(self):
        response, manager = self.post({"scene_ids": ["1"], "sceneIds": ["2"]})
        self.assertEqual(response.status, 400)
        self.assertIn("sceneIds", response.text)
        manager.submit_job.assert_not_called()

    def test_bad_audio_value_is_rejected(self):
        response, manager = self.post({"audio": "song"})
        self.assertEqual(response.status, 400)
        self.assertIn("audio", response.text)
        manager.submit_job.assert_not_called()

    def test_unknown_scene_is_rejected_before_the_job_starts(self):
        response, manager = self.post({"scene_ids": ["7"]})
        self.assertEqual(response.status, 400)
        manager.submit_job.assert_not_called()

    def test_valid_body_starts_the_job(self):
        response, manager = self.post({"scene_ids": ["1", "2"], "output_prefix": "P", "audio": "embedded"})
        self.assertEqual(response.status, 202)
        params = manager.submit_job.call_args.kwargs["params"]
        self.assertEqual(params["scene_ids"], ["1", "2"])

    def test_endpoints_json_documents_the_body_keys(self):
        with open(ROOT / "agent_api" / "endpoints.json", encoding="utf-8") as handle:
            data = json.load(handle)
        routes = data["endpoints"]
        route = next(r for r in routes if r["method"] == "POST" and r["path"] == "/projects/{project_id}/stitch")
        self.assertEqual(sorted(route["body_keys"]), ["audio", "audio_path", "output_prefix", "overlays", "scene_ids"])


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg/ffprobe not installed")
class StitchFrameCountFfmpegTests(StitchBase):
    """Real FFmpeg on synthetic 24 fps clips with the live frame counts."""

    AUDIO_SHORTFALL = 0.020  # each clip's audio ends 20 ms before its last frame, as on Osiris

    def setUp(self):
        super().setUp()
        for index, (clip, frames) in enumerate(zip(self.clips, SCENE_FRAMES), start=1):
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", f"testsrc2=size=160x90:rate={FPS}",
                "-f", "lavfi", "-i", f"sine=frequency={300 + 60 * index}:sample_rate=48000:duration={frames / FPS - self.AUDIO_SHORTFALL:.6f}",
                "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", clip,
            ], capture_output=True, check=True)
        # Song: silent until 6.5 s, then a tone. Scene 2 starts at 6.76 s, so a preview of scenes 2-4 that
        # takes its audio window from the song opens on the tone; one that muxes the song from 0 s opens silent.
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "aevalsrc=if(gt(t\\,6.5)\\,0.5*sin(2*PI*440*t)\\,0):s=48000:d=40",
                        "-c:a", "pcm_s16le", self.song], capture_output=True, check=True)

    def test_clips_hold_the_live_frame_counts(self):
        self.assertEqual([_frame_count(c) for c in self.clips], SCENE_FRAMES)

    def test_embedded_audio_stitch_keeps_every_frame(self):
        result = self.run_stitch({"scene_ids": ["1", "2", "3", "4", "5", "6"],
                                  "output_prefix": "API_PREVIEW_SCENES_001-006", "audio": "embedded"})
        out = result["final_video_path"]
        self.assertEqual(_frame_count(out), sum(SCENE_FRAMES))  # 871
        self.assertAlmostEqual(_stream_duration(out, "v:0"), sum(SCENE_FRAMES) / FPS, places=3)
        self.assertTrue(result["timeline_frame_sync"])

    def test_project_audio_preview_matches_its_window(self):
        result = self.run_stitch({"scene_ids": ["2", "3", "4"], "audio": "project"})
        out = result["final_video_path"]
        frames = sum(SCENE_FRAMES[1:4])
        self.assertEqual(_frame_count(out), frames)
        # The song is cut to the three scenes, not muxed whole from 0 s.
        self.assertLess(abs(_stream_duration(out, "a:0") - frames / FPS), 0.05)
        self.assertGreater(_mean_volume(out, 0.1, 0.5), -40.0)


if __name__ == "__main__":
    unittest.main()
