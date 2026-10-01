"""Phase 1: lyric timing, scenes from lines, and the min/max scene-length endpoint."""

import asyncio
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
jobs_mod = importlib.import_module(f"{pkg_name}.agent_api.jobs")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
orch = importlib.import_module(f"{pkg_name}.agent_api.orchestrator")
lyrics_orch = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.lyrics_orchestrator")
builder_project = importlib.import_module(f"{pkg_name}.builder.project")
utility_workflows = importlib.import_module(f"{pkg_name}.runner.utility_workflows")

LYRICS = "[Verse 1]\nSlide in the room\nShe looking for a king\n\n[bridge]\nI get the bag\nThen I'm out\n"


def timed_payload():
    return {
        "duration": 24.0,
        "segment_mode": "reference_lines",
        "segments": [
            {"start": 2.0, "end": 4.5, "text": "Slide in the room", "type": "vocal"},
            {"start": 4.5, "end": 6.0, "text": "She looking for a king", "type": "vocal"},
            {"start": 12.0, "end": 14.0, "text": "I get the bag", "type": "vocal"},
            {"start": 14.0, "end": 17.5, "text": "Then I'm out", "type": "vocal"},
        ],
    }


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        self.folder = os.path.join(self.root, "Song")
        os.makedirs(os.path.join(self.folder, "project_context"))
        self.audio = os.path.join(self.temp, "song.wav")
        with wave.open(self.audio, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(b"\x00\x00" * 8000)
        self.session_file = os.path.join(self.folder, "vrgdg_builder_session.json")
        self.write_session({
            "project_name": "Song", "project_folder": self.folder, "revision": 1,
            "audio_path": self.audio, "audio_duration": 24.0, "segments": [],
        })
        for target, name, value in (
            (builder_project, "_model_defaults_path", lambda: os.path.join(self.temp, "model_defaults.json")),
            (paths, "get_allowed_project_roots", lambda: [self.root]),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_session(self, data):
        with open(self.session_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def read_session(self):
        with open(self.session_file, "r", encoding="utf-8") as handle:
            return json.load(handle)


class EnforceLengthTests(Base):
    def seed_scenes(self, lengths, texts=None, locked=()):
        cursor, segments = 0.0, []
        for index, length in enumerate(lengths):
            scene = {"id": f"seg_{index}", "label": f"Scene {index + 1}", "start": cursor, "end": cursor + length,
                     "lyric_text": (texts or {}).get(index, f"word{index}")}
            if index in locked:
                scene["video_path"] = os.path.join(self.folder, "rendered_scene_videos", f"video_{index + 1:04d}-audio.mp4")
            segments.append(scene)
            cursor += length
        session = self.read_session()
        session["segments"] = segments
        self.write_session(session)

    def segments(self):
        return self.read_session()["segments"]

    def test_long_scenes_are_cut_and_short_ones_merged(self):
        self.seed_scenes([12.0, 2.1, 2.1, 5.0], {0: "a b c d e f", 1: "x", 2: "y", 3: "z"})
        result = mutations.enforce_scene_lengths_on_project("Song", 3.5, 10.0)
        lengths = [round(s["end"] - s["start"], 2) for s in self.segments()]
        self.assertTrue(all(3.5 <= v <= 10.0 for v in lengths), lengths)
        self.assertEqual(result["still_outside_limits"], [])
        self.assertEqual(result["scenes_after"], len(self.segments()))
        words = " ".join(s["lyric_text"] for s in self.segments()).split()
        self.assertEqual(sorted(words), sorted("a b c d e f x y z".split()))

    def test_dry_run_changes_nothing_and_reports_the_plan(self):
        self.seed_scenes([12.0, 2.0, 6.0])
        before = self.segments()
        result = mutations.enforce_scene_lengths_on_project("Song", 3.5, 10.0, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertGreaterEqual(result["splits"] + result["merges"], 1)
        self.assertEqual(self.segments(), before)

    def test_scenes_with_rendered_video_are_left_alone_and_reported(self):
        self.seed_scenes([14.0, 5.0], locked={0})
        result = mutations.enforce_scene_lengths_on_project("Song", 3.5, 10.0)
        self.assertEqual(self.segments()[0]["end"] - self.segments()[0]["start"], 14.0)
        self.assertEqual([item["scene_id"] for item in result["still_outside_limits"]], ["seg_0"])
        self.assertEqual(result["still_outside_limits"][0]["reason"], "has rendered video")

    def test_split_and_merge_keep_the_lyrics_with_the_right_scene(self):
        self.seed_scenes([12.0], {0: "one two three four"})
        mutations.split_scene("Song", "seg_0", 6.0)
        left, right = self.segments()
        self.assertEqual((left["lyric_text"], right["lyric_text"]), ("one two", "three four"))
        mutations.merge_scenes("Song", left["id"], with_direction="next")
        merged = self.segments()
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["lyric_text"], "one two\nthree four")

    def test_bad_limits_fail_clearly(self):
        self.seed_scenes([5.0, 5.0])
        with self.assertRaises(ValueError):
            mutations.enforce_scene_lengths_on_project("Song", 10.0, 3.5)


class AlignAndCreateTests(Base):
    def setUp(self):
        super().setUp()
        self.fake = orch.FakeComfyClient()
        orch.set_comfy_client(self.fake)
        self.graphs = []

        def fake_build(request):
            self.graphs.append(request)
            return {"workflow_path": "x", "prompt": {"1": {"class_type": "Noop", "inputs": {}}}}

        patcher = patch.object(utility_workflows, "_build_timestamped_transcribe_api_prompt", side_effect=fake_build)
        patcher.start()
        self.addCleanup(patcher.stop)
        original = self.fake.queue_prompt

        def queue(prompt, client_id=None):
            result = original(prompt, client_id)
            self.fake.set_mock_output(result["prompt_id"], text=[json.dumps(timed_payload())])
            return result

        self.fake.queue_prompt = queue
        session = self.read_session()
        session["lyric_mapper"] = {"source_text": LYRICS, "lines": []}
        self.write_session(session)

    def run_job(self, job_type, params):
        async def go():
            manager = jobs_mod.JobManager()
            lyrics_orch.register_lyrics_orchestrator_handlers(manager)
            job = manager.submit_job(job_type, project_id="Song", params=params, is_gpu=False)
            for _ in range(300):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)
            return job

        return asyncio.run(go())

    def test_align_runs_the_timestamp_workflow_and_saves_the_timed_lines(self):
        job = self.run_job("lyrics.align", {})
        self.assertEqual(job.status, jobs_mod.JobStatus.SUCCEEDED, job.error)
        self.assertEqual(job.result["lines"], 4)
        with open(os.path.join(self.folder, "project_context", "timestamped_lyrics.json"), encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(len(saved["segments"]), 4)
        request = self.graphs[0]
        self.assertEqual(request["audio_path"], self.audio)
        self.assertIn("Slide in the room", request["reference_lyrics"])
        self.assertEqual(request["segment_mode"], "reference_lines")

    def test_scenes_from_lines_follow_the_requested_length_range(self):
        job = self.run_job("timeline.from_lines", {"min_scene_seconds": 3.5, "max_scene_seconds": 10.0})
        self.assertEqual(job.status, jobs_mod.JobStatus.SUCCEEDED, job.error)
        scenes = self.read_session()["segments"]
        lengths = [round(s["end"] - s["start"], 2) for s in scenes]
        self.assertTrue(all(3.5 <= v <= 10.0 for v in lengths), lengths)
        self.assertAlmostEqual(scenes[0]["start"], 0.0)
        self.assertAlmostEqual(scenes[-1]["end"], 24.0, places=2)
        spoken = " ".join(s["lyric_text"] for s in scenes if "instrumental" not in s["lyric_text"].lower())
        for line in ("Slide in the room", "She looking for a king", "I get the bag", "Then I'm out"):
            self.assertIn(line, spoken)
        self.assertEqual(self.read_session()["lyric_mapper"]["source_text"], LYRICS.strip())
        self.assertTrue(self.read_session()["show_timeline_lyric_notes"], "the timeline shows the lyric lane")
        self.assertEqual(job.result["scenes"], len(scenes))
        sections = {s["lyric_section"] for s in scenes}
        self.assertTrue({"Verse 1", "bridge"} <= sections, sections)

    def test_the_length_rule_can_be_turned_off_to_keep_one_scene_per_line(self):
        job = self.run_job("timeline.from_lines", {"enforce_lengths": False, "min_scene_seconds": 3.5, "max_scene_seconds": 10.0})
        self.assertEqual(job.status, jobs_mod.JobStatus.SUCCEEDED, job.error)
        lyric_scenes = [s for s in self.read_session()["segments"] if "instrumental" not in s["lyric_text"].lower()]
        self.assertEqual(len(lyric_scenes), 4)

    def test_existing_scenes_are_not_replaced_without_asking(self):
        session = self.read_session()
        session["segments"] = [{"id": "seg_a", "start": 0, "end": 5, "label": "Mine"}]
        self.write_session(session)
        job = self.run_job("timeline.from_lines", {})
        self.assertEqual(job.status, jobs_mod.JobStatus.FAILED)
        self.assertIn("replace_existing", str(job.error))
        job = self.run_job("timeline.from_lines", {"replace_existing": True})
        self.assertEqual(job.status, jobs_mod.JobStatus.SUCCEEDED, job.error)

    def test_scenes_with_media_are_never_orphaned(self):
        session = self.read_session()
        session["segments"] = [{"id": "seg_a", "start": 0, "end": 5, "approved_image_path": "x.png"}]
        self.write_session(session)
        job = self.run_job("timeline.from_lines", {"replace_existing": True})
        self.assertEqual(job.status, jobs_mod.JobStatus.FAILED)

    def test_a_saved_alignment_can_be_reused_without_running_comfyui(self):
        self.run_job("lyrics.align", {})
        self.graphs.clear()
        job = self.run_job("timeline.from_lines", {"use_saved_alignment": True})
        self.assertEqual(job.status, jobs_mod.JobStatus.SUCCEEDED, job.error)
        self.assertEqual(self.graphs, [])

    def test_missing_audio_or_lyrics_fail_with_a_clear_message(self):
        session = self.read_session()
        session.pop("audio_path")
        self.write_session(session)
        job = self.run_job("lyrics.align", {})
        self.assertEqual(job.status, jobs_mod.JobStatus.FAILED)
        self.assertIn("audio", str(job.error).lower())
        session["audio_path"] = self.audio
        session["lyric_mapper"] = {}
        self.write_session(session)
        job = self.run_job("lyrics.align", {})
        self.assertEqual(job.status, jobs_mod.JobStatus.FAILED)
        self.assertIn("lyrics", str(job.error).lower())

    def test_output_without_timed_lines_is_an_error_not_an_empty_timeline(self):
        original = self.fake.queue_prompt

        def queue(prompt, client_id=None):
            result = original(prompt, client_id)
            self.fake.set_mock_output(result["prompt_id"], text=["no json here"])
            return result

        self.fake.queue_prompt = queue
        job = self.run_job("lyrics.align", {})
        self.assertEqual(job.status, jobs_mod.JobStatus.FAILED)
        self.assertIn("timestamped lyrics JSON", str(job.error))


class SafetyCheckTests(unittest.TestCase):
    def test_one_scene_with_two_pasted_lines_is_rejected(self):
        with self.assertRaises(errors.ValidationError):
            lyrics_orch.assert_no_bundled_reference_lyrics(
                ["Slide in the room She looking for a king"], "Slide in the room\nShe looking for a king\nThen I'm out"
            )
        lyrics_orch.assert_no_bundled_reference_lyrics(["Slide in the room"], "Slide in the room\nShe looking for a king")

    def test_parse_accepts_json_with_surrounding_text(self):
        parsed = lyrics_orch.parse_timestamped_lyrics_output('log line\n{"segments": [], "duration": 3}\nmore')
        self.assertEqual(parsed["duration"], 3)


if __name__ == "__main__":
    unittest.main()
