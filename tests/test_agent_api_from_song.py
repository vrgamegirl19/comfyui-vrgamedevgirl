"""Song-to-video pipeline: scene planning and project preparation before the full build."""

import asyncio
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
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
pipeline = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.pipeline_orchestrator")


class PlanSceneBoundariesTests(unittest.TestCase):
    def test_even_scenes_cover_the_whole_song(self):
        self.assertEqual(pipeline.plan_scene_boundaries(12.0, 4.0), [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)])

    def test_a_long_remainder_becomes_its_own_scene_and_all_scenes_are_equal(self):
        scenes = pipeline.plan_scene_boundaries(10.0, 4.0)
        self.assertEqual(len(scenes), 3)
        self.assertEqual(scenes[0][0], 0.0)
        self.assertEqual(scenes[-1][1], 10.0)
        lengths = {round(end - start, 2) for start, end in scenes}
        self.assertLessEqual(max(lengths) - min(lengths), 0.01)

    def test_a_short_remainder_is_absorbed_by_the_last_scene(self):
        scenes = pipeline.plan_scene_boundaries(8.5, 4.0)
        self.assertEqual(len(scenes), 2)
        self.assertEqual(scenes[-1][1], 8.5)

    def test_short_songs_and_bad_inputs(self):
        self.assertEqual(pipeline.plan_scene_boundaries(2.0, 4.0), [(0.0, 2.0)])
        self.assertEqual(pipeline.plan_scene_boundaries(0, 4.0), [])
        self.assertEqual(pipeline.plan_scene_boundaries(None, 4.0), [])
        self.assertEqual(len(pipeline.plan_scene_boundaries(9.0, 0.1)), 9)  # scene length floor is one second

    def test_boundaries_are_contiguous(self):
        scenes = pipeline.plan_scene_boundaries(197.3, 4.0)
        for (_, end), (start, _) in zip(scenes, scenes[1:]):
            self.assertEqual(end, start)


class FromSongJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        self.folder = os.path.join(self.root, "Song")
        os.makedirs(self.folder)
        self.session_file = os.path.join(self.folder, "vrgdg_builder_session.json")
        self.song = os.path.join(self.temp, "song.wav")
        Path(self.song).write_bytes(b"RIFF")
        self.write_session({"project_name": "Song", "project_folder": self.folder, "revision": 1, "segments": []})
        patcher = patch.object(paths, "get_allowed_project_roots", return_value=[self.root])
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_session(self, data):
        with open(self.session_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def read_session(self):
        with open(self.session_file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def fake_attach(self, project_id, audio_path=None, **_):
        session = self.read_session()
        session["audio_path"] = audio_path
        session["audio_duration"] = 10.0
        self.write_session(session)
        return {"audio_path": audio_path, "duration": 10.0}

    def run_job(self, params):
        captured = {}

        async def stub(job, manager):
            captured["params"] = dict(job.params)
            return {"project_id": job.project_id, "pipeline": "full_video", "videos_rendered": 0}

        async def go():
            manager = jobs_mod.JobManager()
            manager.register_handler("pipeline.from_song", pipeline.run_from_song_job)
            job = manager.submit_job("pipeline.from_song", project_id="Song", params=params, is_gpu=False)
            for _ in range(300):
                if job.is_terminal():
                    break
                await asyncio.sleep(0.02)
            return job

        with patch.object(pipeline, "run_build_full_video_job", stub), \
                patch.object(mutations, "attach_project_audio", side_effect=self.fake_attach):
            job = asyncio.run(go())
        return job, captured

    def test_empty_project_gets_audio_lyrics_scenes_then_builds(self):
        job, captured = self.run_job({
            "audio_path": self.song, "lyrics_text": "la la la", "scene_seconds": 4.0,
            "build_mode": "fresh_rebuild", "stitch": False,
        })
        self.assertEqual(job.status, jobs_mod.JobStatus.SUCCEEDED, job.error)
        self.assertEqual(job.result["pipeline"], "from_song")
        self.assertEqual(job.result["scenes_created"], 3)
        session = self.read_session()
        self.assertEqual(len(session["segments"]), 3)
        # Saving snapshots the song into the project folder, so the saved path is the project copy.
        self.assertTrue(os.path.isfile(session["audio_path"]))
        self.assertTrue(session["audio_path"].startswith(self.folder))
        self.assertEqual(captured["params"]["build_mode"], "fresh_rebuild")
        self.assertFalse(captured["params"]["stitch"])
        for key in ("audio_path", "lyrics_text", "scene_seconds"):
            self.assertNotIn(key, captured["params"])

    def test_existing_scenes_are_kept(self):
        session = self.read_session()
        session["audio_path"] = self.song
        session["audio_duration"] = 10.0
        session["segments"] = [{"id": "seg_0001", "label": "Mine", "start": 0.0, "end": 10.0}]
        self.write_session(session)
        job, _ = self.run_job({})
        self.assertEqual(job.status, jobs_mod.JobStatus.SUCCEEDED, job.error)
        self.assertEqual(job.result["scenes_created"], 0)
        self.assertEqual([s["label"] for s in self.read_session()["segments"]], ["Mine"])

    def test_missing_audio_fails_with_a_clear_error(self):
        job, _ = self.run_job({})
        self.assertEqual(job.status, jobs_mod.JobStatus.FAILED)
        self.assertIn("audio", str(job.error).lower())


if __name__ == "__main__":
    unittest.main()
