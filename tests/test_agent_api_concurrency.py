"""Concurrency tests: agent writers, If-Match revisions, and UI snapshots racing agent edits (Apireport Section 11, item 6)."""

import importlib
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
builder_project = importlib.import_module(f"{pkg_name}.builder.project")


class ConcurrentEditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        self.folder = os.path.join(self.root, "Song")
        os.makedirs(self.folder)
        self.session_file = os.path.join(self.folder, "vrgdg_builder_session.json")
        self.save_session({
            "project_name": "Song",
            "project_folder": self.folder,
            "revision": 1,
            "video_engine": "minimax_h3",
            "segments": [
                {"id": "seg_0001", "label": "Scene 1", "start": 0.0, "end": 4.0},
                {"id": "seg_0002", "label": "Scene 2", "start": 4.0, "end": 8.0},
            ],
            "flux_reference_builder": {"subjects": [], "locations": []},
        })
        patcher = patch.object(paths, "get_allowed_project_roots", return_value=[self.root])
        patcher.start()
        self.addCleanup(patcher.stop)

    def save_session(self, data):
        with open(self.session_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def load_session(self):
        with open(self.session_file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_stale_if_match_is_rejected_and_fresh_one_wins(self):
        first = mutations.patch_project_settings("Song", {"project": {"video_engine": "ltx"}}, if_match_revision=1)
        revision_after_first = first["revision"]
        self.assertGreater(revision_after_first, 1)
        with self.assertRaises(errors.RevisionConflictError):
            mutations.patch_project_settings("Song", {"project": {"video_engine": "minimax_h3"}}, if_match_revision=1)
        self.assertEqual(self.load_session()["video_engine"], "ltx")
        mutations.patch_project_settings(
            "Song", {"project": {"video_engine": "minimax_h3"}}, if_match_revision=revision_after_first
        )
        self.assertEqual(self.load_session()["video_engine"], "minimax_h3")

    def test_parallel_writers_lose_no_scenes_and_revisions_only_go_up(self):
        count = 8
        results, failures = [], []

        def add_scene(index):
            try:
                results.append(mutations.create_scene("Song", position="append", label=f"New {index}", duration=2.0))
            except Exception as exc:  # pragma: no cover - reported through the assertion below
                failures.append(exc)

        threads = [threading.Thread(target=add_scene, args=(i,)) for i in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual(failures, [])
        saved = self.load_session()
        self.assertEqual(len(saved["segments"]), 2 + count)
        ids = [segment["id"] for segment in saved["segments"]]
        self.assertEqual(len(ids), len(set(ids)), "scene ids must stay unique")
        revisions = sorted(result["revision"] for result in results)
        self.assertEqual(len(set(revisions)), count, "every write must produce its own revision")
        self.assertEqual(revisions, sorted(revisions))
        self.assertEqual(saved["revision"], max(revisions))

    def test_if_match_race_has_exactly_one_winner(self):
        start_revision = int(self.load_session()["revision"])
        outcomes = []
        barrier = threading.Barrier(4)

        def patch_once(name):
            barrier.wait()
            try:
                mutations.patch_project_settings(
                    "Song", {"project": {"video_engine": name}}, if_match_revision=start_revision
                )
                outcomes.append("ok")
            except errors.RevisionConflictError:
                outcomes.append("conflict")

        threads = [threading.Thread(target=patch_once, args=(f"engine_{i}",)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual(sorted(outcomes), ["conflict", "conflict", "conflict", "ok"])

    def test_ui_snapshot_older_than_an_agent_edit_does_not_overwrite_it(self):
        stale_snapshot = self.load_session()
        stale_snapshot["builder_save_revision"] = 5
        self.save_session({**stale_snapshot, "builder_save_revision": 6})
        mutations.create_scene("Song", position="append", label="Agent scene", duration=2.0)
        agent_state = self.load_session()
        self.assertEqual(len(agent_state["segments"]), 3)

        result = builder_project._save_builder_session({"project_folder": self.folder, "session": stale_snapshot})

        self.assertTrue(result.get("stale"), "the stale UI save must be reported, not silently accepted")
        self.assertEqual(len(self.load_session()["segments"]), 3, "the agent's scene must survive the stale UI save")


if __name__ == "__main__":
    unittest.main()
