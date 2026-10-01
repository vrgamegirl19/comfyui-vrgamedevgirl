"""Describe a reference, extract locations, assign them to scenes (the Reference Builder steps for agents)."""

import asyncio
import importlib
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
jobs_mod = importlib.import_module(f"{pkg_name}.agent_api.jobs")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
ref_orch = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.reference_orchestrator")
builder_project = importlib.import_module(f"{pkg_name}.builder.project")


class FakeLmStudio:
    def __init__(self, loaded_model="gemma-4-e4b-it"):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.requests.append(("GET", self.path))
                body = json.dumps({"data": [
                    {"id": loaded_model, "state": "loaded", "type": "vlm", "loaded_context_length": 32768},
                    {"id": "bonsai-27b", "state": "not-loaded", "type": "vlm"},
                ]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                outer.requests.append(("POST", self.path))
                self.send_response(500)
                self.end_headers()

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        self.folder = os.path.join(self.root, "Song")
        os.makedirs(os.path.join(self.folder, "project_context"))
        self.image = os.path.join(self.temp, "darrel.png")
        Path(self.image).write_bytes(b"png")
        self.lm = FakeLmStudio()
        self.addCleanup(self.lm.close)
        self.session_file = os.path.join(self.folder, "vrgdg_builder_session.json")
        self.segments = [
            {"id": f"seg_{i}", "label": f"SCENE {i + 1}", "start": i * 4.0, "end": (i + 1) * 4.0, "lyric_text": f"line {i + 1}"}
            for i in range(10)
        ]
        self.write_session({
            "project_name": "Song", "project_folder": self.folder, "revision": 1, "video_engine": "minimax_h3",
            "text_gemma_runner": "lm_studio", "lm_studio_base_url": self.lm.url, "lm_studio_model": "bonsai-27b",
            "lm_studio_context_limit": 131072, "segments": self.segments,
            "flux_reference_builder": {
                "subjects": [{"id": "darrel", "name": "Darrel", "description": "", "reference_type": "character", "image": {"path": self.image}}],
                "locations": [],
            },
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

    def add_locations(self, count):
        session = self.read_session()
        session["flux_reference_builder"]["locations"] = [
            {"id": f"loc{i}", "name": f"Place {i}", "description": f"desc {i}", "image": {"path": ""}} for i in range(1, count + 1)
        ]
        self.write_session(session)


class AssignScenesTests(Base):
    def test_blocks_repeat_each_location_then_move_on_and_loop(self):
        self.add_locations(2)
        result = ref_orch.assign_scenes("Song", {"location_pattern": "blocks", "location_block_size": 4})
        mapping = self.read_session()["flux_reference_builder"]["scene_map"]
        order = [mapping[f"seg_{i}"] for i in range(10)]
        self.assertEqual(order, ["loc1"] * 4 + ["loc2"] * 4 + ["loc1"] * 2)
        self.assertEqual(result["scenes_assigned"], 10)
        self.assertEqual(result["location_use"], {"Place 1": 6, "Place 2": 4})

    def test_leave_characters_unchanged_touches_no_character_mapping(self):
        self.add_locations(2)
        ref_orch.assign_scenes("Song", {"character_pattern": "unchanged", "location_pattern": "rotate"})
        refs = self.read_session()["flux_reference_builder"]
        self.assertEqual(refs.get("subject_scene_map", {}), {})
        self.assertEqual([refs["scene_map"][f"seg_{i}"] for i in range(4)], ["loc1", "loc2", "loc1", "loc2"])
        self.assertTrue(refs["use_location_references"])

    def test_existing_mappings_are_only_replaced_when_asked(self):
        self.add_locations(2)
        session = self.read_session()
        session["flux_reference_builder"]["scene_map"] = {"seg_0": "loc2"}
        self.write_session(session)
        ref_orch.assign_scenes("Song", {"location_pattern": "blocks", "location_block_size": 5})
        self.assertEqual(self.read_session()["flux_reference_builder"]["scene_map"]["seg_0"], "loc2", "kept")
        ref_orch.assign_scenes("Song", {"location_pattern": "blocks", "location_block_size": 5, "replace_existing": True})
        self.assertEqual(self.read_session()["flux_reference_builder"]["scene_map"]["seg_0"], "loc1", "replaced")

    def test_character_patterns_and_the_no_character_flag(self):
        session = self.read_session()
        session["flux_reference_builder"]["subjects"].append({"id": "maya", "name": "Maya", "image": {"path": self.image}})
        session["segments"][1]["no_character_present"] = True
        self.write_session(session)
        ref_orch.assign_scenes("Song", {"character_pattern": "rotate"})
        mapping = self.read_session()["flux_reference_builder"]["subject_scene_map"]
        self.assertEqual(mapping["seg_0"], ["darrel"])
        self.assertNotIn("seg_1", mapping, "scenes marked as having no character get none")
        self.assertEqual(mapping["seg_2"], ["darrel"])  # rotation counts target scenes, so scene 2 is the third target
        self.assertEqual(mapping["seg_3"], ["maya"])

    def test_extra_views_of_a_character_are_not_separate_people(self):
        session = self.read_session()
        session["flux_reference_builder"]["subjects"].append({"id": "darrel_side", "name": "Darrel side", "extra_reference_for": "darrel", "image": {"path": self.image}})
        self.write_session(session)
        ref_orch.assign_scenes("Song", {"character_pattern": "rotate"})
        mapping = self.read_session()["flux_reference_builder"]["subject_scene_map"]
        self.assertTrue(all(value == ["darrel"] for value in mapping.values()))

    def test_range_and_selected_scopes(self):
        self.add_locations(1)
        ref_orch.assign_scenes("Song", {"location_pattern": "rotate", "scope": "range", "range_start": 3, "range_end": 4})
        self.assertEqual(sorted(self.read_session()["flux_reference_builder"]["scene_map"]), ["seg_2", "seg_3"])
        ref_orch.assign_scenes("Song", {"location_pattern": "rotate", "scope": "selected", "scene_ids": ["seg_9", "1"]})
        self.assertEqual(sorted(self.read_session()["flux_reference_builder"]["scene_map"]), ["seg_0", "seg_2", "seg_3", "seg_9"])
        with self.assertRaises(errors.ValidationError):
            ref_orch.assign_scenes("Song", {"location_pattern": "rotate", "scope": "selected"})

    def test_random_is_repeatable_with_a_seed_and_avoids_consecutive_repeats(self):
        self.add_locations(3)
        first = ref_orch.assign_scenes("Song", {"location_pattern": "random", "seed": 5, "dry_run": True})["plan"]
        second = ref_orch.assign_scenes("Song", {"location_pattern": "random", "seed": 5, "dry_run": True})["plan"]
        self.assertEqual(first, second)
        ids = [item["location_id"] for item in first]
        self.assertTrue(all(a != b for a, b in zip(ids, ids[1:])))

    def test_dry_run_changes_nothing(self):
        self.add_locations(2)
        before = self.read_session()
        result = ref_orch.assign_scenes("Song", {"location_pattern": "rotate", "dry_run": True})
        self.assertTrue(result["dry_run"])
        self.assertEqual(self.read_session(), before)

    def test_unknown_patterns_are_rejected(self):
        with self.assertRaises(errors.ValidationError):
            ref_orch.assign_scenes("Song", {"location_pattern": "spiral"})


class ExtractLocationsTests(Base):
    def run_extract(self, scout_result=None, params=None):
        captured = {}

        def scout(payload):
            captured.update(payload)
            return scout_result or {"locations": [{"name": "Rooftop Lounge", "description": "A neon rooftop."}, {"name": "Hotel Lobby", "description": "Marble lobby."}], "used_model": payload.get("lmstudio_model")}

        with patch.object(ref_orch.img_gen, "_generate_lm_scout_locations", scout, create=True):
            result = ref_orch.extract_locations("Song", params or {"style_theme": "Los Angeles nightlife, neon, night time"})
        return result, captured

    def test_locations_are_added_with_ids_and_the_style_notes_are_saved(self):
        result, _ = self.run_extract()
        refs = self.read_session()["flux_reference_builder"]
        self.assertEqual([loc["name"] for loc in refs["locations"]], ["Rooftop Lounge", "Hotel Lobby"])
        self.assertTrue(all(loc["id"].startswith("loc_") for loc in refs["locations"]))
        self.assertEqual(refs["location_style_theme"], "Los Angeles nightlife, neon, night time")
        self.assertTrue(refs["use_location_references"])
        self.assertEqual((result["added"], result["updated"], result["total_locations"]), (2, 0, 2))

    def test_the_request_carries_lyrics_style_cast_and_existing_locations(self):
        with open(os.path.join(self.folder, "project_context", "themestyle.txt"), "w", encoding="utf-8") as handle:
            handle.write("cinematic realism")
        self.add_locations(1)
        _, request = self.run_extract()
        self.assertIn("Scene 1: line 1", request["lyrics_text"])
        self.assertIn("Global theme/style:\ncinematic realism", request["style_theme"])
        self.assertIn("Location extraction notes:\nLos Angeles nightlife", request["style_theme"])
        self.assertIn("Darrel (character)", request["subject_context"])
        self.assertEqual([item["name"] for item in request["existing_locations"]], ["Place 1"])

    def test_it_uses_the_loaded_lm_studio_model_never_the_saved_one(self):
        _, request = self.run_extract()
        self.assertEqual(request["lmstudio_model"], "gemma-4-e4b-it", "the saved model 'bonsai-27b' is not loaded")
        self.assertEqual(request["lmstudio_context_limit"], 32768)
        self.assertFalse(request["unload_after"])
        self.assertEqual(set(self.lm.requests), {("GET", "/api/v0/models")}, "only the read-only model list is requested")

    def test_many_locations_added_at_once_get_distinct_ids(self):
        many = {"locations": [{"name": f"Place number {i}", "description": "x"} for i in range(200)]}
        with patch.object(ref_orch.random, "randint", lambda a, b: 7):  # worst case: every id draw collides
            self.run_extract(many)
        ids = [loc["id"] for loc in self.read_session()["flux_reference_builder"]["locations"]]
        self.assertEqual(len(ids), 200)
        self.assertEqual(len(set(ids)), 200)

    def test_existing_locations_are_not_duplicated_and_empty_descriptions_are_filled(self):
        session = self.read_session()
        session["flux_reference_builder"]["locations"] = [{"id": "loc_old", "name": "rooftop lounge", "description": "", "image": {}}]
        self.write_session(session)
        result, _ = self.run_extract()
        locations = self.read_session()["flux_reference_builder"]["locations"]
        self.assertEqual(len(locations), 2)
        self.assertEqual(locations[0]["id"], "loc_old")
        self.assertEqual(locations[0]["description"], "A neon rooftop.")
        self.assertEqual((result["added"], result["updated"]), (1, 1))

    def test_no_lyrics_is_a_clear_error(self):
        session = self.read_session()
        for segment in session["segments"]:
            segment["lyric_text"] = ""
        self.write_session(session)
        with self.assertRaises(errors.ValidationError):
            self.run_extract()

    def test_nothing_is_called_when_lm_studio_has_nothing_loaded(self):
        self.lm.close()
        called = []
        with patch.object(ref_orch.img_gen, "_generate_lm_scout_locations", lambda payload: called.append(payload), create=True):
            with self.assertRaises(errors.AgentApiError) as ctx:
                ref_orch.extract_locations("Song", {})
        self.assertEqual(ctx.exception.code, "LLM_UNAVAILABLE")
        self.assertEqual(called, [])


class DescribeTests(Base):
    def run_describe(self, kind="subjects", ref_id="darrel"):
        captured = {}

        def describe(payload):
            captured.update(payload)
            return {"description": "a man in a black jacket", "used_model": payload.get("lmstudio_model")}

        with patch.object(ref_orch.img_gen, "_generate_builder_reference_description", describe):
            return ref_orch.describe_reference("Song", kind, ref_id), captured

    def test_the_description_is_saved_on_the_subject(self):
        result, request = self.run_describe()
        self.assertEqual(self.read_session()["flux_reference_builder"]["subjects"][0]["description"], "a man in a black jacket")
        self.assertEqual(request["reference_type"], "character")
        self.assertEqual(request["name"], "Darrel")
        self.assertEqual(request["image_path"], self.image)
        self.assertEqual(request["lmstudio_model"], "gemma-4-e4b-it")
        self.assertFalse(request["unload_after"])
        self.assertEqual(result["description"], "a man in a black jacket")

    def test_a_missing_image_or_reference_is_a_clear_error(self):
        session = self.read_session()
        session["flux_reference_builder"]["subjects"][0]["image"] = {"path": ""}
        self.write_session(session)
        with self.assertRaises(errors.ValidationError):
            self.run_describe()
        with self.assertRaises(errors.ValidationError):
            self.run_describe(ref_id="nobody")
        with self.assertRaises(errors.ValidationError):
            self.run_describe(kind="props")

    def test_locations_are_described_as_places(self):
        self.add_locations(1)
        session = self.read_session()
        session["flux_reference_builder"]["locations"][0]["image"] = {"path": self.image}
        self.write_session(session)
        _, request = self.run_describe("locations", "loc1")
        self.assertEqual(request["reference_type"], "location")


class JobTests(Base):
    def test_the_jobs_run_and_report(self):
        async def go():
            manager = jobs_mod.JobManager()
            ref_orch.register_reference_orchestrator_handlers(manager)
            with patch.object(ref_orch.img_gen, "_generate_builder_reference_description", lambda payload: {"description": "d"}):
                job = manager.submit_job("reference.describe", project_id="Song", params={"kind": "subjects", "ref_id": "darrel"}, is_gpu=False)
                for _ in range(300):
                    if job.is_terminal():
                        break
                    await asyncio.sleep(0.02)
            return job

        job = asyncio.run(go())
        self.assertEqual(job.status, jobs_mod.JobStatus.SUCCEEDED, job.error)
        self.assertEqual(job.result["description"], "d")


if __name__ == "__main__":
    unittest.main()
