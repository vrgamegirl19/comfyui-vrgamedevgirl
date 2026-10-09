"""Timed Timeline Notes through the Agent API: saved as the Builder's timeline_markers and used by Story Arc."""

import importlib
import json
import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
story = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.storyboard_orchestrator")
builder_runner = importlib.import_module(f"{pkg_name}.llm.builder_runner")
markers_service = importlib.import_module(f"{pkg_name}.builder.timeline_markers")
from test_agent_api_references_llm import Base  # noqa: E402

PROJECT = "Song"


class FakeServer:
    def __init__(self):
        self.events = []

    def send_sync(self, event, data, sid=None):
        self.events.append((event, data))


class TimelineNoteTests(Base):
    def setUp(self):
        super().setUp()
        session = self.read_session()
        # A note made in the Builder, kept exactly as it was.
        session["timeline_markers"] = [
            {"id": "mark_ui_1", "start": 12.0, "end": None, "type": "chorus", "label": "Drop", "note": "Lights cut out"},
        ]
        session["active_timeline_marker_id"] = "mark_ui_1"
        self.write_session(session)
        self.server = FakeServer()
        server_module = types.ModuleType("server")
        server_module.PromptServer = types.SimpleNamespace(instance=self.server)
        modules = patch.dict(sys.modules, {"server": server_module})
        modules.start()
        self.addCleanup(modules.stop)

    def markers(self):
        return self.read_session()["timeline_markers"]

    def test_create_range_and_point_notes_keeping_existing_notes(self):
        rev = self.read_session()["revision"]
        ranged = mutations.create_timeline_note(PROJECT, {"start": 4.5, "end": 9, "type": "verse", "label": "Fight", "note": "They argue"},
                                                if_match_revision=rev)["note"]
        point = mutations.create_timeline_note(PROJECT, {"id": "mark_api_point", "start": 20, "note": "Door slams"})["note"]
        self.assertEqual((ranged["start"], ranged["end"], ranged["type"], ranged["label"]), (4.5, 9.0, "verse", "Fight"))
        self.assertEqual(ranged["scene_ids"], ["seg_1", "seg_2"], "the scenes the range overlaps")
        self.assertIsNone(point["end"])
        self.assertEqual(point["scene_ids"], ["seg_5"])
        saved = self.markers()
        self.assertEqual([m["id"] for m in saved], [ranged["id"], "mark_ui_1", "mark_api_point"], "sorted by start")
        self.assertEqual(saved[1], {"id": "mark_ui_1", "start": 12.0, "end": None, "type": "chorus", "label": "Drop", "note": "Lights cut out"})
        self.assertTrue(ranged["id"].startswith("mark_"))
        listed = mutations.list_timeline_notes(PROJECT)
        self.assertEqual(listed["count"], 3)
        self.assertEqual(listed["notes"][0]["scene_ids"], ["seg_1", "seg_2"])

    def test_update_changes_only_sent_fields_and_null_end_makes_a_point(self):
        note = mutations.create_timeline_note(PROJECT, {"start": 1, "end": 3, "label": "Keep", "note": "first"})["note"]
        updated = mutations.update_timeline_note(PROJECT, note["id"], {"note": "second"})["note"]
        self.assertEqual((updated["label"], updated["note"], updated["end"]), ("Keep", "second", 3.0))
        updated = mutations.update_timeline_note(PROJECT, note["id"], {"end": None})["note"]
        self.assertIsNone(updated["end"])
        updated = mutations.update_timeline_note(PROJECT, note["id"], {"note": ""})["note"]
        self.assertEqual(updated["note"], "", "a cleared note stays cleared")
        self.assertEqual(next(m for m in self.markers() if m["id"] == note["id"])["note"], "")

    def test_delete_and_errors(self):
        result = mutations.delete_timeline_note(PROJECT, "mark_ui_1")
        self.assertEqual(result["deleted"]["id"], "mark_ui_1")
        self.assertEqual(self.markers(), [])
        self.assertEqual(self.read_session()["active_timeline_marker_id"], "")
        with self.assertRaises(errors.TimelineNoteNotFoundError):
            mutations.delete_timeline_note(PROJECT, "mark_ui_1")
        with self.assertRaises(errors.TimelineNoteNotFoundError):
            mutations.update_timeline_note(PROJECT, "missing", {"note": "x"})
        for fields in ({"end": 3}, {"start": 5, "end": 5}, {"start": -1}, {"start": "soon"}, {"start": 1, "colour": "red"},
                       {"start": 1, "note": ["list"]}):
            with self.subTest(fields=fields), self.assertRaises(errors.ValidationError):
                mutations.create_timeline_note(PROJECT, fields)
        with self.assertRaises(errors.ValidationError):
            mutations.create_timeline_note(PROJECT, {"id": "dup", "start": 1})
            mutations.create_timeline_note(PROJECT, {"id": "dup", "start": 2})
        rev = self.read_session()["revision"]
        with self.assertRaises(errors.RevisionConflictError):
            mutations.create_timeline_note(PROJECT, {"start": 1}, if_match_revision=rev - 1)

    def test_open_windows_get_the_changed_note_ids(self):
        note = mutations.create_timeline_note(PROJECT, {"start": 2, "note": "n"})["note"]
        name, data = self.server.events[-1]
        self.assertEqual(name, "vrgdg.project_changed")
        self.assertEqual(data["change"], {"kind": "timeline_markers", "marker_ids": [note["id"]]})
        self.assertEqual(data["revision"], self.read_session()["revision"])

    def test_notes_reach_story_arc_with_their_ranges_and_scenes(self):
        session = self.read_session()
        sections = ["Verse", "Verse", "Chorus", "Chorus", "Bridge", "Bridge", "Bridge", "Bridge", "Bridge", "Bridge"]
        for segment, section in zip(session["segments"], sections):
            segment["lyric_section"] = section
        session["lyric_mapper"] = {"source_text": "[Verse]\nline 1\nline 2\n[Chorus]\nline 3\nline 4\n[Bridge]\nline 5"}
        self.write_session(session)
        mutations.create_timeline_note(PROJECT, {"start": 8, "end": 16, "note": "CHORUS REVEAL SENTINEL"})
        mutations.update_timeline_note(PROJECT, "mark_ui_1", {"start": 17, "note": "BRIDGE POINT SENTINEL"})
        captured = []

        def run_text(payload, instruction, **kwargs):
            captured.append(instruction)
            if "Story Arc Scene" in kwargs.get("label", ""):
                return "She waits by the window.", {}
            text = {"Verse": "She waits.", "Chorus": "The lights reveal him.", "Bridge": "The door slams."}
            return json.dumps(text) if kwargs.get("json_schema") else "\n\n".join(f"{k}:\n{v}" for k, v in text.items()), {}

        with patch.object(builder_runner, "_run_builder_text_llm", run_text), \
                patch.object(builder_runner, "_runner_supports_json_schema", lambda payload: True):
            result = story.create_story_arc(PROJECT, {"story_idea": "A woman waits for news"})
        self.assertIn("Chorus", result["story_arc"])
        first = captured[0]
        self.assertIn("CHORUS REVEAL SENTINEL", first)
        self.assertIn("BRIDGE POINT SENTINEL", first)
        heading = first.index("USER TIMELINE NOTES")
        decoded, _end = json.JSONDecoder().raw_decode(first, first.index("[", heading))
        notes = {item["note"]: item for item in decoded}
        self.assertEqual((notes["CHORUS REVEAL SENTINEL"]["start"], notes["CHORUS REVEAL SENTINEL"]["end"]), (8.0, 16.0))
        self.assertEqual(notes["CHORUS REVEAL SENTINEL"]["scene_numbers"], [3, 4], "the Chorus scenes")
        self.assertIsNone(notes["BRIDGE POINT SENTINEL"]["end"])
        self.assertEqual(notes["BRIDGE POINT SENTINEL"]["scene_numbers"], [5], "a point note marks one moment")


class MarkerTwinTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required to compare with the Builder")
    def test_python_normalizes_saved_notes_like_the_builder(self):
        source = (ROOT / "web" / "music_video_builder" / "timeline_state.mjs").read_text(encoding="utf-8")
        start = source.index("export function normalizeTimelineMarkers")
        end = source.index("\nexport function", start + 10)
        markers = [
            {"id": "b", "start": 9, "end": 4, "type": "", "label": "", "note": " late "},
            {"id": "a", "start": 2.5, "end": 6, "type": "chorus", "label": "Hook", "note": "x"},
            {"id": "c", "start": 0, "end": None, "type": "beat", "label": "Hit", "note": ""},
        ]
        script = source[start:end].replace("export ", "") + f"\nconsole.log(JSON.stringify(normalizeTimelineMarkers({json.dumps(markers)})));"
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(markers_service.normalize_markers(markers), json.loads(result.stdout))


if __name__ == "__main__":
    unittest.main()
