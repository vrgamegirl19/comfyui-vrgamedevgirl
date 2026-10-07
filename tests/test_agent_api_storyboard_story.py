"""Story settings, story arc, story brief and scene beats for agents (the Storyboard's LLM steps)."""

import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
story = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.storyboard_orchestrator")
from test_agent_api_references_llm import Base  # noqa: E402


class StoryTests(Base):
    def setUp(self):
        super().setUp()
        session = self.read_session()
        refs = session["flux_reference_builder"]
        refs["locations"] = [{"id": "loc1", "name": "Rooftop", "description": "a neon rooftop", "image": {"path": ""}}]
        refs["subject_scene_map"] = {s["id"]: ["darrel"] for s in self.segments}
        refs["scene_map"] = {s["id"]: "loc1" for s in self.segments}
        session["lyric_mapper"] = {"source_text": "line 1\nline 2"}
        self.write_session(session)

    def test_scene_cards_carry_the_mapped_character_and_location(self):
        cards = story.scene_cards(self.read_session())
        self.assertEqual(len(cards), 10)
        self.assertEqual(cards[0]["subjects"], ["Darrel"])
        self.assertEqual(cards[0]["location_ref"]["name"], "Rooftop")
        self.assertEqual(cards[0]["scene_number"], 1)
        self.assertEqual(cards[3]["lyrics"], "line 4")

    def test_settings_are_saved_under_the_ui_keys(self):
        result = story.set_story_settings("Song", {
            "defaults": {"video_style": "Cinematic realism", "camera_motion_speed": 9},
            "story": {"overall_story_idea": "A man moves through LA"},
        })
        session = self.read_session()
        self.assertEqual(session["builder_storyboard_defaults"]["video_style"], "Cinematic realism")
        self.assertEqual(session["builder_storyboard_defaults"]["camera_motion_speed"], 9)
        self.assertEqual(session["builder_story_layer"]["overall_story_idea"], "A man moves through LA")
        self.assertGreater(result["revision"], 1)

    def test_unknown_settings_are_rejected(self):
        with self.assertRaises(errors.ValidationError):
            story.set_story_settings("Song", {"defaults": {"bogus": 1}})
        with self.assertRaises(errors.ValidationError):
            story.set_story_settings("Song", {"story": {"image_world_style": "bogus"}})

    def test_arc_needs_a_story_idea(self):
        with self.assertRaises(errors.ValidationError):
            story.create_story_arc("Song", {})

    def test_arc_is_written_by_the_llm_with_the_loaded_model_and_saved(self):
        seen = {}

        def fake_arc(payload):
            seen.update(payload)
            return {"story_arc": "Act one. Act two.", "used_model": payload["lmstudio_model"]}

        with patch.object(story.story_funcs, "_build_story_layer_arc", fake_arc):
            result = story.create_story_arc("Song", {"story_idea": "A man who never stays"})
        self.assertEqual(seen["lmstudio_model"], "gemma-4-e4b-it", "the loaded model, not the saved one")
        self.assertEqual(seen["story_idea"], "A man who never stays")
        self.assertEqual(len(seen["scenes"]), 10)
        self.assertEqual(seen["line_mapping_lyrics"], "line 1\nline 2")
        self.assertEqual(result["story_arc"], "Act one. Act two.")
        layer = self.read_session()["builder_story_layer"]
        self.assertEqual(layer["user_story_arc"], "Act one. Act two.")
        self.assertEqual(layer["overall_story_idea"], "A man who never stays")
        self.assertEqual({m for m, _ in self.lm.requests}, {"GET"}, "LM Studio only received read-only requests")

    def test_brief_is_saved_next_to_the_arc(self):
        story.set_story_settings("Song", {"story": {"overall_story_idea": "idea", "user_story_arc": "arc"}})
        with patch.object(story.story_funcs, "_build_story_layer_brief", lambda p: {"story_brief": "A brief."}):
            story.create_story_brief("Song", {})
        layer = self.read_session()["builder_story_layer"]
        self.assertEqual(layer["song_story_brief"], "A brief.")
        self.assertEqual(layer["overall_story_idea"], "idea")

    def test_beats_need_the_story_first(self):
        with self.assertRaises(errors.ValidationError):
            story.create_scene_beats("Song", {})

    def beats_with_fake(self, params):
        calls = []

        def fake_beat(payload):
            card = payload["storyboard_payload"]["scenes"][0]
            calls.append((card["scene_number"], payload["previous_beat"], payload["next_lyrics"], payload["lmstudio_model"]))
            return {"story_beat": f"beat {card['scene_number']}"}

        with patch.object(story.story_funcs, "_build_story_layer_scene_beat", fake_beat):
            return story.create_scene_beats("Song", params), calls

    def test_beats_are_written_in_order_with_context_and_saved_on_the_scenes(self):
        story.set_story_settings("Song", {"story": {"overall_story_idea": "idea", "user_story_arc": "arc"}})
        result, calls = self.beats_with_fake({"limit": 3})
        self.assertEqual(result["created"], 3)
        self.assertEqual([c[0] for c in calls], [1, 2, 3])
        self.assertEqual(calls[1][1], "beat 1", "the previous scene's beat is passed on")
        self.assertEqual(calls[0][2], "line 2", "the next scene's lyric is passed on")
        self.assertEqual(calls[0][3], "gemma-4-e4b-it")
        segments = self.read_session()["segments"]
        self.assertEqual([s.get("story_beat") for s in segments[:4]], ["beat 1", "beat 2", "beat 3", None])

    def test_only_missing_beats_are_written_unless_replacing(self):
        story.set_story_settings("Song", {"story": {"overall_story_idea": "idea"}})
        self.beats_with_fake({"limit": 2})
        result, calls = self.beats_with_fake({"limit": 2})
        self.assertEqual([c[0] for c in calls], [3, 4])
        result, calls = self.beats_with_fake({"limit": 1, "replace_existing": True})
        self.assertEqual([c[0] for c in calls], [1])

    def test_nothing_is_called_when_no_llm_is_loaded(self):
        story.set_story_settings("Song", {"story": {"overall_story_idea": "idea"}})
        session = self.read_session()
        session["lm_studio_base_url"] = "http://127.0.0.1:1/v1"
        self.write_session(session)
        with self.assertRaises(Exception) as ctx:
            self.beats_with_fake({"limit": 1})
        self.assertEqual(getattr(ctx.exception, "code", ""), "LLM_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
