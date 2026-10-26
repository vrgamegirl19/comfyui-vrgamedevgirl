"""A scene is named by its number on the timeline as well as by its id: the MCP tools and instructions say so, and the speaking routes take it."""

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mcp_server import server, tools  # noqa: E402

audio_api = importlib.import_module(f"{ROOT.name}.agent_api.scene_audio")
dialogue_api = importlib.import_module(f"{ROOT.name}.agent_api.scene_dialogue")
errors = importlib.import_module(f"{ROOT.name}.agent_api.errors")

VOICE = {"enabled": True, "voice_id": "voice1", "name": "Alice voice"}
DRAFT = {"speaker_id": "alice", "text": "Stay behind me.", "delivery": "Quiet", "allow_rewrite": False, "model_id": "eleven_v4"}


def session():
    return {
        "video_type": "speaking", "revision": 7, "text_gemma_runner": "own_server", "audio_clips": [],
        "segments": [{"id": "seg_aaa", "start": 0, "end": 4}, {"id": "seg_bbb", "start": 4, "end": 8}, {"id": "seg_ccc", "start": 8, "end": 12}],
        "flux_reference_builder": {"subjects": [{"id": "alice", "name": "Alice", "elevenlabs_voice": VOICE}]},
    }


class SceneNumberWordingTests(unittest.TestCase):
    def test_every_scene_argument_says_a_scene_has_a_number(self):
        naming = 0
        for name, tool in tools.ALL_TOOLS.items():
            properties = tool.to_mcp_dict()["inputSchema"].get("properties", {})
            for key in ("scene_id", "scene_ids"):
                if key in properties:
                    naming += 1
                    self.assertIn("number", properties[key].get("description", ""), f"{name} {key}")
        self.assertGreater(naming, 30)

    def test_the_instructions_and_the_scene_tools_tell_an_agent_to_use_the_number(self):
        self.assertIn("Scene 11", server.SERVER_INSTRUCTIONS)
        self.assertIn("scene_id", server.SERVER_INSTRUCTIONS)
        for name in ("scene_list", "scene_get"):
            self.assertIn("number", tools.ALL_TOOLS[name].description, name)

    def test_a_missing_scene_says_how_a_scene_is_named(self):
        steps = tools._derive_next_steps("SCENE_NOT_FOUND")
        self.assertTrue(any("number" in step and "scene_list" in step for step in steps), steps)


class SpeakingRoutesTakeANumberTests(unittest.TestCase):
    def test_find_scene_takes_an_id_or_a_number(self):
        saved = session()
        self.assertEqual(audio_api.find_scene(saved, "seg_bbb")["id"], "seg_bbb")
        self.assertEqual(audio_api.find_scene(saved, "2")["id"], "seg_bbb")
        self.assertEqual(audio_api.find_scene(saved, 3)["id"], "seg_ccc")
        for nothing in ("0", "4", "nope", ""):
            self.assertIsNone(audio_api.find_scene(saved, nothing), nothing)
        saved["segments"][0]["id"] = "2"
        self.assertEqual(audio_api.find_scene(saved, "2")["start"], 0, "an id wins over a number")

    def test_audio_settings_are_read_and_saved_by_number(self):
        saved = session()
        with patch.object(audio_api, "_get_active_session_and_folder", return_value=("folder", saved)), \
             patch.object(audio_api, "_persist_session", return_value={"revision": 8}) as save:
            by_id = audio_api.get_audio_settings("p", "seg_bbb")
            self.assertEqual(audio_api.get_audio_settings("p", "2"), by_id)
            with self.assertRaises(errors.SceneNotFoundError):
                audio_api.get_audio_settings("p", "4")
            by_number = audio_api.patch_audio_settings("p", {"use_project_defaults": False, "silence_after": 1}, "2", 7)
            self.assertEqual(by_number["settings"]["scene_id"], "seg_bbb")
            expected = audio_api.audio_settings_view(
                save.call_args.args[1], next(s for s in save.call_args.args[1]["segments"] if s["id"] == "seg_bbb"))
            self.assertEqual(by_number["settings"], expected)

    def test_dialogue_is_requested_by_number_for_the_scene_it_names(self):
        with patch.object(dialogue_api, "_get_active_session_and_folder", return_value=("folder", session())):
            by_id = dialogue_api.dialogue_request("p", "seg_bbb", DRAFT)
            by_number = dialogue_api.dialogue_request("p", "2", DRAFT)
            self.assertEqual(by_number["scene_id"], "seg_bbb")
            self.assertEqual(by_number, by_id)
            with self.assertRaises(errors.SceneNotFoundError):
                dialogue_api.dialogue_request("p", "4", DRAFT)


if __name__ == "__main__":
    unittest.main()
