"""API story beats and prompts send the LLM the same scene-card context the Storyboard UI sends.

Each test captures the text that actually reaches the LLM runner. Timed Timeline Notes stay out of these
per-scene requests; only Story Arc planning reads them (test_agent_api_timeline_notes.py).
"""

import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
story = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.storyboard_orchestrator")
mm = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.minimax_prompt_orchestrator")
image_inputs = importlib.import_module(f"{pkg_name}.agent_api.image_prompt_inputs")
video_inputs = importlib.import_module(f"{pkg_name}.agent_api.video_prompt_inputs")
builder_runner = importlib.import_module(f"{pkg_name}.llm.builder_runner")
img_mod = importlib.import_module(f"{pkg_name}.llm.image_prompt_generation")
vid_mod = importlib.import_module(f"{pkg_name}.llm.video_prompt_generation")
from test_agent_api_references_llm import Base  # noqa: E402

PROJECT = "Song"
CARD = {
    "timeline_note": "DIRECTOR SENTINEL hold on the hands",
    "notes": "PLANNING SENTINEL practical lamps",
    "i2v_notes": "VIDEO SENTINEL slow push in",
    "shot_type": "SHOT SENTINEL medium close-up",
    "camera_motion": "CAMERA SENTINEL dolly left",
    "character_motion": "CHARACTER SENTINEL turns away",
    "performance_style": "PERFORMANCE SENTINEL restrained",
    "facial_performance_custom": "FACE SENTINEL jaw tight",
    "audio_direction": "AUDIO SENTINEL rain on glass",
    "continuity": "CONTINUITY SENTINEL coat stays wet",
    "include_microphone": False,
    "prompt_summary": "SUMMARY SENTINEL card only",
    "trigger_phrase": "TRIGGER SENTINEL",
}
DIRECTIONS = ("DIRECTOR", "PLANNING", "VIDEO", "CAMERA", "CHARACTER", "PERFORMANCE", "FACE", "AUDIO", "CONTINUITY", "SUMMARY")


class SceneCardLlmInputTests(Base):
    def setUp(self):
        super().setUp()
        session = self.read_session()
        refs = session["flux_reference_builder"]
        refs["locations"] = [{"id": "loc1", "name": "Rooftop", "description": "a neon rooftop", "image": {"path": ""}}]
        refs["subject_scene_map"] = {s["id"]: ["darrel"] for s in self.segments}
        refs["scene_map"] = {s["id"]: "loc1" for s in self.segments}
        session["timeline_markers"] = [{"id": "mark_1", "start": 0, "end": 8, "note": "TIMED SENTINEL storm arrives"}]
        session["builder_story_layer"] = {"enabled": True, "overall_story_idea": "idea", "user_story_arc": "arc"}
        self.write_session(session)
        # The API edit saves the card (and Storyboard-only fields) like a UI edit.
        mutations.patch_scene(PROJECT, "seg_0", CARD)
        self.captured = []

    def capture(self, reply):
        def run_text(payload, instruction, **kwargs):
            self.captured.append(instruction)
            return reply, {"runner": "test"}
        return run_text

    def assertDirections(self, text, expected=DIRECTIONS):
        for word in expected:
            self.assertIn(f"{word} SENTINEL", text, word)
        self.assertNotIn("TIMED SENTINEL", text, "timed Timeline Notes are not per-scene directions")

    def scene_card_from(self, text):
        start = text.index("Complete scene_card:\n") + len("Complete scene_card:\n")
        card, _end = json.JSONDecoder().raw_decode(text, start)
        return card

    def test_scene_beats_get_the_complete_card(self):
        with patch.object(builder_runner, "_run_builder_text_llm", self.capture("She steps to the rail.")):
            story.create_scene_beats(PROJECT, {"scene_ids": ["seg_0"]})
        instruction = self.captured[0]
        self.assertDirections(instruction)
        self.assertIn('"scene_card"', instruction)
        self.assertIn('"director_note": "DIRECTOR SENTINEL hold on the hands"', instruction)
        self.assertIn("Read every populated field in scene_card", instruction)

    def test_image_prompts_get_the_card_and_stay_a_still_frame(self):
        inputs = image_inputs.scene_image_prompt_inputs(self.read_session(), "seg_0", "zimage")
        payload = {**inputs, "text_gemma_runner": "lm_studio", "lm_studio_model": "gemma-4-e4b-it", "project_folder": self.folder}
        with patch.object(img_mod, "_run_builder_text_llm", self.capture("A woman at a rainy rooftop rail.")), \
                patch.object(img_mod, "_repair_and_validate_builder_gemma_prompt", lambda payload, text, label: text):
            img_mod._generate_builder_t2i_prompt(payload)
        instruction = self.captured[0]
        self.assertDirections(instruction)
        self.assertIn("Director note (timeline):\nDIRECTOR SENTINEL", instruction)
        self.assertIn("motion or audio context does not make a still image animated or audible", instruction)
        self.assertTrue(instruction.rstrip().endswith(image_inputs.IMAGE_PREP_RULE), "the still-image rule stays last")
        card = self.scene_card_from(instruction)
        self.assertEqual(card["timeline_note"], CARD["timeline_note"])
        self.assertIs(card["include_microphone"], False)
        self.assertEqual(card["trigger_phrase"], "TRIGGER SENTINEL")

    def test_video_prompts_get_motion_audio_and_performance_directions(self):
        inputs = video_inputs.scene_video_prompt_inputs(self.read_session(), "seg_0", "i2v")
        payload = {**inputs, "text_gemma_runner": "lm_studio", "lm_studio_model": "gemma-4-e4b-it", "project_folder": self.folder}
        with patch.object(vid_mod, "_run_builder_text_llm", self.capture("She turns away as rain streaks the glass.")), \
                patch.object(vid_mod, "_repair_and_validate_builder_gemma_prompt", lambda payload, text, label: text):
            vid_mod._generate_builder_i2v_prompt(payload)
        instruction = "\n".join(self.captured)
        self.assertDirections(instruction)
        self.assertIn("Exact manual audio / sound direction:\nAUDIO SENTINEL", instruction)
        self.assertIn("Storyboard character motion guidance:\nCHARACTER SENTINEL", instruction)
        self.assertEqual(self.scene_card_from(instruction)["motion_summary"], CARD["i2v_notes"])

    def test_minimax_prompts_get_the_builder_sections_and_the_card(self):
        tasks = []

        def fake_writer(request):
            tasks.append(request["t2i_prompt"])
            return {"prompt": '{"shots":[{"description":"<Subject 1> (Darrel) steps to the rail as the camera dollies left."}]}'}

        with patch.object(mm.vid_gen, "_generate_builder_t2v_prompt", fake_writer):
            mm.create_minimax_prompts(PROJECT, {"scene_ids": ["seg_0"]})
        task = tasks[0]
        self.assertIn("Motion/camera request:\nVIDEO SENTINEL slow push in", task)
        self.assertIn("Manual audio direction for staging only:\nAUDIO SENTINEL", task)
        self.assertIn("Continuity notes for staging only:\nCONTINUITY SENTINEL", task)
        self.assertIn("Storyboard Builder context:\nDirector note (timeline):\nDIRECTOR SENTINEL", task)
        self.assertDirections(task)
        self.assertEqual(self.scene_card_from(task)["prompt_summary"], CARD["prompt_summary"])


if __name__ == "__main__":
    unittest.main()
