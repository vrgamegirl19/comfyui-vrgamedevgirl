"""PUT /story/settings accepts every scene default and story field the Builder saves."""

import importlib
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
story = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.storyboard_orchestrator")
from builder_source import function_source, read_builder_module  # noqa: E402
from test_agent_api_references_llm import Base  # noqa: E402

# Storyboard defaults as an older Cut and Gun session saved them (normalizeBuilderStoryboardDefaults output).
OLD_PROJECT_DEFAULTS = {
    "global_consistency_phrase": "gritty 16mm western noir",
    "camera_motion_speed": 5,
    "character_motion_speed": 6,
    "minimax_h3_cut_frequency": "medium",
    "camera_guidance": "",
    "character_guidance": "",
    "performance_style": "intense",
    "short_film_planning_mode": "guided_film",
    "camera_flow": "custom",
    "custom_camera_flow_sequence": "wide, medium, close",
    "image_shot_flow": "cinematic",
    "image_aesthetic": "film still",
    "video_style": "Cinematic realism",
    "video_style_custom": "",
    "temporal_world_effect": "",
    "temporal_world_effect_custom": "",
    "temporal_allow_background_extras": True,
    "temporal_background_intensity": 8,
    "temporal_environment_time_passage": True,
    "temporal_protected_characters": "all_referenced",
    "temporal_protected_custom": "",
    "fx_preset": "",
    "fx_custom_json": "",
}


def builder_return_keys(function_name):
    """Keys of the object a Builder normalizer returns (web/music_video_builder/model_settings.mjs)."""
    source = function_source(read_builder_module("model_settings.mjs"), function_name)
    body = source[source.rindex("return {"):]
    return set(re.findall(r"^ {4}([A-Za-z_]\w*):", body, re.M))


class StorySettingsFieldTests(Base):
    def test_allowed_defaults_are_the_builders_storyboard_defaults(self):
        builder_keys = builder_return_keys("normalizeBuilderStoryboardDefaults")
        self.assertIn("temporal_background_intensity", builder_keys)
        self.assertEqual(set(story.DEFAULT_KEYS), builder_keys)

    def test_allowed_story_fields_are_the_builders_story_layer(self):
        builder_keys = builder_return_keys("normalizeBuilderStoryLayer")
        self.assertIn("enabled", builder_keys)
        self.assertEqual(set(story.STORY_LAYER_KEYS), builder_keys)

    def test_an_old_projects_storyboard_defaults_save(self):
        result = story.set_story_settings("Song", {"defaults": dict(OLD_PROJECT_DEFAULTS, story_arc_detail="rich")})
        saved = self.read_session()["builder_storyboard_defaults"]
        for key, value in OLD_PROJECT_DEFAULTS.items():
            self.assertEqual(saved[key], value, key)
        self.assertEqual(saved["story_arc_detail"], "rich")
        self.assertEqual(result["defaults"]["short_film_planning_mode"], "guided_film")

    def test_story_enabled_saves(self):
        story.set_story_settings("Song", {"story": {"enabled": False, "overall_story_idea": "idea"}})
        layer = self.read_session()["builder_story_layer"]
        self.assertIs(layer["enabled"], False)
        self.assertEqual(layer["overall_story_idea"], "idea")

    def test_new_fields_are_checked_like_the_builder(self):
        story.set_story_settings("Song", {"defaults": {"temporal_background_intensity": 14,
                                                       "short_film_planning_mode": "Fully Custom"}})
        saved = self.read_session()["builder_storyboard_defaults"]
        self.assertEqual(saved["temporal_background_intensity"], 10, "intensity clamps to 0-10")
        self.assertEqual(saved["short_film_planning_mode"], "fully_custom")
        for bad in ({"temporal_protected_characters": "everyone"}, {"short_film_planning_mode": "bogus"},
                    {"temporal_allow_background_extras": "no"}, {"temporal_background_intensity": "lots"}):
            with self.assertRaises(errors.ValidationError, msg=bad):
                story.set_story_settings("Song", {"defaults": bad})
        with self.assertRaises(errors.ValidationError):
            story.set_story_settings("Song", {"story": {"enabled": "yes"}})

    def test_unknown_keys_are_still_rejected(self):
        with self.assertRaises(errors.ValidationError) as ctx:
            story.set_story_settings("Song", {"defaults": {"bogus": 1}})
        self.assertIn("temporal_background_intensity", ctx.exception.message, "the error lists what is allowed")
        with self.assertRaises(errors.ValidationError):
            story.set_story_settings("Song", {"story": {"bogus": 1}})


if __name__ == "__main__":
    unittest.main()
