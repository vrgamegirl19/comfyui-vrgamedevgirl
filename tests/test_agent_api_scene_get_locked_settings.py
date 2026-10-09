"""GET /projects/{pid}/scenes/{sid} shows the scene's Audio Mask and its locked MiniMax H3 settings, read-only.

The Video Builder saves them on the scene as ``audio_mask`` (Audio Mask, ``web/music_video_builder/audio_mask.mjs``),
``use_scene_minimax_h3_settings`` + ``minimax_h3_settings`` (scene settings lock) and, after a render,
``minimax_h3_continuity_mode_used`` + ``minimax_h3_continuity_source_scene_id`` (``video_render.mjs``).
"""

import copy
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

projects = importlib.import_module(f"{ROOT.name}.agent_api.projects")
mutations = importlib.import_module(f"{ROOT.name}.agent_api.mutations")
errors = importlib.import_module(f"{ROOT.name}.agent_api.errors")
from test_agent_api_references_llm import Base  # noqa: E402

AUDIO_MASK = {
    "version": 2, "user_off": False, "enabled": True, "model_name": "htdemucs_6s", "input_gain_db": 0,
    "stems": {
        "vocals": {"mask": True, "regions": [{"start": 0.5, "end": 2.25, "fade_ms": 30}], "db": 0, "mute": False},
        "drums": {"mask": False, "regions": [], "db": -6, "mute": False},
    },
}
LOCKED = {"video_mode": "reference_to_video", "render_pass": "two_pass",
          "continuity_mode": "latent_continuation_masked", "latent_context_frames": 39}

SCENE_FIELDS = ("audio_mask", "use_scene_minimax_h3_settings", "minimax_h3_settings",
                "minimax_h3_continuity_mode_used", "minimax_h3_continuity_source_scene_id")


class SceneGetLockedSettingsTests(Base):
    def setUp(self):
        super().setUp()
        session = self.read_session()
        session["minimax_h3_settings"] = {"video_mode": "reference_to_video", "continuity_mode": "off"}
        scene = session["segments"][9]
        scene.update({
            "audio_mask": copy.deepcopy(AUDIO_MASK),
            "use_scene_minimax_h3_settings": True,
            "minimax_h3_settings": dict(LOCKED),
            "minimax_h3_continuity_mode_used": "latent_continuation_masked",
            "minimax_h3_continuity_source_scene_id": "seg_8",
        })
        self.write_session(session)

    def test_scene_get_shows_the_audio_mask_and_locked_minimax_settings(self):
        scene = projects.get_scene_detail("Song", "seg_9")
        self.assertEqual(scene["audio_mask"], AUDIO_MASK)
        self.assertIs(scene["use_scene_minimax_h3_settings"], True)
        self.assertEqual(scene["minimax_h3_settings"], LOCKED)
        self.assertEqual(scene["minimax_h3_settings"]["continuity_mode"], "latent_continuation_masked")
        self.assertEqual(scene["minimax_h3_continuity_mode_used"], "latent_continuation_masked")
        self.assertEqual(scene["minimax_h3_continuity_source_scene_id"], "seg_8")

    def test_a_scene_without_them_reports_empty_values(self):
        scene = projects.get_scene_detail("Song", "1")
        self.assertIsNone(scene["audio_mask"])
        self.assertIs(scene["use_scene_minimax_h3_settings"], False)
        self.assertIsNone(scene["minimax_h3_settings"])
        # The Builder reads a missing record as "off" (normalizeMiniMaxH3ContinuityMode).
        self.assertEqual(scene["minimax_h3_continuity_mode_used"], "off")
        self.assertEqual(scene["minimax_h3_continuity_source_scene_id"], "")

    def test_a_retired_spelling_of_the_recorded_mode_is_shown_the_way_the_builder_reads_it(self):
        session = self.read_session()
        session["segments"][9]["minimax_h3_continuity_mode_used"] = "latent_masked"
        self.write_session(session)
        self.assertEqual(projects.get_scene_detail("Song", "10")["minimax_h3_continuity_mode_used"], "latent_continuation_masked")

    def test_the_response_is_a_copy_not_the_session(self):
        scene = projects.get_scene_detail("Song", "10")
        scene["audio_mask"]["enabled"] = False
        scene["minimax_h3_settings"]["continuity_mode"] = "off"
        again = projects.get_scene_detail("Song", "10")
        self.assertTrue(again["audio_mask"]["enabled"])
        self.assertEqual(again["minimax_h3_settings"]["continuity_mode"], "latent_continuation_masked")

    def test_the_scene_list_stays_lean(self):
        listed = projects.get_project_scenes("Song")[9]
        for key in SCENE_FIELDS:
            self.assertNotIn(key, listed)

    def test_audio_mask_and_the_continuity_record_are_not_patchable(self):
        for field, value in (("audio_mask", {"enabled": False}), ("minimax_h3_continuity_mode_used", "off"),
                             ("minimax_h3_continuity_source_scene_id", "")):
            with self.assertRaises(errors.ValidationError, msg=field):
                mutations.patch_scene("Song", "seg_9", {field: value})
        saved = self.read_session()["segments"][9]
        self.assertEqual(saved["audio_mask"], AUDIO_MASK)
        self.assertEqual(saved["minimax_h3_continuity_mode_used"], "latent_continuation_masked")


if __name__ == "__main__":
    unittest.main()
