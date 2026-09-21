import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
BACKEND_SOURCE = (ROOT / "VRGDG_MusicVideoBuilderNodes.py").read_text(encoding="utf-8")
INSTRUCTION_SOURCE = (ROOT / "VRGDG_MiniMaxH3PromptInstructions.py").read_text(encoding="utf-8")


class MiniMaxFrameContinuityPromptTests(unittest.TestCase):
    def test_setting_is_saved_and_only_enabled_for_latent_modes(self):
        self.assertIn("continuity_prompt_from_last_frame: false", UI_SOURCE)
        self.assertIn("continuity_prompt_from_last_frame: miniMaxContinuityPromptFromLastFrame.input.checked", UI_SOURCE)
        self.assertIn("isMiniMaxH3LatentContinuationMode(settings.continuity_mode)", UI_SOURCE)
        self.assertIn('sceneSlotNumber(segment) <= 1', UI_SOURCE)

    def test_previous_final_frame_is_first_vision_input(self):
        self.assertIn('const visionImages = [{ path: framePath, frame_continuity_source: true }]', UI_SOURCE)
        self.assertIn("Attached Picture 1 is the previous rendered scene's actual final frame", UI_SOURCE)
        self.assertIn("HIGHEST PRIORITY", UI_SOURCE)
        self.assertIn("frame_continuity_prompt", BACKEND_SOURCE)

    def test_location_change_contract_forbids_nonphysical_replacement(self):
        for required in (
            "connect both locations as coherent 3D space",
            "Reveal the new location progressively",
            "old location visible",
            "fade, dissolve, morph, teleport",
            "abrupt background swap",
        ):
            self.assertIn(required, UI_SOURCE)

    def test_generation_retries_ten_times_and_never_accepts_blank(self):
        self.assertIn("attempt <= 10", UI_SOURCE)
        self.assertIn("returned an empty frame-to-frame continuity prompt", UI_SOURCE)
        self.assertIn("could not create a valid frame-to-frame continuity prompt after 10 attempts", UI_SOURCE)
        generation_call = UI_SOURCE.index("await createMiniMaxH3FrameContinuityPrompt")
        prompt_validation = UI_SOURCE.index("if (!prompt) throw new Error", generation_call)
        self.assertLess(generation_call, prompt_validation)

    def test_feature_forces_one_continuous_shot(self):
        self.assertIn("frequency: 0, cut_times_seconds: [], cue_driven: false", UI_SOURCE)
        self.assertIn("All listed cues occur inside the same uninterrupted shot", UI_SOURCE)
        self.assertIn("Continuing without a cut from the previous shot, the camera stays on course", UI_SOURCE)

    def test_llm_preset_contains_no_inactive_setting_branches(self):
        core = INSTRUCTION_SOURCE.split("_MINIMAX_H3_TEXT_TO_VIDEO_MODE", 1)[0]
        for inactive_branch in (
            "If a lyric/dialogue line",
            "If the scene context supplies a vocal cue map",
            "For instrumental or otherwise non-vocal intervals",
            "In multi-subject vocal scenes",
            "If the scene is visual-only",
        ):
            self.assertNotIn(inactive_branch, core)
        self.assertIn("already contains the Builder's resolved mode", core)
        self.assertNotIn("is_minimax_h3_shot_json_task and not frame_continuity_prompt", BACKEND_SOURCE)

    def test_character_motion_rule_is_resolved_for_current_performance(self):
        self.assertIn("visualOnly || segment?.no_character_present || !lyricText", UI_SOURCE)
        self.assertIn("Do not add singing, speaking, or lip sync", UI_SOURCE)
        self.assertIn('performanceMode === "speaking"', UI_SOURCE)
        self.assertIn("required dialogue lip sync", UI_SOURCE)
        self.assertIn("required singing lip sync", UI_SOURCE)


if __name__ == "__main__":
    unittest.main()
