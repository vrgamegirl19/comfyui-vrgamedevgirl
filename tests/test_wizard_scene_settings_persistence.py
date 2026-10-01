import unittest
from pathlib import Path


WEB = Path(__file__).resolve().parents[1] / "web"
PERSISTENCE_SOURCE = (WEB / "storyboard_builder" / "persistence.mjs").read_text(encoding="utf-8")
WIZARD_SOURCE = (WEB / "music_video_builder" / "wizard.mjs").read_text(encoding="utf-8")


class WizardSceneSettingsPersistenceTests(unittest.TestCase):
    def test_storyboard_load_keeps_live_builder_fields_over_the_saved_copy(self):
        # B-roll / no lip-sync and the Replace All fields must come from the live Video Builder scenes.
        self.assertIn("const liveOwned = incomingScenes.length", PERSISTENCE_SOURCE)
        for field in (
            "lyric_no_lip_sync: fresh.lyric_no_lip_sync",
            "lyric_singers: fresh.lyric_singers",
            "performance_style: fresh.performance_style",
            "facial_performance: fresh.facial_performance",
            "shot_type: fresh.shot_type || normalized.shot_type",
            "camera_motion: fresh.camera_motion || normalized.camera_motion",
        ):
            self.assertIn(field, PERSISTENCE_SOURCE)
        # liveOwned must be spread after the saved scene so it wins.
        self.assertLess(PERSISTENCE_SOURCE.index("...normalized,\n            ...liveOwned"), PERSISTENCE_SOURCE.index("id: fresh.id || normalized.id"))

    def test_no_character_flag_follows_live_scenes_when_they_exist(self):
        self.assertIn("? Boolean(fresh.no_character_present)", PERSISTENCE_SOURCE)
        self.assertIn("subject_refs: noCharacterPresent ? [] : subjectRefs", PERSISTENCE_SOURCE)

    def test_wizard_replace_all_reads_the_visible_controls_first(self):
        self.assertIn("const syncWizardStateFromControls", WIZARD_SOURCE)
        self.assertIn("imageShotSelect.value || wizardState.imageShotFlow", WIZARD_SOURCE)
        self.assertIn("cameraSelect.value || wizardState.cameraFlow", WIZARD_SOURCE)
        self.assertIn("performanceSelect.value ?? wizardState.performanceStyle", WIZARD_SOURCE)
        self.assertNotIn("wizardState.cameraFlow || cameraSelect.value", WIZARD_SOURCE)
        self.assertNotIn("wizardState.performanceStyle ?? performanceSelect.value", WIZARD_SOURCE)


if __name__ == "__main__":
    unittest.main()
