import unittest
from pathlib import Path

from builder_source import function_source, read_builder_source


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = read_builder_source()

SETTINGS_SOURCE = function_source(BUILDER_SOURCE, "openSettingsModal")

WIZARD_SOURCE = function_source(BUILDER_SOURCE, "openWizardFromBuilder")


class BuilderWizardButtonTests(unittest.TestCase):
    def test_project_storage_handlers_stay_inside_settings_modal_scope(self):
        for handler in (
            "chooseProjectRootButton.onclick",
            "saveProjectRootButton.onclick",
            "clearProjectRootButton.onclick",
        ):
            self.assertIn(handler, SETTINGS_SOURCE)
            self.assertNotIn(handler, WIZARD_SOURCE)

    def test_wizard_button_surfaces_opening_errors(self):
        self.assertIn('const wizardButton = makeButton("Wizard", "primary");', BUILDER_SOURCE)
        self.assertIn('lines: ["Wizard"]', BUILDER_SOURCE)
        self.assertIn("wizardButton.onclick = () => {", BUILDER_SOURCE)
        self.assertIn("openWizardFromBuilder();", BUILDER_SOURCE)
        self.assertIn(
            'console.error("[VRGDG] Wizard failed to open", error)',
            BUILDER_SOURCE,
        )
        self.assertIn("Wizard failed to open:", BUILDER_SOURCE)

    def test_only_current_wizard_is_available(self):
        self.assertEqual(BUILDER_SOURCE.count('const wizardButton = makeButton("Wizard", "primary");'), 1)
        for legacy in ("Wizard Legacy", "Wizard Beta", "wizardBetaButton", "openMusicVideoWizard", '"./wizard.mjs"'):
            self.assertNotIn(legacy, BUILDER_SOURCE)
        self.assertFalse((ROOT / "web/music_video_builder/wizard.mjs").exists())
        self.assertFalse((ROOT / "web/VRGDG_WizardBeta.js").exists())
        self.assertTrue((ROOT / "web/VRGDG_Wizard.js").exists())

    def test_legacy_draft_routes_and_storage_helpers_are_removed(self):
        routes = (ROOT / "builder/routes.py").read_text(encoding="utf-8")
        project = (ROOT / "builder/project.py").read_text(encoding="utf-8")
        for name in ("save_wizard_draft", "load_wizard_draft"):
            self.assertNotIn(name, routes)
            self.assertNotIn(name, project)
        # Location generation is shared by the Ingredients Builder.
        self.assertIn("/vrgdg/music_builder/wizard_locations_from_lyrics", routes)


if __name__ == "__main__":
    unittest.main()
