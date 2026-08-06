import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(
    encoding="utf-8"
)

SETTINGS_START = BUILDER_SOURCE.index("function openSettingsModal()")
SETTINGS_END = BUILDER_SOURCE.index("async function importPromptJson", SETTINGS_START)
SETTINGS_SOURCE = BUILDER_SOURCE[SETTINGS_START:SETTINGS_END]

WIZARD_START = BUILDER_SOURCE.index("function openWizardFromBuilder()")
WIZARD_END = BUILDER_SOURCE.index("for (const control of [labelInput", WIZARD_START)
WIZARD_SOURCE = BUILDER_SOURCE[WIZARD_START:WIZARD_END]


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
        self.assertIn("wizardButton.onclick = () => {", BUILDER_SOURCE)
        self.assertIn("openWizardFromBuilder();", BUILDER_SOURCE)
        self.assertIn(
            'console.error("VRGDG Video Wizard failed to open", error)',
            BUILDER_SOURCE,
        )
        self.assertIn("Video Wizard failed to open:", BUILDER_SOURCE)


if __name__ == "__main__":
    unittest.main()
