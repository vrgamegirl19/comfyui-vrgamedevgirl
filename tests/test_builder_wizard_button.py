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
        self.assertIn('const wizardButton = makeButton("Wizard Legacy", "primary");', BUILDER_SOURCE)
        self.assertIn('lines: ["Wizard", "Legacy"]', BUILDER_SOURCE)
        self.assertIn("wizardButton.onclick = () => {", BUILDER_SOURCE)
        self.assertIn("openWizardFromBuilder();", BUILDER_SOURCE)
        self.assertIn(
            'console.error("VRGDG Video Wizard failed to open", error)',
            BUILDER_SOURCE,
        )
        self.assertIn("Video Wizard failed to open:", BUILDER_SOURCE)


if __name__ == "__main__":
    unittest.main()
