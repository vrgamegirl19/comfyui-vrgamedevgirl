import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STORYBOARD_SOURCE = (
    ROOT / "web" / "VRGDG_StoryboardBuilderUI.js"
).read_text(encoding="utf-8")
BUILDER_SOURCE = (
    ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js"
).read_text(encoding="utf-8")


class StoryboardSubjectPickerTests(unittest.TestCase):
    def test_subject_button_opens_multi_subject_picker(self):
        self.assertIn("const openStoryboardSubjectPicker = (scene) =>", STORYBOARD_SOURCE)
        self.assertIn("const selected = new Set(", STORYBOARD_SOURCE)
        self.assertIn("scene.subject_refs = selectedSubjects;", STORYBOARD_SOURCE)
        self.assertIn(
            'openStoryboardSubjectPicker(scene)',
            STORYBOARD_SOURCE,
        )

    def test_imported_references_merge_with_existing_references(self):
        self.assertIn(
            "if (incomingRefs.subjects.length) {",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "if (incomingRefs.locations.length && !refs.locations_cleared) {",
            BUILDER_SOURCE,
        )
        self.assertNotIn(
            "if (incomingRefs.subjects.length && !(refs.subjects || []).length) {",
            BUILDER_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
