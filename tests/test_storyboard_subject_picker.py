import unittest
from pathlib import Path

from builder_source import read_builder_source, read_storyboard_source


ROOT = Path(__file__).resolve().parents[1]
STORYBOARD_SOURCE = read_storyboard_source()
BUILDER_SOURCE = read_builder_source()


class StoryboardSubjectPickerTests(unittest.TestCase):
    def test_subject_button_opens_multi_subject_picker(self):
        self.assertIn("function openStoryboardSubjectPicker(scene) {", STORYBOARD_SOURCE)
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
