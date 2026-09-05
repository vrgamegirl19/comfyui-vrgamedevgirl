import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")


class MiniMaxReferenceConditioningVisibilityTests(unittest.TestCase):
    def test_generic_reference_sizing_is_only_visible_for_single_pass_reference_modes(self):
        self.assertIn(
            'miniMaxReferenceConditioningSettings.style.display = !hideMultiPassIgnoredSettings',
            BUILDER_SOURCE,
        )
        self.assertIn(
            '&& ["reference_to_video", "video_to_video"].includes(mode)',
            BUILDER_SOURCE,
        )

    def test_two_pass_modes_keep_their_dedicated_reference_sizing_controls(self):
        self.assertIn(
            'makeField("Reference image sizing", miniMaxTwoPassRefImageSize',
            BUILDER_SOURCE,
        )
        self.assertIn(
            'makeField("Reference image sizing", miniMaxThreePassRefImageSize',
            BUILDER_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
