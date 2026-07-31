import unittest
from pathlib import Path


BUILDER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "web"
    / "VRGDG_MusicVideoBuilderUI.js"
).read_text(encoding="utf-8")


class BuilderMenuScrollTests(unittest.TestCase):
    def test_main_menu_is_limited_to_the_viewport_and_scrollable(self):
        self.assertIn("max-height:min(760px,calc(100vh - 88px))", BUILDER_SOURCE)
        self.assertIn("overflow-y:auto", BUILDER_SOURCE)
        self.assertIn("scrollbar-gutter:stable", BUILDER_SOURCE)


if __name__ == "__main__":
    unittest.main()
