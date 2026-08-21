import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")


def function_body(name):
    match = re.search(
        rf"  function {re.escape(name)}\(segment\) \{{(?P<body>.*?)(?=\n  function )",
        UI,
        flags=re.DOTALL,
    )
    if not match:
        raise AssertionError(f"Function not found: {name}")
    return match.group("body")


class BuilderNullPreviewGuardTests(unittest.TestCase):
    def test_null_preview_call_is_supported_by_image_source(self):
        self.assertIn("syncPreview(null);", UI)
        body = function_body("segmentImageSource")
        self.assertIn("if (!segment) return null;", body)
        self.assertLess(body.index("if (!segment) return null;"), body.index("segment.image_history"))

    def test_null_segment_has_no_thumbnail_path(self):
        body = function_body("selectedSegmentImageThumbnailPath")
        self.assertIn('if (!segment) return "";', body)
        self.assertLess(body.index('if (!segment) return "";'), body.index("segment.image_history"))

    def test_engine_switch_restores_an_active_scene_before_panel_sync(self):
        match = re.search(
            r"  function syncProjectVideoEngineUI\(\) \{(?P<body>.*?)(?=\n  (?:async )?function )",
            UI,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        body = match.group("body")
        self.assertIn("if (!activeSegment())", body)
        self.assertIn('state.activeId = state.segments[0]?.id || state.overlaySegments[0]?.id || "";', body)
        self.assertLess(body.index("if (!activeSegment())"), body.index("syncMiniMaxH3Panel();"))


if __name__ == "__main__":
    unittest.main()
