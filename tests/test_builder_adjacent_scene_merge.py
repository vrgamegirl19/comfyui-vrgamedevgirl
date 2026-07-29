import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js"


class BuilderAdjacentSceneMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = BUILDER_SOURCE.read_text(encoding="utf-8")
        start = cls.source.index("  async function mergeAdjacentBaseScene(")
        end = cls.source.index("\n  function openSegmentContextMenu(", start)
        cls.merge_source = cls.source[start:end]

    def test_context_menu_offers_left_and_right_merge(self):
        self.assertIn('"Merge with scene on left"', self.source)
        self.assertIn('"Merge with scene on right"', self.source)
        self.assertIn('mergeAdjacentBaseScene(segment, "left")', self.source)
        self.assertIn('mergeAdjacentBaseScene(segment, "right")', self.source)

    def test_merge_keeps_adjacent_timeline_order_and_combines_lyrics(self):
        self.assertIn('const leftScene = direction === "left" ? neighbor : selected;', self.merge_source)
        self.assertIn('const rightScene = direction === "left" ? selected : neighbor;', self.merge_source)
        self.assertIn("mergeTimestampedLyricText(leftLyric, rightLyric", self.merge_source)
        self.assertIn(
            "state.segments = state.segments.filter((scene) => String(scene.id) !== String(rightScene.id));",
            self.merge_source,
        )

    def test_merge_does_not_ripple_other_scene_timings(self):
        self.assertIn("leftScene.start = Math.min(", self.merge_source)
        self.assertIn("leftScene.end = Math.max(", self.merge_source)
        self.assertNotIn("closeBaseTimelineGap", self.merge_source)
        self.assertNotIn("shiftTimeline", self.merge_source)
        self.assertNotIn("state.duration =", self.merge_source)
        self.assertNotIn("state.timingFrozen", self.merge_source)

    def test_merge_migrates_scene_mappings_and_renumbers_generic_labels(self):
        self.assertIn("migrateSceneMappingsAfterMerge(", self.merge_source)
        self.assertIn("renumberGenericBaseSceneLabelsAfterMerge();", self.merge_source)
        self.assertIn("delete idLoraBuilder.scene_map[removedId];", self.source)
        self.assertIn('["scene_map", "scene_trigger_map", "ingredients_scene_map"]', self.source)


if __name__ == "__main__":
    unittest.main()
