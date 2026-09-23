import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")


class LyricReviewMoveLastWordTests(unittest.TestCase):
    def test_every_review_row_has_the_move_button(self):
        self.assertIn('makeButton("Move last word to start of next scene")', SOURCE)
        self.assertIn('moveLastWord.disabled = index >= scenes.length - 1', SOURCE)
        self.assertIn('moveLastWord.onclick = () => moveLastWordToNextReviewRow(row)', SOURCE)

    def test_only_current_and_next_row_are_changed(self):
        start = SOURCE.index("const moveLastWordToNextReviewRow")
        end = SOURCE.index("const moveWordAcrossStructuredLyricRows", start)
        helper = SOURCE[start:end]
        self.assertIn("const nextRow = rows[currentIndex + 1] || null", helper)
        self.assertIn("setReviewRowRawLyricText(currentRow", helper)
        self.assertIn("setReviewRowRawLyricText(nextRow", helper)
        self.assertNotIn("for (const row of rows", helper)

    def test_save_updates_embedded_cues_and_all_note_files(self):
        self.assertIn("moveWordAcrossStructuredLyricRows(source.lyric_cue_map, target.lyric_cue_map", SOURCE)
        self.assertIn("moveWordAcrossStructuredLyricRows(source.minimax_speaker_assignments, target.minimax_speaker_assignments", SOURCE)
        self.assertIn("moveWordAcrossStructuredLyricRows(source.speaker_assignments, target.speaker_assignments", SOURCE)
        self.assertIn("moveWordAcrossStructuredLyricRows(source.dialogue_cues, target.dialogue_cues", SOURCE)
        save_start = SOURCE.index("const saveReviewChanges =", SOURCE.index("function openLyricReviewModal"))
        save_end = SOURCE.index("function openLyricMappingWorkflowModal", save_start)
        save_source = SOURCE[save_start:save_end]
        self.assertLess(save_source.index("applyPendingReviewWordMoves"), save_source.index("saveSession({ quiet: true, throwOnError: true })"))
        self.assertIn('await syncLyricAndSubjectNoteFiles("session save")', SOURCE)


if __name__ == "__main__":
    unittest.main()
