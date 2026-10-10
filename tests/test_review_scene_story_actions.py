"""Review Lines + Map Performers can replace one scene's scene beat and LLM prompt through the Story Builder."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
REVIEW = (WEB / "music_video_builder" / "lyric_review.mjs").read_text(encoding="utf-8")
BRIDGE = (WEB / "music_video_builder" / "storyboard_bridge.mjs").read_text(encoding="utf-8")
BUILDER = (WEB / "music_video_builder" / "builder.mjs").read_text(encoding="utf-8")
STORYBOARD = (WEB / "storyboard_builder" / "storyboard.mjs").read_text(encoding="utf-8")


class ReviewSceneStoryActionTests(unittest.TestCase):
    def test_every_review_row_has_both_buttons_and_saves_first(self):
        self.assertIn('makeButton("Replace Scene Beat")', REVIEW)
        self.assertIn('makeButton("Replace LLM Prompt")', REVIEW)
        # The review edits are saved before the Story Builder runs, and a failed save stops the run.
        self.assertIn("if (!(await saveReviewChanges(true)) || !backdrop.isConnected) return;", REVIEW)
        self.assertIn("sceneActions: { sceneId: live.id, beat, prompt }", REVIEW)

    def test_the_review_window_gets_the_story_builder_opener(self):
        self.assertIn("openStoryboardBuilderFromProject: (...args) => openStoryboardBuilderFromProject(...args)", BUILDER)

    def test_a_single_scene_run_writes_back_through_that_scene_only(self):
        self.assertIn("sceneActions: options.sceneActions,", BRIDGE)
        # The all-scenes sync is not used for a single-scene run.
        self.assertIn("onFocusedSave: options.sceneActions ? null : async (updates) => {", BRIDGE)
        self.assertIn("state.selected = new Set([scene.id]);", STORYBOARD)
        self.assertIn("if (state.onSceneChanged) await state.onSceneChanged(slimSceneForRequest(scene, sceneIndex));", STORYBOARD)

    def test_the_story_builder_run_stays_hidden_and_closes(self):
        self.assertIn('backdrop.dataset.vrgdgSceneActions = sceneActions.sceneId;', STORYBOARD)
        self.assertIn("await replaceSingleSceneStory(sceneActions);", STORYBOARD)
        self.assertIn("closeStoryboard();", STORYBOARD)

    def test_changed_scenes_turn_both_buttons_red_until_replaced(self):
        # Edits to a scene's line, timing, people, performers, flags, facial performance or location make both buttons red.
        # Red means "differs from the snapshot taken at the last replace", so undoing an edit clears it.
        self.assertIn("const storyBaselines = new Map();", REVIEW)
        self.assertIn("entry[kind] !== undefined && entry[kind] !== rowStorySnapshot(row)", REVIEW)
        self.assertIn("if (beat) entry.beat = replacedSnapshot;", REVIEW)
        self.assertIn("if (prompt) entry.llm = replacedSnapshot;", REVIEW)
        self.assertIn("background:#b91c1c;border-color:#ef4444", REVIEW)
        # A button goes back to normal only when the Story Builder reports that its replace worked.
        self.assertIn("onSceneActionsDone: (ok) => { replaced = Boolean(ok); },", REVIEW)
        self.assertIn("payload.onSceneActionsDone?.(replaced);", STORYBOARD)
        self.assertIn("onSceneActionsDone: options.onSceneActionsDone,", BRIDGE)

    def test_saving_with_unreplaced_changes_warns_and_cancel_keeps_the_window(self):
        self.assertIn("Changes will not be applied to renders unless Replace Scene Beat and Replace LLM Prompt are run", REVIEW)
        self.assertIn("OK saves your changes and closes. Cancel keeps this window open.", REVIEW)
        self.assertIn("if (!(await confirmUnreplacedStory())) return;", REVIEW)
        self.assertIn("if (await saveReviewChanges(true) && backdrop.isConnected) closeModal();", REVIEW)


if __name__ == "__main__":
    unittest.main()
