import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
RUNNER_SOURCE = (ROOT / "VRGDG_WorkflowRunnerNodes.py").read_text(encoding="utf-8")


class MiniMaxImageReferenceOptionalTests(unittest.TestCase):
    def test_mode_is_presented_as_i2v_with_optional_references(self):
        self.assertIn('makeButton("Image to Video\\n2 Pass")', BUILDER_SOURCE)
        self.assertIn("Reference Builder images are optional", BUILDER_SOURCE)
        self.assertIn('imageReferenceTwoPass ? "exact_start_frame"', BUILDER_SOURCE)
        self.assertIn('miniMaxSceneImageUse.disabled = !segment || imageReferenceTwoPass;', BUILDER_SOURCE)
        self.assertIn('mode === "image_reference_to_video" && !sceneImageSourceAvailable', BUILDER_SOURCE)

    def test_scene_image_controls_accept_both_reference_modes(self):
        eligibility = '["reference_to_video", "image_reference_to_video"].includes(miniMaxH3ModeForSegment(item))'
        self.assertGreaterEqual(BUILDER_SOURCE.count(eligibility), 2)
        self.assertNotIn('miniMaxH3ModeForSegment(item) === "reference_to_video"', BUILDER_SOURCE)

    def test_backend_requires_start_frame_but_not_supporting_references(self):
        self.assertIn('combined_images = [start_frame_path] + ([last_frame_path] if last_frame_path else []) + image_paths', RUNNER_SOURCE)
        self.assertNotIn("Image + Reference two-pass requires at least one Reference Builder image", RUNNER_SOURCE)
        self.assertIn('for ref_index, loader_slot in enumerate(range(ref_start, len(paths))):', RUNNER_SOURCE)


if __name__ == "__main__":
    unittest.main()
