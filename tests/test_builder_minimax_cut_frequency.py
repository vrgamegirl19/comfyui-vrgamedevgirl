import unittest
from pathlib import Path

from builder_source import read_builder_source, read_storyboard_source


ROOT = Path(__file__).resolve().parents[1]
STORYBOARD_SOURCE = read_storyboard_source()
BUILDER_SOURCE = read_builder_source()
INSTRUCTION_SOURCE = (ROOT / "llm" / "prompts" / "minimax.py").read_text(
    encoding="utf-8"
)


class BuilderMiniMaxCutFrequencyTests(unittest.TestCase):
    def test_scene_defaults_exposes_zero_to_ten_slider_in_video_prep(self):
        self.assertIn('cutFrequencyLabel.textContent = "Cut frequency"', STORYBOARD_SOURCE)
        self.assertIn('cutFrequencyInput.min = "0"', STORYBOARD_SOURCE)
        self.assertIn('cutFrequencyInput.max = "10"', STORYBOARD_SOURCE)
        self.assertIn(
            "const cutFrequencyEligible = isVideoPrepMode;",
            STORYBOARD_SOURCE,
        )

    def test_cut_plan_scales_against_exact_segment_duration(self):
        self.assertIn(
            'function storyboardCutPlanForDuration(durationValue, frequencyValue, engineValue = "minimax_h3")',
            STORYBOARD_SOURCE,
        )
        self.assertIn(
            "const maximumCuts = Math.max(0, Math.ceil(Math.max(0, duration - 0.000001)) - 1)",
            STORYBOARD_SOURCE,
        )
        self.assertIn(
            "frequency >= 10\n      ? maximumCuts",
            STORYBOARD_SOURCE,
        )
        self.assertIn(
            "Array.from({ length: cutCount }, (_, index) => index + 1)",
            STORYBOARD_SOURCE,
        )

    def test_zero_is_continuous_and_active_plans_require_explicit_cut_to(self):
        self.assertIn(
            "Use one smooth, continuous, uninterrupted shot",
            STORYBOARD_SOURCE,
        )
        self.assertIn(
            "write an explicit new timestamp block beginning with CUT TO:",
            STORYBOARD_SOURCE,
        )

    def test_saved_default_reaches_all_minimax_prompt_creation_paths(self):
        self.assertIn(
            "minimax_h3_cut_frequency: cutFrequency",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "storyboardCutPlanForDuration(duration, state.builderStoryboardDefaults?.minimax_h3_cut_frequency)",
            BUILDER_SOURCE,
        )
        self.assertIn("cutPlan.instruction,", BUILDER_SOURCE)
        self.assertIn(
            "cut_plan: cutPlan",
            BUILDER_SOURCE,
        )

    def test_minimax_cut_plan_contract_is_authoritative(self):
        self.assertIn(
            "Do not omit, merge, add, or shift a scheduled cut.",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "[Shot ${index + 2}] At ${miniMaxH3Timecode(time)}, begin a new continuity-preserving cut.",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "A supplied `EDITING / CUT PLAN — MANDATORY` contract is also locked",
            INSTRUCTION_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
