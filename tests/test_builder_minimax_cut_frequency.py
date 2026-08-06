import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STORYBOARD_SOURCE = (ROOT / "web" / "VRGDG_StoryboardBuilderUI.js").read_text(
    encoding="utf-8"
)
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(
    encoding="utf-8"
)
INSTRUCTION_SOURCE = (ROOT / "VRGDG_MiniMaxH3PromptInstructions.py").read_text(
    encoding="utf-8"
)


class BuilderMiniMaxCutFrequencyTests(unittest.TestCase):
    def test_scene_defaults_exposes_minimax_only_zero_to_ten_slider(self):
        self.assertIn('cutFrequencyLabel.textContent = "Cut frequency"', STORYBOARD_SOURCE)
        self.assertIn('cutFrequencyInput.min = "0"', STORYBOARD_SOURCE)
        self.assertIn('cutFrequencyInput.max = "10"', STORYBOARD_SOURCE)
        self.assertIn(
            'const cutFrequencyEligible = isVideoPrepMode && state.projectVideoEngine === "minimax_h3"',
            STORYBOARD_SOURCE,
        )

    def test_cut_plan_scales_against_exact_segment_duration(self):
        self.assertIn(
            "export function storyboardCutPlanForDuration(durationValue, frequencyValue)",
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
            "const cutPlan = storyboardCutPlanForDuration(duration, state.builderStoryboardDefaults?.minimax_h3_cut_frequency)",
            BUILDER_SOURCE,
        )
        self.assertIn("cutPlan.instruction,", BUILDER_SOURCE)
        self.assertIn(
            "cut_plan: cutPlan",
            BUILDER_SOURCE,
        )

    def test_permanent_minimax_instructions_make_cut_plan_authoritative(self):
        self.assertIn("EDITING / CUT PLAN AUTHORITY", INSTRUCTION_SOURCE)
        self.assertIn(
            "It overrides TIMELINE DENSITY",
            INSTRUCTION_SOURCE,
        )
        self.assertIn(
            "never write `CUT TO:`",
            INSTRUCTION_SOURCE,
        )
        self.assertIn(
            "begin its visual description with the literal words `CUT TO:`",
            INSTRUCTION_SOURCE,
        )
        self.assertIn(
            "Do not omit, merge, add, or shift a scheduled cut.",
            INSTRUCTION_SOURCE,
        )
        self.assertIn(
            "A supplied `EDITING / CUT PLAN — MANDATORY` contract is also locked",
            INSTRUCTION_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
